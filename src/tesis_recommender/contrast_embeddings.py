"""Embeddings de lotes auditados, con almacén de textos reanudable y modelo fijado."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

from .embedding_cache import build_embedding_cache
from .io import read_awards, read_opportunities
from .semantic import LaBSEEmbedder
from .validation import sha256


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class CheckedLaBSE(LaBSEEmbedder):
    """No cambia la fragmentación. Detecta si al recodificar se perderían tokens."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        lengths = self.token_lengths(texts)
        if any(length > self.model_max_seq_length for length in lengths):
            raise ValueError("Un fragmento recodificado supera el límite del modelo. Revisar el procedimiento antes de generar vectores truncados")
        self.chunks_checked = getattr(self, "chunks_checked", 0) + len(texts)
        self.maximum_chunk_tokens_checked = max(getattr(self, "maximum_chunk_tokens_checked", 0), max(lengths, default=0))
        return super().encode(texts)


class DocumentVectorStore:
    """Persiste bloques float32 y comprueba modelo, estrategia y hashes al reanudar."""

    def __init__(self, root: Path, embedder, fingerprint: dict, documents_per_block: int = 128):
        import numpy as np
        self.np = np
        self.root, self.base = root, embedder
        self.fingerprint, self.documents_per_block = fingerprint, documents_per_block
        if documents_per_block <= 0:
            raise ValueError("El bloque debe tener documentos")
        root.mkdir(parents=True, exist_ok=True)
        self.index_path = root / "source_index.json"
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
            if self.index["fingerprint"] != fingerprint:
                raise ValueError("El almacén pertenece a otro modelo o estrategia")
        else:
            self.index = {"fingerprint": fingerprint, "blocks": {}, "texts": {}}
        self.blocks = {}
        for name, info in self.index["blocks"].items():
            path = root / name
            if sha256(path) != info["sha256"]:
                raise ValueError("Un bloque de embeddings fue modificado")
            values = np.load(path, allow_pickle=False)
            if list(values.shape) != info["shape"] or not np.isfinite(values).all():
                raise ValueError("Dimensiones o valores inválidos en el almacén")
            self.blocks[name] = values
        self.model_name = embedder.model_name
        self.model_revision = getattr(embedder, "model_revision", None)
        self.model_max_seq_length = getattr(embedder, "model_max_seq_length", None)

    def token_lengths(self, texts):
        return self.base.token_lengths(texts)

    def encode(self, texts):
        return self.encode_documents(texts)

    def encode_documents(self, texts):
        missing = list(dict.fromkeys(t for t in texts if _digest(t) not in self.index["texts"]))
        for start in range(0, len(missing), self.documents_per_block):
            batch = missing[start:start + self.documents_per_block]
            vectors = self.base.encode_documents(batch)
            matrix = self.np.asarray(vectors, dtype=self.np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(batch) or not self.np.isfinite(matrix).all():
                raise ValueError("El codificador no devolvió un vector finito por documento")
            norms = self.np.linalg.norm(matrix, axis=1)
            if not self.np.allclose(norms, 1, atol=1e-4):
                raise ValueError("Los vectores deben estar normalizados")
            lengths = self.base.token_lengths(batch)
            document_stats = getattr(self.base, "last_document_encoding_stats", [])
            number = max([int(n.split("_")[-1].split(".")[0]) for n in self.index["blocks"]], default=0) + 1
            name = f"block_{number:05d}.npy"
            path = self.root / name
            while path.exists():
                number += 1
                name = f"block_{number:05d}.npy"
                path = self.root / name
            temporary = path.with_suffix(".npy.part")
            with temporary.open("wb") as handle:
                self.np.save(handle, matrix, allow_pickle=False)
            temporary.replace(path)
            self.index["blocks"][name] = {"sha256": sha256(path), "shape": list(matrix.shape),
                "execution_device": str(getattr(getattr(self.base, "_model", None), "device", "not_recorded")),
                "batch_size": getattr(self.base, "batch_size", None)}
            self.blocks[name] = matrix
            limit = self.model_max_seq_length - 2
            for row, (text, length) in enumerate(zip(batch, lengths)):
                stats = document_stats[row] if len(document_stats) == len(batch) else {"original_tokens": length, "chunks": math.ceil(max(length - 2, 1) / limit)}
                self.index["texts"][_digest(text)] = {"block": name, "row": row, **stats}
            _atomic_json(self.index_path, self.index)
            print(json.dumps({"source_documents_cached": len(self.index["texts"]), "new_documents_encoded": min(start + len(batch), len(missing)), "new_documents_planned": len(missing)}), flush=True)
        entries = [self.index["texts"][_digest(t)] for t in texts]
        self.last_encoding_audit = {"documents": len(texts), "chunks": sum(e["chunks"] for e in entries),
            "chunk_token_limit_without_special_tokens": self.model_max_seq_length - 2,
            "documents_split_into_multiple_chunks": sum(e["chunks"] > 1 for e in entries),
            "maximum_original_tokens_with_special_tokens": max([e["original_tokens"] for e in entries], default=0),
            "chunk_partition_version": "retokenization_checked_v2",
            "chunk_boundary_adjustments": sum(e.get("chunk_boundary_adjustments", 0) for e in entries),
            "source_documents_reused": len(texts) - len(missing), "source_documents_newly_encoded": len(missing),
            "new_chunks_checked_before_encoding": getattr(self.base, "chunks_checked", None),
            "maximum_new_chunk_tokens_checked": getattr(self.base, "maximum_chunk_tokens_checked", None)}
        return [self.blocks[e["block"]][e["row"]].tolist() for e in entries]


def encode_audited_batch(cohort_dir: Path, preparation_dir: Path, store_dir: Path,
                        output_dir: Path, batch_size: int = 8, amendment_path: Path | None = None) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("No sobrescribir un caché final. Reanudar con otra carpeta de salida y el mismo almacén")
    if batch_size <= 0:
        raise ValueError("El batch size debe ser positivo")
    plan_path = preparation_dir / "frozen_plan_with_code.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if amendment_path:
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if amendment["original_plan_sha256"] != sha256(plan_path):
            raise ValueError("La enmienda no corresponde al plan")
        plan = {**plan, "source_sha256": {**plan["source_sha256"], **amendment["updated_source_sha256"]}}
    report_path = cohort_dir / "candidate_validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["input_sha256"][str(plan_path)] != sha256(plan_path):
        raise ValueError("La auditoría no corresponde al plan")
    for name, digest in report["output_sha256"].items():
        if sha256(cohort_dir / name) != digest:
            raise ValueError("Se modificaron entradas auditadas")
    package = Path(__file__).parent
    for name in ("semantic.py", "embedding_cache.py", "profiles.py", "schemas.py", "io.py"):
        if sha256(package / name) != plan["source_sha256"][name]:
            raise ValueError("Cambió un módulo del procedimiento de embeddings congelado")
    embedder = CheckedLaBSE(batch_size=batch_size)
    if embedder.model_name != plan["model_name"] or embedder.model_revision != plan["model_revision"]:
        raise ValueError("El modelo cargado no coincide con la revisión congelada")
    fingerprint = {"model_name": embedder.model_name, "model_revision": embedder.model_revision,
                   "profile_strategy": plan["profile_strategy"], "opportunity_strategy": plan["opportunity_strategy"],
                   "semantic_source_sha256": plan["source_sha256"]["semantic.py"], "max_seq_length": embedder.model_max_seq_length}
    store = DocumentVectorStore(store_dir, embedder, fingerprint)
    awards_path, opportunities_path = cohort_dir / "historical_awards.csv", cohort_dir / "test_opportunities.csv"
    metadata = build_embedding_cache(read_awards(awards_path), read_opportunities(opportunities_path), output_dir,
                                     store, cutoff=date.fromisoformat(plan["cutoff"]), input_paths=[awards_path, opportunities_path, report_path, plan_path])
    metadata.update({"evaluation_ready": report["evaluation_ready"], "candidate_opportunities_planned": report["candidate_opportunities_planned"],
                     "evaluation_status": report["evaluation_status"], "labels_read": False,
                     "source_store_index": str(store.index_path), "source_store_index_sha256": sha256(store.index_path)})
    metadata["embedding_amendment"] = {"path": str(amendment_path), "sha256": sha256(amendment_path)} if amendment_path else None
    import torch
    metadata["execution"] = {"device": str(embedder._model.device), "batch_size": batch_size,
        "torch_num_threads": torch.get_num_threads(), "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "source_blocks_by_device": {device: sum(info["shape"][0] for info in store.index["blocks"].values()
            if info.get("execution_device", "legacy_not_recorded") == device)
            for device in {info.get("execution_device", "legacy_not_recorded") for info in store.index["blocks"].values()}}}
    # El almacén compartido crece entre lotes. Conservar el índice de este lote
    # evita que su procedencia dependa del estado mutable de la siguiente ejecución.
    index_snapshot = output_dir / "source_store_index.json"
    shutil.copyfile(store.index_path, index_snapshot)
    metadata["source_store_index_snapshot"] = str(index_snapshot)
    metadata["source_store_index_snapshot_sha256"] = sha256(index_snapshot)
    archive = output_dir / "code_snapshot"
    archive.mkdir()
    for name in ("contrast_embeddings.py", "semantic.py", "embedding_cache.py", "profiles.py", "schemas.py", "io.py"):
        shutil.copyfile(package / name, archive / name)
    metadata["code_archive_sha256"] = {p.name: sha256(p) for p in archive.iterdir()}
    _atomic_json(output_dir / "metadata.json", metadata)
    return metadata


def record_embedding_amendment(preparation_dir: Path, diagnosis_path: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("No sobrescribir una enmienda")
    original_path = preparation_dir / "frozen_plan_with_code.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    if diagnosis["source_sha256"] != original["source_sha256"]["semantic.py"] or not diagnosis["overflow_chunks"]:
        raise ValueError("El diagnóstico no demuestra el problema del procedimiento congelado")
    package = Path(__file__).parent
    for name in ("hybrid.py", "bm25.py", "profiles.py", "schemas.py", "embedding_cache.py", "metrics.py", "pipeline.py", "io.py"):
        if sha256(package / name) != original["source_sha256"][name]:
            raise ValueError("Cambió otro componente del experimento además de la fragmentación")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "code_snapshot"
    archive.mkdir()
    for name in ("semantic.py", "contrast_embeddings.py"):
        shutil.copyfile(package / name, archive / name)
    amendment = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "original_plan_sha256": sha256(original_path),
        "diagnosis_path": str(diagnosis_path), "diagnosis_sha256": sha256(diagnosis_path),
        "reason": "WordPiece decode/encode can exceed the encoder token limit. Check and reduce chunk boundaries before encoding, with every original token position assigned.",
        "chunk_partition_version": "retokenization_checked_v2", "model_name": original["model_name"], "model_revision": original["model_revision"],
        "ranking_weights_unchanged": original["models"], "labels_read": False, "results_used_to_tune_change": False,
        "updated_source_sha256": {"semantic.py": sha256(package / "semantic.py")},
        "code_archive_sha256": {p.name: sha256(p) for p in archive.iterdir()},
        "comparison_limit": "Old caches without this check remain archived, not silently relabelled. All future contrast variants must use the same corrected embeddings."}
    _atomic_json(output_dir / "amendment.json", amendment)
    return amendment
