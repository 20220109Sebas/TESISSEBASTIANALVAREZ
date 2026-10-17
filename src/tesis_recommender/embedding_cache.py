from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

from .profiles import build_profiles
from .schemas import Award, Opportunity
from .semantic import Embedder


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _award_text(award: Award) -> str:
    return " ".join(
        value for value in (award.title, award.description, award.technical_text) if value
    ).strip()


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_embedding_cache(
    awards: list[Award],
    opportunities: list[Opportunity],
    output_dir: Path,
    embedder: Embedder,
    cutoff: date = date(2025, 1, 1),
    input_paths: list[Path] | None = None,
) -> dict[str, object]:
    """Materializa LaBSE una sola vez y agrega cada perfil por adjudicación.

    La agregación evita truncar todo el historial empresarial como una única
    secuencia. Cada adjudicación se codifica por separado y después se calcula
    el promedio normalizado de sus embeddings.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("La construcción del caché requiere numpy") from exc

    profiles = build_profiles(awards, cutoff)
    candidates = [
        item
        for item in opportunities
        if item.publication_date >= cutoff
        and item.object_type.lower() == "services"
        and item.text
    ]
    award_texts = [_award_text(award) for profile in profiles.values() for award in profile.awards]
    candidate_texts = [item.text for item in candidates]
    unique_source_texts = list(dict.fromkeys(award_texts + candidate_texts))
    encode_documents = getattr(embedder, "encode_documents", embedder.encode)
    source_vectors = encode_documents(unique_source_texts)
    source_by_hash = {_sha_text(text): vector for text, vector in zip(unique_source_texts, source_vectors)}

    entries: list[dict[str, object]] = []
    final_vectors: list[list[float]] = []
    seen: dict[str, int] = {}

    def add(kind: str, identifier: str, text: str, vector: list[float]) -> None:
        digest = _sha_text(text)
        if digest in seen:
            existing = final_vectors[seen[digest]]
            if any(abs(left - right) > 1e-6 for left, right in zip(existing, vector)):
                raise ValueError("Un mismo texto produjo vectores distintos según su rol")
            row = seen[digest]
        else:
            row = len(final_vectors)
            seen[digest] = row
            final_vectors.append(vector)
        entries.append({"kind": kind, "id": identifier, "text_sha256": digest, "row": row})

    for supplier_id, profile in profiles.items():
        vectors = [source_by_hash[_sha_text(_award_text(award))] for award in profile.awards]
        mean = [sum(values) / len(vectors) for values in zip(*vectors)]
        add("profile", supplier_id, profile.text, _normalize(mean))
    for item in candidates:
        add("opportunity", item.opportunity_id, item.text, source_by_hash[_sha_text(item.text)])

    output_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = output_dir / "embeddings.npy"
    np.save(vectors_path, np.asarray(final_vectors, dtype=np.float32), allow_pickle=False)
    (output_dir / "index.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lengths: list[int] = []
    if hasattr(embedder, "token_lengths"):
        lengths = list(getattr(embedder, "token_lengths")(unique_source_texts))
    max_length = getattr(embedder, "model_max_seq_length", None)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": embedder.model_name,
        "model_revision": getattr(embedder, "model_revision", None),
        "embedding_dimension": len(final_vectors[0]) if final_vectors else 0,
        "normalized_embeddings": True,
        "model_max_seq_length": max_length,
        "profile_strategy": "mean_l2_normalized_of_chunk_aware_individual_award_embeddings",
        "opportunity_strategy": "chunk_aware_title_description_technical_text_embedding",
        "profiles": len(profiles),
        "opportunities": len(candidates),
        "historical_awards": len(award_texts),
        "unique_source_texts": len(unique_source_texts),
        "vector_count": len(final_vectors),
        "token_length_audit": {
            "audited": bool(lengths),
            "maximum_observed": max(lengths) if lengths else None,
            "over_model_limit": sum(value > max_length for value in lengths)
            if lengths and max_length
            else None,
        },
        "chunking_audit": getattr(embedder, "last_encoding_audit", None),
        "package_versions": {
            "sentence-transformers": _package_version("sentence-transformers"),
            "transformers": _package_version("transformers"),
            "torch": _package_version("torch"),
            "numpy": _package_version("numpy"),
        },
        "inputs": [
            {"path": str(path), "sha256": _sha_file(path)} for path in (input_paths or [])
        ],
        "embeddings_sha256": _sha_file(vectors_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
