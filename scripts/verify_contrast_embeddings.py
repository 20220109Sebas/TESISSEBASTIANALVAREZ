"""Comprueba un caché de contraste sin consultar adjudicaciones futuras."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from tesis_recommender.io import read_awards, read_opportunities
from tesis_recommender.profiles import build_profiles
from tesis_recommender.semantic import PrecomputedEmbedder
from tesis_recommender.validation import sha256


def verify(cache: Path, cohort: Path, store: Path, previous: Path, prior_source_index: Path | None = None) -> dict:
    metadata = json.loads((cache / "metadata.json").read_text())
    for item in metadata["inputs"]:
        assert sha256(Path(item["path"])) == item["sha256"], item["path"]
    for name, digest in metadata["code_archive_sha256"].items():
        assert sha256(cache / "code_snapshot" / name) == digest, name
    amendment = metadata["embedding_amendment"]
    assert sha256(Path(amendment["path"])) == amendment["sha256"]
    snapshot = Path(metadata["source_store_index_snapshot"])
    assert sha256(snapshot) == metadata["source_store_index_snapshot_sha256"]
    sources = json.loads(snapshot.read_text())
    blocks = {}
    for name, info in sources["blocks"].items():
        assert sha256(store / name) == info["sha256"], name
        values = np.load(store / name, allow_pickle=False)
        assert list(values.shape) == info["shape"]
        assert values.dtype == np.float32 and np.isfinite(values).all()
        assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5)
        blocks[name] = values
    for digest, item in sources["texts"].items():
        assert len(digest) == 64
        assert item["original_token_ids_covered"] == item["original_tokens"] - 2
        assert 0 <= item["row"] < len(blocks[item["block"]])
    prior_sources_checked = 0
    if prior_source_index:
        prior_sources = json.loads(prior_source_index.read_text())
        assert sources["fingerprint"] == prior_sources["fingerprint"]
        for name, info in prior_sources["blocks"].items():
            assert sources["blocks"][name] == info
        for digest, item in prior_sources["texts"].items():
            assert sources["texts"][digest] == item
        prior_sources_checked = len(prior_sources["texts"])

    profiles = build_profiles(read_awards(cohort / "historical_awards.csv"), date(2025, 1, 1))
    opportunities = read_opportunities(cohort / "test_opportunities.csv")
    expected = [("profile", identifier, profile.text) for identifier, profile in profiles.items()]
    expected += [("opportunity", item.opportunity_id, item.text) for item in opportunities]
    index = json.loads((cache / "index.json").read_text())
    text_digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    assert [(item["kind"], item["id"], item["text_sha256"]) for item in index] == [
        (kind, identifier, text_digest(text)) for kind, identifier, text in expected]
    matrix = np.load(cache / "embeddings.npy", allow_pickle=False)
    assert matrix.shape == (metadata["vector_count"], 768)
    assert matrix.dtype == np.float32 and np.isfinite(matrix).all()
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5)
    cached = PrecomputedEmbedder(cache)
    resolved = np.asarray(cached.encode([text for _, _, text in expected]))
    assert np.allclose(resolved, matrix[[item["row"] for item in index]], atol=1e-7)
    # Recalcular únicamente la agregación, sin volver a ejecutar LaBSE.
    for row, profile in enumerate(profiles.values()):
        vectors = []
        for award in profile.awards:
            text = " ".join(v for v in (award.title, award.description, award.technical_text) if v).strip()
            source = sources["texts"][text_digest(text)]
            vectors.append(blocks[source["block"]][source["row"]])
        mean = np.asarray(vectors, dtype=np.float64).mean(axis=0)
        mean /= np.linalg.norm(mean)
        assert np.allclose(mean, resolved[row], atol=1e-6)
    for row, opportunity in enumerate(opportunities, start=len(profiles)):
        source = sources["texts"][text_digest(opportunity.text)]
        assert np.allclose(resolved[row], blocks[source["block"]][source["row"]], atol=1e-7)

    prior_index = json.loads((previous / "index.json").read_text())
    prior_matrix = np.load(previous / "embeddings.npy", allow_pickle=False)
    current = {(item["kind"], item["id"]): item for item in index}
    for item in prior_index:
        other = current[item["kind"], item["id"]]
        assert other["text_sha256"] == item["text_sha256"]
        assert np.array_equal(prior_matrix[item["row"]], matrix[other["row"]])
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "cache": str(cache),
            "profiles": len(profiles), "opportunities": len(opportunities), "unique_vectors": len(matrix),
            "embedding_dimension": matrix.shape[1], "source_texts": len(sources["texts"]),
            "source_blocks_checked": len(blocks), "prior_entries_bitwise_unchanged": len(prior_index),
            "prior_source_documents_hash_and_index_unchanged": prior_sources_checked,
            "finite_and_l2_normalized": True, "all_original_token_positions_assigned": True,
            "source_and_code_hashes_checked": True, "source_aggregation_verified": True,
            "chunking_audit": metadata["chunking_audit"], "labels_read": False,
            "evaluation_ready": metadata["evaluation_ready"],
            "verification_source_sha256": sha256(Path(__file__)),
            "output_sha256": {name: sha256(cache / name) for name in
                              ("metadata.json", "index.json", "embeddings.npy", "source_store_index.json")}}


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Ejecutar sin -O para mantener activas las comprobaciones")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--prior-source-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("No sobrescribir una verificación anterior")
    report = verify(args.cache, args.cohort, args.store, args.previous, args.prior_source_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    code_copy = args.output.with_suffix(".code.py")
    if code_copy.exists():
        raise ValueError("No sobrescribir una copia de verificación anterior")
    shutil.copyfile(Path(__file__), code_copy)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
