from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

from .hybrid import DEFAULT_WEIGHTS, HybridRanker
from .metrics import evaluate_rankings
from .profiles import assert_no_temporal_leakage, build_profiles
from .quality import audit_dataset
from .schemas import Award, Opportunity
from .semantic import Embedder


ABLATIONS = {
    "bm25": {"bm25": 1.0, "semantic": 0.0, "category": 0.0, "region": 0.0, "amount": 0.0},
    "labse": {"bm25": 0.0, "semantic": 1.0, "category": 0.0, "region": 0.0, "amount": 0.0},
    "bm25_labse": {"bm25": 0.412, "semantic": 0.588, "category": 0.0, "region": 0.0, "amount": 0.0},
    "hybrid_full": DEFAULT_WEIGHTS,
}


def _digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_experiment(
    awards: list[Award], opportunities: list[Opportunity], embedder: Embedder, cutoff: date, top_k: int = 10,
    relevant_override: dict[str, set[str]] | None = None,
) -> dict[str, object]:
    profiles = build_profiles(awards, cutoff)
    assert_no_temporal_leakage(profiles)
    candidates = [item for item in opportunities if item.publication_date >= cutoff and item.object_type.lower() == "services" and item.text]
    ranker = HybridRanker(candidates, embedder)
    relevant = relevant_override if relevant_override is not None else {
        supplier_id: {item.opportunity_id for item in candidates if item.awarded_supplier_id == supplier_id}
        for supplier_id in profiles
    }
    candidate_ids = {item.opportunity_id for item in candidates}
    if len(candidate_ids) != len(candidates):
        raise ValueError("Las oportunidades candidatas deben tener identificadores únicos")
    if set(relevant) - set(profiles):
        raise ValueError("Hay etiquetas para perfiles no evaluables")
    if any(items - candidate_ids for items in relevant.values()):
        raise ValueError("Hay positivos fuera de la colección candidata")
    runs: dict[str, object] = {}
    for name, weights in ABLATIONS.items():
        detailed = {supplier_id: ranker.rank(profile, weights, top_k=None) for supplier_id, profile in profiles.items()}
        rankings = {supplier_id: [item.opportunity_id for item in items] for supplier_id, items in detailed.items()}
        display_name = name
        if embedder.model_name == "deterministic-test-embedder-not-for-research":
            display_name = {"labse": "semantic_test", "bm25_labse": "bm25_semantic_test"}.get(name, name)
        runs[display_name] = {
            "metrics": evaluate_rankings(rankings, relevant),
            "top_k": {
                supplier_id: [
                    {"opportunity_id": item.opportunity_id, "score": round(item.score, 6), "source_url": item.source_url, "explanation": item.explanation}
                    for item in items[:top_k]
                ] for supplier_id, items in detailed.items()
            },
        }
    return {
        "model": embedder.model_name,
        "cutoff": cutoff.isoformat(),
        "quality": audit_dataset(awards, opportunities, cutoff),
        "evaluation_note": "Evaluación retrospectiva con positivos observados y candidatos no etiquetados.",
        "protocol_version": "audit-corrected-v2",
        "evaluation_population": {"profiles": len(profiles), "candidates": len(candidates),
                                  "profiles_with_known_positives": sum(bool(relevant.get(p)) for p in profiles),
                                  "profiles_without_known_positives": sum(not relevant.get(p) for p in profiles)},
        "missing_attribute_policy": "Fixed weights; unavailable values contribute zero, not evidence of incompatibility; amount requires identical identified currency.",
        "runs": runs,
    }


def save_experiment(result: dict[str, object], output_dir: Path, input_paths: list[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": _digest(input_paths),
        "inputs": [str(path) for path in input_paths],
        "model": result["model"],
        "cutoff": result["cutoff"],
        "ablations": ABLATIONS,
        "seed": 42,
        "protocol_version": result.get("protocol_version"),
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(Path(__file__).parent.glob("*.py"))},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
