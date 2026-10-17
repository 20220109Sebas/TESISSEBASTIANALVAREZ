"""Sanity replay of a locked candidate on OBSERVED source pages, never new F1."""
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.freeze_document_experience_candidate_v1 import validate_lock
from scripts.register_browser_experience_sources import dump, sha, verify_hashes
from tesis_recommender.document_experience_candidate_v1 import predict_documents

ROOT = Path(__file__).resolve().parents[1]


def verify(lock_dir, output):
    if output.exists():
        raise ValueError("Use a new observed regression directory")
    validate_lock(lock_dir)
    inputs = {}
    verify_artifact(lock_dir, inputs)
    inputs.update(json.loads((lock_dir / "manifest.json").read_text())["input_sha256"])
    source_dir = ROOT / "artifacts/experience_holdout_source_view_v1"
    verify_artifact(source_dir, inputs)
    inputs.update(json.loads((source_dir / "manifest.json").read_text())["input_sha256"])
    documents = json.loads((source_dir / "source_documents.json").read_text())
    result = predict_documents(documents)
    units = result["occurrence_projection"]
    summary = {"stage": "locked_candidate_observed_source_regression_only",
        "documents": len(documents), "original_occurrences": len(units["occurrences"]),
        "candidate_identity_groups": len(units["candidate_units"]),
        "unit_status_counts": dict(Counter(u["status"] for u in units["candidate_units"])),
        "identity_gate_passed": units["identity_gate"]["passed"],
        "unresolved_ocr_pages": sum(len(c["unresolved_ocr_pages"]) for c in result["coverage"]),
        "all_documents_previously_inspected": True, "new_sample_inference_performed": False,
        "metrics_computed": False, "new_ocr_performed": False, "ranking_inputs_modified": False}
    validate_lock(lock_dir)
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "candidate_outputs.json", result)
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.lock.resolve(), args.output.resolve())))
