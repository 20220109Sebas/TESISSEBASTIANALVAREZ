"""Audit occurrence identity on fixed development output; never compute F1."""
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import validate_source_document, verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes
from tesis_recommender.document_experience_dev_v2 import verify_provenance
from tesis_recommender.document_experience_units_dev_v1 import VERSION, project_units

ROOT = Path(__file__).resolve().parents[1]


def build(development, source, output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    inputs = dict(protected_outputs())
    for directory in (development, source):
        verify_artifact(directory, inputs)
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    policy = ROOT / "docs/document_experience_occurrence_policy_v1.md"
    code = (Path(__file__).resolve(), ROOT / "src/tesis_recommender/document_experience_units_dev_v1.py",
            ROOT / "tests/test_document_experience_units_dev_v1.py", policy)
    inputs.update({str(path): sha(path) for path in code})
    documents = json.loads((development / "document_candidates.json").read_text())
    archived = json.loads((source / "source_documents.json").read_text())
    by_id = {d["opportunity_id"]: d for d in archived}
    if len(by_id) != len(archived) or {d["opportunity_id"] for d in documents} != set(by_id):
        raise ValueError("Development/source population differs")
    for doc in documents:
        original = by_id[doc["opportunity_id"]]
        validate_source_document(original)
        if doc["source_pdf_sha256"] != original["source"]["sha256"]:
            raise ValueError("Source PDF identity differs")
        verify_provenance(doc["result"], original["pages"])
    result = project_units(documents)
    units = result["candidate_units"]
    summary = {"version": VERSION, "stage": "development_not_new_unseen_evaluation",
        "documents": len(documents), "original_occurrences": len(result["occurrences"]),
        "candidate_identity_groups": len(units),
        "unit_status_counts": dict(Counter(u["status"] for u in units)),
        "blocked_reason_counts": dict(Counter(reason for u in units for reason in u["reasons"])),
        "identity_gate_passed": result["identity_gate"]["passed"],
        "all_occurrences_preserved": True, "source_provenance_verified": True,
        "prior_user_resolution_generalized": False, "frozen_evaluation_modified": False,
        "reference_used_for_identity_or_fields": False, "new_sample_selected": False,
        "metrics_computed": False, "ranking_inputs_modified": False,
        "policy_status": "development_candidate_must_be_included_in_future_protocol_before_new_sample"}
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "occurrence_projection.json", result)
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.development.resolve(), args.source.resolve(), args.output.resolve())))
