"""Persist development candidates from already-inspected archived source pages.

No reference condition, prediction or expected value is given to the extractor.
Historical evaluation artifacts are verified and preserved, not overwritten or
reinterpreted as an independent test of this new development code.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import validate_source_document, verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes
from tesis_recommender.document_experience_dev_v1 import extract_development, verify_provenance

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ARTIFACTS = (
    "experience_holdout_reference_lock_v1", "experience_holdout_inference_plan_v1",
    "experience_holdout_predictions_v1", "experience_holdout_concordance_v1",
)


def build(source_dir, output):
    if output.exists():
        raise ValueError("Use a fresh development directory; preserve all prior outputs")
    inputs = dict(protected_outputs())
    for directory in (source_dir, *(ROOT / "artifacts" / name for name in FROZEN_ARTIFACTS)):
        verify_artifact(directory, inputs)
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    lock_path = ROOT / "artifacts/experience_holdout_reference_lock_v1/reference_lock.json"
    lock = json.loads(lock_path.read_text())
    verify_hashes(lock["frozen_model_code_sha256"])
    verify_hashes(lock["comparison_code_sha256"])
    inputs.update(lock["frozen_model_code_sha256"])
    inputs.update(lock["comparison_code_sha256"])
    code = (Path(__file__).resolve(), ROOT / "src/tesis_recommender/document_experience_dev_v1.py",
            ROOT / "tests/test_document_experience_dev_v1.py")
    for path in code:
        inputs[str(path)] = sha(path)
    documents = []
    # Source-only extraction. Frozen reference values are never loaded here.
    sources = json.loads((source_dir / "source_documents.json").read_text())
    for source in sources:
        validate_source_document(source)
        result = extract_development(source["pages"])
        verify_provenance(result, source["pages"])
        documents.append({"opportunity_id": source["opportunity_id"], "sample_number": source["sample_number"],
            "source_pdf_sha256": source["source"]["sha256"], "physical_pages": len(source["pages"]),
            "source_text_view_unchanged": True, "result": result})
    records = [r for d in documents for r in d["result"]["records"]]
    summary = {"version": "document_experience_development_v1", "stage": "observed_source_development_only",
        "all_documents_previously_inspected": True, "documents_processed": len(documents),
        "physical_pages": sum(d["physical_pages"] for d in documents),
        "candidate_occurrences": len(records),
        "documents_with_candidates": sum(bool(d["result"]["records"]) for d in documents),
        "counts_by_role": dict(Counter(r["section_role"] for r in records)),
        "counts_by_branch": dict(Counter(r["branch"] for r in records)),
        "complete_numeric_tokens": sum(r["fields"]["billed_amount"] is not None for r in records),
        "abstained_absolute_amounts": sum(r["fields"]["billed_amount"] is None for r in records),
        "explicit_item_occurrences": sum(r["item"] is not None for r in records),
        "amount_status_counts": dict(Counter(r["amount_status"] for r in records)),
        "skipped_unsupported_contexts": sum(len(d["result"]["skipped_contexts"]) for d in documents),
        "traceability_verified": True, "original_frozen_evaluation_preserved": True,
        "frozen_model_files_verified": len(lock["frozen_model_code_sha256"]),
        "frozen_comparison_files_verified": len(lock["comparison_code_sha256"]),
        "f1_computed": False, "independent_validation_performed": False,
        "generalization_proven": False, "reference_or_expected_values_used_for_inference": False,
        "automatic_deduplication_performed": False, "new_ocr_performed": False,
        "archived_text_modified": False, "ranking_inputs_modified": False,
        "limitations": ["Centavos check is not a complete Spanish words-to-number validator.",
            "Unsupported headings, role context and formulations remain visible as skipped candidates.",
            "Relative requirements are not converted to absolute amounts or inferred currency.",
            "Repeated source occurrences are retained; no new general consolidation policy is adopted.",
            "The observed holdout is now development data for this version, not an unseen test."]}
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "document_candidates.json", documents)
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "development_code_sha256": {str(path): sha(path) for path in code},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source.resolve(), args.output.resolve())))
