"""Run heading-context development without replacing any frozen evaluation."""
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import validate_source_document, verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes
from scripts.run_document_experience_development_v1 import FROZEN_ARTIFACTS
from tesis_recommender.document_experience_dev_v2 import VERSION, extract_development, verify_provenance

ROOT = Path(__file__).resolve().parents[1]


def predicate_key(ref):
    span = ref["source_spans"][0]
    return span["page"], span["char_start"]


def build(source_dir, output):
    if output.exists():
        raise ValueError("Use a fresh directory; preserve prior outputs")
    inputs = dict(protected_outputs())
    prior_dir = ROOT / "artifacts/document_experience_development_v1"
    for directory in (source_dir, prior_dir, *(ROOT / "artifacts" / name for name in FROZEN_ARTIFACTS)):
        verify_artifact(directory, inputs)
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    lock = json.loads((ROOT / "artifacts/experience_holdout_reference_lock_v1/reference_lock.json").read_text())
    for mapping in (lock["frozen_model_code_sha256"], lock["comparison_code_sha256"]):
        verify_hashes(mapping)
        inputs.update(mapping)
    code = (Path(__file__).resolve(), ROOT / "src/tesis_recommender/document_experience_dev_v2.py",
            ROOT / "tests/test_document_experience_dev_v2.py")
    inputs.update({str(path): sha(path) for path in code})
    documents = []
    sources = json.loads((source_dir / "source_documents.json").read_text())
    for source in sources:
        validate_source_document(source)
        result = extract_development(source["pages"])
        verify_provenance(result, source["pages"])
        documents.append({"opportunity_id": source["opportunity_id"], "sample_number": source["sample_number"],
            "source_pdf_sha256": source["source"]["sha256"], "physical_pages": len(source["pages"]),
            "source_text_view_unchanged": True, "result": result})
    # Compare source locations only AFTER extraction; never pass v1 output or truth
    # into the extractor. This is change diagnosis, not semantic adjudication.
    prior = json.loads((prior_dir / "document_candidates.json").read_text())
    by_id = {d["opportunity_id"]: d for d in documents}
    review = []
    for old in prior:
        current = by_id[old["opportunity_id"]]["result"]
        found = {predicate_key(r["predicate_evidence"]): r for r in current["records"]}
        skipped = {predicate_key(r["predicate"]): r for r in current["skipped_contexts"]}
        for row in old["result"]["skipped_contexts"]:
            key = predicate_key(row["predicate"])
            recovered = found.get(key)
            if recovered is None and key not in skipped:
                raise ValueError("Prior skipped source occurrence disappeared")
            match = recovered or skipped[key]
            review.append({"opportunity_id": old["opportunity_id"], "sample_number": old["sample_number"],
                "predicate": row["predicate"], "prior_role": row["role"], "prior_subject": row["subject"],
                "status": "recognized_full_line_heading_context" if recovered else "unsupported_role_retained",
                "current_role": match["section_role"] if recovered else match["role"],
                "current_subject": match["subject"], "section_evidence": match["section_evidence"],
                "subject_evidence": match["subject_evidence"],
                "current_prediction_id": recovered["prediction_id"] if recovered else None,
                "semantic_accuracy_adjudicated": False})
    records = [r for d in documents for r in d["result"]["records"]]
    summary = {"version": VERSION, "stage": "observed_source_development_only",
        "documents_processed": len(documents), "physical_pages": sum(d["physical_pages"] for d in documents),
        "candidate_occurrences": len(records), "documents_with_candidates": sum(bool(d["result"]["records"]) for d in documents),
        "complete_numeric_tokens": sum(r["fields"]["billed_amount"] is not None for r in records),
        "abstained_absolute_amounts": sum(r["fields"]["billed_amount"] is None for r in records),
        "counts_by_role": dict(Counter(r["section_role"] for r in records)),
        "counts_by_branch": dict(Counter(r["branch"] for r in records)),
        "prior_skipped_contexts_reviewed": len(review),
        "context_review_counts": dict(Counter(r["status"] for r in review)),
        "skipped_unsupported_contexts": sum(len(d["result"]["skipped_contexts"]) for d in documents),
        "traceability_verified": True, "f1_computed": False, "independent_validation_performed": False,
        "reference_or_expected_values_used_for_inference": False, "archived_text_modified": False,
        "new_ocr_performed": False, "automatic_deduplication_performed": False, "ranking_inputs_modified": False,
        "original_frozen_evaluation_preserved": True, "development_v1_preserved": True,
        "frozen_model_files_verified": len(lock["frozen_model_code_sha256"]),
        "frozen_comparison_files_verified": len(lock["comparison_code_sha256"]),
        "limitations": ["Observed documents are regression data, not an unseen evaluation.",
            "Heading recovery does not guarantee a complete monetary token or lookback expression.",
            "Supplier-profile/TDR contexts remain separate, not silently classified as qualification.",
            "Missing OCR content is not supplied from reference labels or visual-reading notes.",
            "Spanish number words, duplicate policy and new validation remain pending."]}
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "document_candidates.json", documents)
    dump(output / "skipped_context_review.json", review)
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
