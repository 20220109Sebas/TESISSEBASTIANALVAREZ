"""Check source lineage and freeze the comparator before any holdout predictions.

This audit does not complete a reference review or calculate model accuracy.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]


def validate_source_document(doc):
    pages = doc["pages"]
    if [p["page"] for p in pages] != list(range(1, len(pages) + 1)):
        raise ValueError("Missing, reordered or repeated physical pages")
    if doc.get("ranking_eligible") is not False or doc.get("historical_content_proven") is not False:
        raise ValueError("Document is unexpectedly eligible for historical ranking")
    for page in pages:
        for value, checksum in ((page["text"], page["text_sha256"]),
                                (page["native_text"], page["native_text_sha256"])):
            if hashlib.sha256(value.encode()).hexdigest() != checksum:
                raise ValueError("Page text differs from its hash")
        if not page["ocr_selected"] and (page["text"] != page["native_text"] or page["ocr_used"]):
            raise ValueError("Unselected page lost native text")
        if page["ocr_used"] and (not page["ocr_selected"] or page["selected_ocr_unresolved"]):
            raise ValueError("Inconsistent OCR status")
        if page["selected_ocr_unresolved"] and (not page["ocr_selected"] or page["ocr_used"]):
            raise ValueError("Inconsistent unresolved OCR status")


def verify_artifact(directory, inputs):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    verify_hashes(manifest["input_sha256"])
    outputs = {str(directory / p): h for p, h in manifest["output_sha256"].items()}
    verify_hashes(outputs)
    inputs.update(outputs)
    inputs[str(path)] = sha(path)


def audit(packet_dir, source_dir, draft_dir, protocol_path, output):
    if output.exists():
        raise ValueError("Preserve prior audit, use a new output directory")
    protected = protected_outputs()
    inputs = dict(protected)
    for directory in (packet_dir, source_dir, draft_dir):
        verify_artifact(directory, inputs)
    model_lock_path = ROOT / "artifacts/experience_holdout_models_lock_v1/model_lock.json"
    frozen = json.loads(model_lock_path.read_text())["original_code_sha256"]
    verify_hashes(frozen)
    inputs.update(frozen)
    inputs[str(model_lock_path)] = sha(model_lock_path)
    prior_rule_plan = ROOT / "artifacts/experience_substitute_plan_v1/pending_plan.json"
    prior_rules = json.loads(prior_rule_plan.read_text())["frozen_rule_and_runner_sha256"]
    verify_hashes(prior_rules)
    inputs.update(prior_rules)
    inputs[str(prior_rule_plan)] = sha(prior_rule_plan)
    packet = json.loads((packet_dir / "source_review_packet.json").read_text())
    documents = json.loads((source_dir / "source_documents.json").read_text())
    draft = json.loads((draft_dir / "reference_draft.json").read_text())
    if len(packet) != 50 or len({r["opportunity_id"] for r in packet}) != 50:
        raise ValueError("The unchanged 50-case sample is required")
    if len({d["opportunity_id"] for d in documents}) != len(documents):
        raise ValueError("Repeated source document")
    selected = {r["opportunity_id"]: r for r in packet}
    for doc in documents:
        validate_source_document(doc)
        row = selected[doc["opportunity_id"]]
        if row["source_status"] != "downloaded_unique_pdf" or len(doc["pages"]) != row["pdf_total_pages"]:
            raise ValueError("Source view differs from the original source packet")
        pdf = Path(doc["source"]["file_path"]).resolve()
        verify_hashes({str(pdf): doc["source"]["sha256"]})
        inputs[str(pdf)] = doc["source"]["sha256"]
    if {d["opportunity_id"] for d in documents} != {r["opportunity_id"] for r in packet if r["source_status"] == "downloaded_unique_pdf"}:
        raise ValueError("A downloaded source was silently omitted")
    for row in draft:
        if (row.get("predictions_seen") is not False or row.get("document_reference_complete") is not False
            or row.get("human_annotation") is not False):
            raise ValueError("Unexpected draft status")
        if row["opportunity_id"] not in selected:
            raise ValueError("Draft belongs outside the fixed sample")
    protocol = json.loads(protocol_path.read_text())
    if protocol["reference_type"] != "automatic_source_review" or protocol["predictions_generated"] is not False:
        raise ValueError("Unexpected evaluation protocol")
    locked_code = {str(ROOT / p): sha(ROOT / p) for p in (
        "src/tesis_recommender/reference_evaluation.py", "tests/test_reference_evaluation.py",
        "scripts/audit_holdout_evaluation_readiness.py")}
    inputs[str(protocol_path)] = sha(protocol_path)
    inputs.update(locked_code)
    verify_hashes(inputs)
    by_ocid = {d["opportunity_id"]: d for d in documents}
    ledger = []
    for row in packet:
        doc = by_ocid.get(row["opportunity_id"])
        ledger.append({"sample_number": row["sample_number"], "opportunity_id": row["opportunity_id"],
            "source_status": row["source_status"], "source_pages": len(doc["pages"]) if doc else None,
            "ocr_selected": sum(p["ocr_selected"] for p in doc["pages"]) if doc else None,
            "ocr_unresolved": sum(p["selected_ocr_unresolved"] for p in doc["pages"]) if doc else None,
            "partial_reference_conditions": sum(r["opportunity_id"] == row["opportunity_id"] for r in draft),
            "whole_document_reference_complete": False, "second_source_pass_complete": False,
            "evaluable_for_experience_f1": False, "ranking_eligible": False})
    summary = {"selected_cases": len(packet), "downloaded_unique_pdfs": len(documents),
        "source_status_counts": dict(Counter(r["source_status"] for r in packet)),
        "physical_pages": sum(len(d["pages"]) for d in documents),
        "ocr_selected_pages": sum(p["ocr_selected"] for d in documents for p in d["pages"]),
        "ocr_pages_with_text": sum(p["ocr_used"] for d in documents for p in d["pages"]),
        "ocr_unresolved_pages": sum(p["selected_ocr_unresolved"] for d in documents for p in d["pages"]),
        "source_lineage_validated": True, "protected_output_files_verified": len(protected),
        "frozen_model_files_verified": len(frozen), "prior_frozen_rule_files_verified": len(prior_rules),
        "partial_reference_conditions": len(draft), "complete_reference_documents": 0,
        "predictions_generated": False, "f1_estimable": False, "ranking_inputs_modified": False,
        "stage": "source_reading_available_reference_review_pending",
        "blocking_gates": ["whole_document_scope_and_condition_review", "second_source_only_pass", "final_reference_lock"]}
    output.mkdir(parents=True)
    dump(output / "readiness_ledger.json", ledger)
    dump(output / "summary.json", summary)
    dump(output / "evaluation_lock.json", {"protocol_path": str(protocol_path), "protocol_sha256": sha(protocol_path),
        "comparison_code_sha256": locked_code, "reference_locked": False,
        "predictions_generated": False, "interpretation": "Automatic reference concordance only, not expert accuracy"})
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "source", "draft", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.packet.resolve(), args.source.resolve(), args.draft.resolve(), args.protocol.resolve(), args.output.resolve())))
