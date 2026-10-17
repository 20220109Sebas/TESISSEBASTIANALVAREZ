"""Create per-document source transcripts and a pending review ledger, without labels.

Transcripts support automatic source reading. They do not certify that a reviewer
has read every page and never replace preserved native/OCR page evidence.
"""
import argparse
import json
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import validate_source_document, verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes


def transcript(doc):
    validate_source_document(doc)
    lines = ["SOURCE-ONLY READING TRANSCRIPT", f'OCID: {doc["opportunity_id"]}',
             f'PDF SHA256: {doc["source"]["sha256"]}',
             "Generated transcript is not a completed document review.",
             "No model predictions or presumed absent requirements are included.", ""]
    for page in doc["pages"]:
        lines.extend([f'PHYSICAL PAGE {page["page"]}', f'METHOD: {page["extraction_method"]}',
                      f'PAGE TEXT SHA256: {page["text_sha256"]}',
                      f'NATIVE TEXT SHA256: {page["native_text_sha256"]}',
                      f'OCR UNRESOLVED: {page["selected_ocr_unresolved"]}', "", page["text"], ""])
    return "\n".join(lines)


def prepare(packet_dir, source_dir, draft_dir, output):
    if output.exists():
        raise ValueError("Use a new source-reading directory")
    inputs = dict(protected_outputs())
    for directory in (packet_dir, source_dir, draft_dir):
        verify_artifact(directory, inputs)
    packet = json.loads((packet_dir / "source_review_packet.json").read_text())
    docs = json.loads((source_dir / "source_documents.json").read_text())
    draft = json.loads((draft_dir / "reference_draft.json").read_text())
    by_ocid = {d["opportunity_id"]: d for d in docs}
    if len(packet) != 50 or len({r["opportunity_id"] for r in packet}) != 50 or len(by_ocid) != len(docs):
        raise ValueError("Unexpected reading sample or duplicate source")
    text_files, queue = {}, []
    for case in packet:
        doc = by_ocid.get(case["opportunity_id"])
        name = f'source_text/{case["sample_number"]:02d}_{case["opportunity_id"]}.txt' if doc else None
        if doc:
            text_files[name] = transcript(doc)
        queue.append({"sample_number": case["sample_number"], "opportunity_id": case["opportunity_id"],
            "source_status": case["source_status"], "source_pdf": doc["source"] if doc else None,
            "transcript_path_relative": name, "source_view_path": str(source_dir / "source_documents.json"),
            "expected_physical_pages": [p["page"] for p in doc["pages"]] if doc else None,
            "existing_partial_decisions": [r for r in draft if r["opportunity_id"] == case["opportunity_id"]],
            "reviewed_physical_pages_pass1": [], "reviewed_physical_pages_pass2": [],
            "scope_decision": None, "all_conditions_checked": False,
            "reference_conditions": None, "predictions_seen": False,
            "human_annotation": False, "whole_document_reference_complete": False})
    verify_hashes(inputs)
    output.mkdir(parents=True)
    (output / "source_text").mkdir()
    for name, value in text_files.items():
        (output / name).write_text(value, encoding="utf-8")
    dump(output / "pending_source_reviews.json", queue)
    summary = {"selected_cases": len(queue), "source_transcripts": len(text_files),
        "physical_pages": sum(len(d["pages"]) for d in docs),
        "complete_reviews_claimed": 0, "predictions_included": False,
        "purpose": "source_reading_aid_not_semantic_validation"}
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "source", "draft", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.packet.resolve(), args.source.resolve(), args.draft.resolve(), args.output.resolve())))
