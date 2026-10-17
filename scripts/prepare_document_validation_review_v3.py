"""Register actual partial source readings; never synthesize unread references.

Pending complete PDFs remain pending, not inaccessible/negative, and cannot be
locked. This packet does not invoke the candidate or generate field metrics.
"""
import argparse
import copy
import json
from pathlib import Path

from scripts.document_validation_run_v2 import checked, write_artifact
from scripts.document_validation_run_v3 import validate_run
from scripts.register_browser_experience_sources import sha
from tesis_recommender.document_experience_evaluation_v1 import build_reference, validate_reading

ROOT = Path(__file__).resolve().parents[1]


def prepare(design_dir, source_dir, source_gate_dir, notes_path, output):
    validate_run(design_dir)
    inputs = {}
    for directory in (design_dir, source_dir, source_gate_dir):
        checked(directory, inputs)
    gate_manifest = json.loads((source_gate_dir / "manifest.json").read_text())
    if gate_manifest["input_sha256"].get(str(source_dir / "manifest.json")) != sha(source_dir / "manifest.json"):
        raise ValueError("Source population gate does not cover this view")
    notes = json.loads(notes_path.read_text())
    if (notes["predictions_consulted"] is not False or notes["reference_locked"] is not False
            or notes["model_rules_consulted_during_annotation"] is not False):
        raise ValueError("Only disclosed source-only unlocked readings")
    inputs[str(notes_path)] = sha(notes_path)
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    documents = json.loads((source_dir / "source_documents.json").read_text())
    ledger = json.loads((source_dir / "coverage_ledger.json").read_text())
    sources = {d["opportunity_id"]: d for d in documents}
    annotations = {n["opportunity_id"]: n for n in notes["documents"]}
    if len(annotations) != len(notes["documents"]) or not set(annotations) <= set(sources):
        raise ValueError("Repeated/out-of-source review annotation")
    draft = {"reference_type": "automatic_source_review", "locked": False, "predictions_seen_before_lock": False,
        "is_human_or_expert": False, "independent_reviewer": False, "independence_note": notes["independence_note"],
        "documents": []}
    progress = []
    for case in ledger:
        ocid = case["opportunity_id"]
        source, annotation = sources.get(ocid), annotations.get(ocid)
        entry = {"opportunity_id": ocid, "scope": "pending_reference" if source else "inaccessible_source",
                 "reason": "Fuente disponible pendiente de lectura" if source else "Sin PDF completo elegible; no es un negativo",
                 "conditions": [], "readings": [], "source_pdf_sha256": source["source"]["sha256"] if source else None}
        if annotation:
            if annotation["sample_number"] != case["sample_number"]:
                raise ValueError("Review sample number differs")
            if annotation.get("source_pdf_sha256", source["source"]["sha256"]) != source["source"]["sha256"]:
                raise ValueError("Review PDF differs")
            page_hashes = [p["text_sha256"] for p in source["pages"]]
            if annotation.get("page_text_sha256", page_hashes) != page_hashes:
                raise ValueError("Review page view differs")
            entry.update(scope=annotation["scope"], reason=annotation["reason"], conditions=copy.deepcopy(annotation["conditions"]))
            for reviewed in annotation["readings"]:
                expected_pages = list(range(1, len(source["pages"]) + 1))
                if (sorted(reviewed["pages_read_order"]) != expected_pages
                        or len(reviewed["page_notes"]) != len(expected_pages)):
                    raise ValueError("A full reading cannot omit physical pages or their review outcomes")
                reading = {"pass_number": reviewed["pass_number"], "reviewer": notes["reviewer"],
                    "full_supplied_document_read": reviewed["full_supplied_document_read"],
                    "predictions_consulted": False, "model_rules_consulted": False, "scope": reviewed["scope"],
                    "conditions": copy.deepcopy(annotation["conditions"]),
                    "page_reviews": [{"page": p["page"], "page_text_sha256": p["text_sha256"],
                        "content_status": "readable", "review_note": reviewed["page_notes"][p["page"]-1]}
                                     for p in source["pages"]]}
                if "scoped_absence_statement" in reviewed:
                    reading["scoped_absence_statement"] = reviewed["scoped_absence_statement"]
                validate_reading(reading, source, {})
                entry["readings"].append(reading)
        draft["documents"].append(entry)
        progress.append({"sample_number": case["sample_number"], "opportunity_id": ocid,
            "source_status": case["source_status"], "physical_pages": len(source["pages"]) if source else 0,
            "first_pass_complete": any(r["pass_number"] == 1 for r in entry["readings"]),
            "second_pass_complete": any(r["pass_number"] == 2 for r in entry["readings"]),
            "reference_locked": False, "predictions_generated": False, "ranking_eligible": False})
    # This must fail while real source readings remain incomplete. Do not create
    # a lock/inference artifact merely to make the workflow appear completed.
    try:
        build_reference(draft, documents, ledger)
        ready, stop = True, None
    except ValueError as exc:
        ready, stop = False, str(exc)
    summary = {"stage": "source_reference_review_in_progress", "selected_cases": len(ledger),
        "complete_source_documents": len(documents), "physical_pages": sum(len(d["pages"]) for d in documents),
        "first_pass_complete_documents": sum(r["first_pass_complete"] for r in progress),
        "second_pass_complete_documents": sum(r["second_pass_complete"] for r in progress),
        "first_pass_pending_documents": sum(not r["first_pass_complete"] for r in progress if r["physical_pages"]),
        "second_pass_pending_documents": sum(not r["second_pass_complete"] for r in progress if r["physical_pages"]),
        "ready_to_lock_reference": ready, "reference_gate_message": stop,
        "reference_locked": False, "predictions_generated": False, "metrics_computed": False,
        "ranking_inputs_modified": False, "independent_or_expert_review": False}
    return write_artifact(output, {"reference_draft.json": draft, "review_progress.json": progress,
                                  "summary.json": summary}, inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("design", "source", "source-gate", "notes", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(*(getattr(args,n).resolve() for n in ("design","source","source_gate","notes","output"))), ensure_ascii=False))
