"""Reconcile source-review progress without producing labels or predictions.

Hashes verify provenance, not the truth of semantic review attestations. A full
source reading is scoped to the supplied PDF, not the entire procurement file.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes
from scripts.verify_holdout_source_review import check_quote, quotations

ROOT = Path(__file__).resolve().parents[1]
EMPTY_DECISIONS = {"explicit_no_qualification_document_scope", "abstain_unsuitable_source"}


def validate_second_pass(review, sources, first_pass):
    if review.get("review_kind") != "automatic_source_only_second_pass":
        raise ValueError("Wrong second-pass review kind")
    for flag in ("human_annotation", "expert_annotation", "predictions_consulted",
                 "model_rules_consulted", "reference_locked"):
        if review.get(flag) is not False:
            raise ValueError("Second pass must be source-only, automatic and unlocked")
    seen = set()
    for doc in review["documents"]:
        ocid = doc["opportunity_id"]
        if ocid in seen:
            raise ValueError("Duplicate second-pass document")
        seen.add(ocid)
        source, previous = sources[ocid], first_pass[ocid]
        if doc["sample_number"] != previous["sample_number"] or doc["sample_number"] != source["source"]["sample_number"]:
            raise ValueError("Sample identity differs between passes")
        expected = {p["page"] for p in source["pages"]}
        read = doc["pages_read"]
        visual = doc["complete_rendered_pages_read"]
        for values in (read, visual):
            if any(type(p) is not int for p in values) or len(values) != len(set(values)) or not set(values) <= expected:
                raise ValueError("Invalid second-pass page coverage")
        if doc.get("full_supplied_document_read") is not True or set(read) != expected:
            raise ValueError("Second pass does not cover the supplied document")
        if previous["coverage"].get("full_pdf_semantically_read") is not True:
            raise ValueError("First pass remains partial")
        if doc.get("entire_procurement_file_reviewed") is not False:
            raise ValueError("Do not extrapolate review beyond the supplied PDF")
        if doc["decision"] not in EMPTY_DECISIONS or doc["decision"] != previous["decision"]:
            raise ValueError("Decision disagreement needs a separate adjudication")
        if doc["bidder_experience_conditions"] != [] or previous["bidder_experience_conditions"] != []:
            raise ValueError("This version only consolidates empty-condition source decisions")
        if not doc["evidence"]:
            raise ValueError("Source decision requires evidence")
        for quote in doc["evidence"]:
            page = next(p for p in source["pages"] if p["page"] == quote["page"])
            if quote["page"] not in visual or page["text"].count(quote["quote"]) != 1:
                raise ValueError("Second-pass evidence is ambiguous or not visually reviewed")
        if doc["decision"] == "explicit_no_qualification_document_scope" and not any(
            "REQUISITOS DE CALIFICACIÓN" in q["quote"] and "No corresponde" in q["quote"]
            for q in doc["evidence"]
        ):
            raise ValueError("An empty reference needs explicit scoped absence evidence")
    return {d["opportunity_id"]: d for d in review["documents"]}


def build(reading_dir, source_dir, first_path, second_path, draft_dir, scope_dir, output):
    if output.exists():
        raise ValueError("Use a new progress directory")
    inputs = dict(protected_outputs())
    for directory in (reading_dir, source_dir, draft_dir, scope_dir):
        verify_artifact(directory, inputs)
    # Preserve the old rules, model code and the separately frozen comparator.
    for path, key in (
        (ROOT / "artifacts/experience_substitute_plan_v1/pending_plan.json", "frozen_rule_and_runner_sha256"),
        (ROOT / "artifacts/experience_holdout_models_lock_v1/model_lock.json", "original_code_sha256"),
    ):
        lock = json.loads(path.read_text())
        inputs.update(lock[key])
        inputs[str(path)] = sha(path)
    comparator = ROOT / "artifacts/experience_holdout_evaluation_preflight_v1"
    verify_artifact(comparator, inputs)
    eval_lock = json.loads((comparator / "evaluation_lock.json").read_text())
    # Its source code is also in the manifest's input hashes.
    if eval_lock.get("predictions_generated") is True:
        raise ValueError("Predictions must not precede reference consolidation")
    first, second = json.loads(first_path.read_text()), json.loads(second_path.read_text())
    for path in (first_path, second_path):
        inputs[str(path)] = sha(path)
    for flag in ("human_annotation", "expert_annotation", "predictions_consulted", "model_rules_consulted",
                 "second_pass_performed", "reference_locked", "evaluation_ready"):
        if first.get(flag) is not False:
            raise ValueError("Unexpected first-pass attestation")
    sources = {d["opportunity_id"]: d for d in json.loads((source_dir / "source_documents.json").read_text())}
    first_by_id = {d["opportunity_id"]: d for d in first["documents"]}
    if len(first_by_id) != len(first["documents"]):
        raise ValueError("Duplicate first-pass document")
    for ocid, doc in first_by_id.items():
        source = sources[ocid]
        if Path(doc["pdf_path"]).resolve() != Path(source["source"]["file_path"]).resolve() or doc["pdf_sha256"] != source["source"]["sha256"]:
            raise ValueError("First-pass PDF lineage differs")
        expected = {p["page"] for p in source["pages"]}
        covered = set(doc["coverage"]["native_pages_read"]) | set(doc["coverage"]["complete_rendered_pages_read"]) | set(doc["coverage"].get("ocr_pages_read", []))
        if not covered <= expected or set(doc["coverage"]["unreviewed_pages"]) != expected - covered:
            raise ValueError("First-pass coverage does not reconcile")
        if doc["coverage"]["full_pdf_semantically_read"] and covered != expected:
            raise ValueError("Incomplete first-pass coverage")
        for evidence in quotations(doc):
            check_quote(next(p for p in source["pages"] if p["page"] == evidence["page"]), evidence)
            if evidence["visual_reviewed"]:
                image = ROOT / evidence["render_path"]
                inputs[str(image)] = sha(image)
    second_by_id = validate_second_pass(second, sources, first_by_id)
    attached = []
    for ocid, doc in second_by_id.items():
        source = sources[ocid]
        for evidence in doc["evidence"]:
            page = next(p for p in source["pages"] if p["page"] == evidence["page"])
            at = page["text"].index(evidence["quote"])
            image = ROOT / evidence["image"]
            inputs[str(image)] = sha(image)
            attached.append({"opportunity_id": ocid, "page": page["page"], "quote": evidence["quote"],
                "char_start": at, "char_end": at + len(evidence["quote"]),
                "page_text_sha256": hashlib.sha256(page["text"].encode()).hexdigest(),
                "pdf_sha256": source["source"]["sha256"], "image_path": str(image), "image_sha256": sha(image),
                "method": page["extraction_method"] + "_and_full_page_visual_review",
                "human_or_expert": False, "entire_procurement_file_reviewed": False})
    draft = json.loads((draft_dir / "reference_draft.json").read_text())
    scope = json.loads((scope_dir / "scope_observations.json").read_text())
    root_counts = Counter(d["opportunity_id"] for d in draft)
    scope_by_id = {d["opportunity_id"]: d for d in scope}
    ledger = json.loads((reading_dir / "pending_source_reviews.json").read_text())
    if len(ledger) != 50 or len({r["opportunity_id"] for r in ledger}) != 50:
        raise ValueError("Preserve the fifty-case denominator")
    rows = []
    for case in ledger:
        ocid = case["opportunity_id"]
        source, a, b = sources.get(ocid), first_by_id.get(ocid), second_by_id.get(ocid)
        if source:
            pdf = Path(source["source"]["file_path"]).resolve()
            inputs[str(pdf)] = source["source"]["sha256"]
        rows.append({"sample_number": case["sample_number"], "opportunity_id": ocid,
            "source_status": case["source_status"], "physical_pages": len(source["pages"]) if source else None,
            "first_pass_status": "full_supplied_pdf" if a and a["coverage"]["full_pdf_semantically_read"] else
                "partial" if a or root_counts[ocid] or ocid in scope_by_id else "not_performed",
            "first_pass_decision": a["decision"] if a else None,
            "root_scope_observation": scope_by_id.get(ocid),
            "root_partial_condition_count": root_counts[ocid],
            "reviewer_partial_inventory_count": len(a["bidder_experience_conditions"]) if a else 0,
            "inventories_merged_as_labels": False,
            "second_pass_status": "full_supplied_pdf" if b else "not_performed",
            "two_pass_agreement": b["decision"] if b else None,
            "document_absence_confirmed_two_pass": bool(b and b["decision"] == "explicit_no_qualification_document_scope"),
            "source_insufficient_confirmed_two_pass": bool(b and b["decision"] == "abstain_unsuitable_source"),
            "entire_procurement_file_reviewed": False, "reference_locked": False,
            "predictions_generated": False, "historical_ranking_eligible": False})
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "review_progress.json", rows)
    dump(output / "second_pass_evidence.json", attached)
    summary = {"selected_cases": len(rows), "downloaded_pdf": sum(r["physical_pages"] is not None for r in rows),
        "source_status_counts": dict(Counter(r["source_status"] for r in rows)),
        "first_pass_full_supplied_pdf": sum(r["first_pass_status"] == "full_supplied_pdf" for r in rows),
        "first_pass_partial": sum(r["first_pass_status"] == "partial" for r in rows),
        "second_pass_full_supplied_pdf": len(second_by_id),
        "scoped_absence_agreements": sum(r["document_absence_confirmed_two_pass"] for r in rows),
        "insufficient_source_agreements": sum(r["source_insufficient_confirmed_two_pass"] for r in rows),
        "root_partial_conditions": len(draft),
        "reviewer_first_pass_inventory": sum(len(d["bidder_experience_conditions"]) for d in first["documents"]),
        "inventories_must_not_be_added": True,
        "unresolved_cross_review_issues": second["unresolved_cross_review_issues"],
        "positive_document_references_complete": 0, "reference_locked": False,
        "predictions_generated": False, "f1_estimable": False,
        "semantic_correctness_proven_by_hashes": False, "ranking_inputs_modified": False}
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reading", "source", "first", "second", "draft", "scope", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(*(getattr(args, name).resolve() for name in
        ("reading", "source", "first", "second", "draft", "scope", "output")))))
