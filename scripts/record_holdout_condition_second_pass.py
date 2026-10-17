"""Register a source-only condition rereading, without locking labels or scoring predictions.

Structural checks and literal evidence checks do not prove semantic correctness.
Previous inventories remain unchanged. Conditions are paired by identity, not values.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("subject", "section_role", "item", "branch")
FIELDS = ("billed_amount", "currency", "lookback_years", "window_anchor", "reference_date", "similar_services")


def keyed_conditions(conditions):
    result = {}
    for condition in conditions:
        key = tuple(condition.get(field) for field in IDENTITY)
        if key in result:
            raise ValueError("Duplicate condition identity")
        result[key] = condition
    return result


def compare_conditions(prior, current):
    left, right = keyed_conditions(prior), keyed_conditions(current)
    comparisons = []
    for key in sorted(set(left) | set(right), key=repr):
        a, b = left.get(key), right.get(key)
        comparisons.append({
            "identity": dict(zip(IDENTITY, key)),
            "first_pass_present": a is not None, "second_pass_present": b is not None,
            "fields": {field: {"first": a.get(field) if a else None,
                "second": b.get(field) if b else None,
                "status": "unknown" if a is None or b is None or a.get(field) is None or b.get(field) is None
                else "match" if a[field] == b[field] else "disagreement"} for field in FIELDS},
        })
    return comparisons


def validate(review, sources, first):
    if review.get("review_kind") != "automatic_source_only_condition_second_pass":
        raise ValueError("Unexpected review kind")
    for flag in ("human_annotation", "expert_annotation", "independent_reviewer",
                 "predictions_consulted", "model_rules_consulted", "reference_locked"):
        if review.get(flag) is not False:
            raise ValueError("Require disclosed automatic source-only review")
    if review.get("first_pass_notes_consulted") is not True or not review.get("independence_note"):
        raise ValueError("Disclose consultation of first-pass notes")
    if not review.get("reviewer") or not review.get("documents"):
        raise ValueError("Reviewer and documents required")
    seen = set()
    for doc in review["documents"]:
        key = doc["opportunity_id"]
        if key in seen:
            raise ValueError("Duplicate document")
        seen.add(key)
        source, prior = sources[key], first[key]
        if (doc["sample_number"] != source["source"]["sample_number"]
                or doc["sample_number"] != prior["sample_number"]
                or doc["pdf_sha256"] != source["source"]["sha256"]):
            raise ValueError("Source identity mismatch")
        pages = {p["page"] for p in source["pages"]}
        if (doc.get("full_supplied_document_read") is not True
                or prior["coverage"].get("full_pdf_semantically_read") is not True
                or doc.get("entire_procurement_file_reviewed") is not False
                or doc.get("unreviewed_pages") != []):
            raise ValueError("Both supplied-PDF readings must be complete")
        for name in ("pages_read", "native_pages_read", "ocr_pages_read", "complete_rendered_pages_read"):
            values = doc[name]
            if any(type(p) is not int for p in values) or len(values) != len(set(values)) or not set(values) <= pages:
                raise ValueError("Invalid page coverage")
        native, ocr = set(doc["native_pages_read"]), set(doc["ocr_pages_read"])
        if set(doc["pages_read"]) != pages or native | ocr != pages or native & ocr:
            raise ValueError("Incomplete or overlapping text coverage")
        if not ocr <= set(doc["complete_rendered_pages_read"]):
            raise ValueError("Scanned pages require visual checking")
        if not doc.get("source_warnings") or not doc.get("scope_observation") or not doc.get("reading_sections"):
            raise ValueError("Reading observations and limitations required")
        evidence = {item["id"]: item for item in doc["evidence"]}
        if not evidence or len(evidence) != len(doc["evidence"]):
            raise ValueError("Unique evidence required")
        for item in evidence.values():
            page = next((p for p in source["pages"] if p["page"] == item["page"]), None)
            if (page is None or item["page"] not in doc["complete_rendered_pages_read"]
                    or not item["quote"].strip() or page["text"].count(item["quote"]) != 1):
                raise ValueError("Quote must be unique and visually checked")
        conditions = doc["bidder_experience_conditions"]
        if not conditions:
            raise ValueError("Use a scoped absence registrar for empty inventories")
        for condition in conditions:
            if (condition.get("subject") != "bidder" or condition.get("section_role") != "qualification"
                    or not condition.get("branch") or condition.get("document_reference_complete") is not False
                    or condition.get("company_applicability") != "not_determined"):
                raise ValueError("Do not infer locked references or company eligibility")
            amount, currency, years = (condition.get(field) for field in FIELDS[:3])
            if amount is not None and (not isinstance(amount, str) or not re.fullmatch(r"\d+\.\d{2}", amount)):
                raise ValueError("Canonical decimal amount required")
            if currency is not None and (not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency)):
                raise ValueError("Canonical currency required")
            if years is not None and (type(years) is not int or years <= 0):
                raise ValueError("Explicit positive integer window required")
            ids = condition.get("evidence_ids", [])
            if not ids or not set(ids) <= set(evidence):
                raise ValueError("Condition must cite recorded evidence")
        comparisons = compare_conditions(prior["bidder_experience_conditions"], conditions)
        if doc["decision"] == "source_conditions_agree_with_warnings_pending_reference_lock":
            if any(not c["first_pass_present"] or not c["second_pass_present"]
                   or any(v["status"] != "match" for v in c["fields"].values()) for c in comparisons):
                raise ValueError("Agreement claim does not match inventories")
        elif doc["decision"] != "source_conditions_discrepant_pending_reference_lock":
            raise ValueError("Unsupported source decision")


def build(base, source_dir, review_path, output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    inputs = dict(protected_outputs())
    for directory in (base, source_dir):
        verify_artifact(directory, inputs)
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    review = json.loads(review_path.read_text())
    first = {d["opportunity_id"]: d for d in json.loads((base / "compiled_source_reviews.json").read_text())["documents"]}
    sources = {d["opportunity_id"]: d for d in json.loads((source_dir / "source_documents.json").read_text())}
    validate(review, sources, first)
    inputs[str(review_path)] = sha(review_path)
    rows = json.loads((base / "review_progress.json").read_text())
    if len(rows) != 50 or len({r["opportunity_id"] for r in rows}) != 50:
        raise ValueError("Preserve fifty-case denominator")
    by_id = {r["opportunity_id"]: r for r in rows}
    evidence_rows = json.loads((base / "second_pass_evidence.json").read_text())
    for doc in review["documents"]:
        row, source = by_id[doc["opportunity_id"]], sources[doc["opportunity_id"]]
        if (row["second_pass_status"] != "not_performed" or row["first_pass_status"] != "full_supplied_pdf"
                or row["source_status"] != "downloaded_unique_pdf" or row["reference_locked"] is not False
                or row["predictions_generated"] is not False or row["historical_ranking_eligible"] is not False):
            raise ValueError("Cannot replace readings or unlock protected use")
        row.update(second_pass_status="full_supplied_pdf", two_pass_agreement=doc["decision"],
            second_pass_reviewer=review["reviewer"], second_pass_independent_reviewer=False,
            second_pass_independence_note=review["independence_note"],
            second_pass_source_warnings=doc["source_warnings"], reference_adjudication_pending=True,
            second_pass_field_comparison=compare_conditions(first[doc["opportunity_id"]]["bidder_experience_conditions"], doc["bidder_experience_conditions"]))
        for number in doc["complete_rendered_pages_read"]:
            image = ROOT / f"tmp/pdfs/holdout_case{doc['sample_number']}_second_p{number}.png"
            inputs[str(image)] = sha(image)
        for item in doc["evidence"]:
            page = next(p for p in source["pages"] if p["page"] == item["page"])
            start = page["text"].index(item["quote"])
            image = ROOT / f"tmp/pdfs/holdout_case{doc['sample_number']}_second_p{item['page']}.png"
            evidence_rows.append({"opportunity_id": doc["opportunity_id"], "evidence_id": item["id"],
                "page": item["page"], "quote": item["quote"], "char_start": start,
                "char_end": start + len(item["quote"]),
                "page_text_sha256": hashlib.sha256(page["text"].encode()).hexdigest(),
                "pdf_sha256": doc["pdf_sha256"], "image_path": str(image), "image_sha256": sha(image),
                "method": page["extraction_method"] + "_and_full_page_visual_review",
                "human_or_expert": False, "independent_reviewer": False,
                "entire_procurement_file_reviewed": False})
    summary = json.loads((base / "summary.json").read_text())
    summary.update(second_pass_full_supplied_pdf=sum(r["second_pass_status"] == "full_supplied_pdf" for r in rows),
        condition_source_agreements_pending_reference=sum(r.get("two_pass_agreement") == "source_conditions_agree_with_warnings_pending_reference_lock" for r in rows),
        completed_first_readings_in_this_block=0, partial_readings_recorded_in_this_block=0,
        new_second_passes_in_this_block=len(review["documents"]),
        positive_document_references_complete=0, reference_locked=False, predictions_generated=False,
        f1_estimable=False, ranking_inputs_modified=False)
    for issue in summary.get("unresolved_cross_review_issues", []):
        if issue["sample_number"] in {d["sample_number"] for d in review["documents"]}:
            issue.update(second_full_source_pass_pending=False, reference_adjudication_pending=True)
    verify_hashes(inputs)
    output.mkdir(parents=True)
    changed = {"review_progress.json", "second_pass_evidence.json", "summary.json", "manifest.json", "condition_second_source_reviews.json"}
    for path in base.iterdir():
        if path.is_file() and path.name not in changed:
            shutil.copyfile(path, output / path.name)
    previous = base / "condition_second_source_reviews.json"
    inventory = json.loads(previous.read_text()) if previous.exists() else []
    dump(output / "condition_second_source_reviews.json", inventory + [review])
    dump(output / "review_progress.json", rows)
    dump(output / "second_pass_evidence.json", evidence_rows)
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return {k: summary[k] for k in ("first_pass_full_supplied_pdf", "second_pass_full_supplied_pdf", "condition_source_agreements_pending_reference", "f1_estimable")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "source", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(*(getattr(args, name).resolve() for name in ("base", "source", "review", "output")))))
