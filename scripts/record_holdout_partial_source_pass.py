"""Save source-only partial readings without promoting them to complete references.

Coverage is an automatic reviewer's attestation. Literal checks establish
provenance, not accuracy or an independent second semantic reading.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.record_holdout_full_source_pass import FLAGS
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes
from scripts.verify_holdout_source_review import check_quote

ROOT = Path(__file__).resolve().parents[1]


def prepare_review(value, sources):
    review = deepcopy(value)
    if review.get("review_kind") != "automatic_source_only_partial_reading_notes":
        raise ValueError("Expected partial source notes")
    if any(review.get(flag) is not False for flag in FLAGS):
        raise ValueError("Only unlocked automatic source readings are permitted")
    if not review.get("reviewer") or not review.get("documents"):
        raise ValueError("Missing reviewer or documents")
    seen = set()
    for doc in review["documents"]:
        ocid = doc["opportunity_id"]
        if ocid in seen:
            raise ValueError("Duplicate document")
        seen.add(ocid)
        source = sources[ocid]
        if (doc["sample_number"] != source["source"]["sample_number"]
                or doc["pdf_sha256"] != source["source"]["sha256"]
                or Path(doc["pdf_path"]).resolve() != Path(source["source"]["file_path"]).resolve()
                or doc["total_pages"] != len(source["pages"])):
            raise ValueError("Source identity differs")
        for flag in ("entire_procurement_file_reviewed", "historical_content_proven", "ranking_eligible"):
            if doc.get(flag) is not False:
                raise ValueError("Partial evidence cannot claim historical or complete-file validity")
        pages = {p["page"]: p for p in source["pages"]}
        coverage = doc["coverage"]
        if coverage.get("full_pdf_semantically_read") is not False:
            raise ValueError("A partial pass cannot claim full coverage")
        for key in ("native_pages_read", "ocr_pages_read", "complete_rendered_pages_read", "unreviewed_pages"):
            values = coverage[key]
            if (any(type(p) is not int for p in values) or len(values) != len(set(values))
                    or not set(values) <= set(pages)):
                raise ValueError("Invalid physical-page coverage")
        native, ocr, visual = (set(coverage[k]) for k in
            ("native_pages_read", "ocr_pages_read", "complete_rendered_pages_read"))
        examined = native | ocr | visual
        if not examined or examined == set(pages) or set(coverage["unreviewed_pages"]) != set(pages) - examined:
            raise ValueError("Partial coverage and remaining pages do not reconcile")
        for number in native:
            if not pages[number].get("native_text", "").strip() and number not in visual:
                raise ValueError("Empty native text cannot attest reading")
        for number in ocr:
            if pages[number].get("ocr_used") is not True or (
                    pages[number].get("selected_ocr_unresolved") is True and number not in visual):
                raise ValueError("OCR is unavailable")
        renders = doc["rendered_pages"]
        if (len(renders) != len(visual) or {r["page"] for r in renders} != visual
                or any(not r.get("render_path") for r in renders)):
            raise ValueError("Each visual page needs its saved render")
        render_map = {r["page"]: r["render_path"] for r in renders}
        evidence_ids = set()
        if not doc.get("evidence"):
            raise ValueError("Source notes require literal evidence")
        for evidence in doc["evidence"]:
            number, quote = evidence["page"], evidence["quote"]
            if evidence["id"] in evidence_ids:
                raise ValueError("Duplicate evidence identity")
            evidence_ids.add(evidence["id"])
            if number not in visual or evidence["render_path"] != render_map[number]:
                raise ValueError("Evidence must match an examined full-page render")
            page = pages[number]
            text = page["text"] if page["ocr_used"] else page["native_text"]
            if not quote or text.count(quote) != 1:
                raise ValueError("Missing or ambiguous literal quotation")
            start = text.index(quote)
            evidence.update(char_start=start, char_end=start + len(quote),
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                quote_basis="ocr_text_verbatim_not_error_free" if page["ocr_used"] else "native_text_verbatim",
                visual_reviewed=True)
            check_quote(page, evidence)
        for condition in doc["bidder_experience_conditions"]:
            if (condition.get("subject") != "bidder" or condition.get("document_reference_complete") is not False
                    or not condition.get("evidence_ids") or not set(condition["evidence_ids"]) <= evidence_ids):
                raise ValueError("Conditions must remain source-linked drafts")
        for observation in doc.get("visual_observations", []):
            if observation["page"] not in visual or observation.get("replaces_ocr_text") is not False:
                raise ValueError("Visual observations cannot silently rewrite OCR")
        doc["reviewer"] = review["reviewer"]
    return review


def apply_partial_pass(rows, review):
    result = deepcopy(rows)
    if len(result) != 50 or len({r["opportunity_id"] for r in result}) != 50:
        raise ValueError("Preserve the fifty-case denominator")
    by_id = {r["opportunity_id"]: r for r in result}
    for doc in review["documents"]:
        row = by_id[doc["opportunity_id"]]
        if (row["source_status"] != "downloaded_unique_pdf" or row["sample_number"] != doc["sample_number"]
                or row["physical_pages"] != doc["total_pages"]):
            raise ValueError("Progress identity differs")
        if row["first_pass_status"] not in ("not_performed", "partial") or row["second_pass_status"] != "not_performed":
            raise ValueError("Never replace a complete or second-pass judgment with partial notes")
        if any(row.get(flag) is not False for flag in
                ("reference_locked", "predictions_generated", "historical_ranking_eligible")):
            raise ValueError("Partial review cannot follow locked evaluation or ranking use")
        row.update(first_pass_status="partial", latest_partial_first_pass_reviewer=review["reviewer"],
            latest_partial_first_pass_decision=doc["decision"],
            latest_partial_first_pass_condition_count=len(doc["bidder_experience_conditions"]),
            latest_partial_first_pass_coverage=deepcopy(doc["coverage"]),
            latest_partial_source_open_issues=deepcopy(doc.get("open_issues", [])))
    return result


def build(base, source_dir, review_path, output):
    if output.exists():
        raise ValueError("Use a new output directory")
    inputs = dict(protected_outputs())
    for directory in (base, source_dir):
        verify_artifact(directory, inputs)
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    sources = {d["opportunity_id"]: d for d in json.loads((source_dir / "source_documents.json").read_text())}
    inputs[str(review_path)] = sha(review_path)
    review = prepare_review(json.loads(review_path.read_text()), sources)
    for doc in review["documents"]:
        inputs[str(Path(doc["pdf_path"]).resolve())] = doc["pdf_sha256"]
        for rendered in doc["rendered_pages"]:
            image = ROOT / rendered["render_path"]
            inputs[str(image)] = sha(image)
            rendered["render_sha256"] = sha(image)
    previous = json.loads((base / "review_progress.json").read_text())
    rows = apply_partial_pass(previous, review)
    summary = json.loads((base / "summary.json").read_text())
    summary.update(first_pass_full_supplied_pdf=sum(r["first_pass_status"] == "full_supplied_pdf" for r in rows),
        first_pass_partial=sum(r["first_pass_status"] == "partial" for r in rows),
        downloaded_sources_not_read=sum(r["source_status"] == "downloaded_unique_pdf"
            and r["first_pass_status"] == "not_performed" for r in rows),
        completed_first_readings_in_this_block=0, partial_readings_recorded_in_this_block=len(review["documents"]),
        new_second_passes_in_this_block=0, previous_progress_preserved=True,
        source_clauses_not_merged_as_reference=True)
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "partial_source_reviews.json", review)
    dump(output / "review_progress.json", rows)
    dump(output / "summary.json", summary)
    for filename in ("compiled_source_reviews.json", "second_pass_evidence.json"):
        if (base / filename).exists():
            dump(output / filename, json.loads((base / filename).read_text()))
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "source", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(*(getattr(args, n).resolve() for n in ("base", "source", "review", "output")))))
