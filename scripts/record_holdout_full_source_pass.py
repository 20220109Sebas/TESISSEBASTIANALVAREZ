"""Record completed source readings without creating a second pass or labels.

Coverage is a reviewer's attestation. Hashes and offsets verify provenance only.
This script never reads predictions, executes an extractor or freezes a reference.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes
from scripts.verify_holdout_source_review import check_quote, quotations

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("human_annotation", "expert_annotation", "predictions_consulted", "model_rules_consulted",
         "second_pass_performed", "reference_locked", "evaluation_ready")


def prepare_review(value, sources):
    """Attach literal offsets to authored notes and validate full-page coverage."""
    review = deepcopy(value)
    if any(review.get(flag) is not False for flag in FLAGS):
        raise ValueError("Only an unlocked automatic first source pass is allowed")
    notes = review.get("review_kind") == "automatic_source_only_full_reading_notes"
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
            raise ValueError("Source identity or physical-page count differs")
        if doc.get("entire_procurement_file_reviewed") is not False:
            raise ValueError("Coverage must be limited to the supplied PDF")
        page_map = {p["page"]: p for p in source["pages"]}
        coverage = doc["coverage"]
        if coverage.get("full_pdf_semantically_read") is not True or coverage["unreviewed_pages"] != []:
            raise ValueError("This ledger records actual complete readings only")
        for key in ("native_pages_read", "ocr_pages_read", "complete_rendered_pages_read"):
            values = coverage[key]
            if (any(type(p) is not int for p in values) or len(values) != len(set(values))
                or not set(values) <= set(page_map)):
                raise ValueError("Invalid coverage page list")
        native, ocr, visual = (set(coverage[k]) for k in
            ("native_pages_read", "ocr_pages_read", "complete_rendered_pages_read"))
        if native | ocr | visual != set(page_map):
            raise ValueError("Full review has uncovered pages")
        # Legacy readings retain their quotation-linked renders. New explicit
        # inventories must cover every attested visual page, including pages
        # without a quotation. Neither inventory proves semantic correctness.
        if "rendered_pages" in doc:
            renders = doc["rendered_pages"]
            if (len(renders) != len(visual)
                or any(type(r.get("page")) is not int for r in renders)
                or {r["page"] for r in renders} != visual
                or any(not r.get("render_path") for r in renders)):
                raise ValueError("Rendered inventory differs from visual coverage")
            render_map = {r["page"]: r["render_path"] for r in renders}
            for evidence in doc.get("evidence", []):
                if evidence.get("render_path") != render_map.get(evidence["page"]):
                    raise ValueError("Quotation render differs from page inventory")
            for observation in doc.get("visual_observations", []):
                if (observation["page"] not in visual
                    or observation.get("replaces_ocr_text") is not False):
                    raise ValueError("Visual notes cannot silently rewrite OCR")
        for number in native:
            if not page_map[number].get("native_text", "").strip() and number not in visual:
                raise ValueError("Empty native page requires visual review")
        for number in ocr:
            if (page_map[number].get("ocr_used") is not True
                or page_map[number].get("selected_ocr_unresolved") is True and number not in visual):
                raise ValueError("Unavailable OCR cannot prove page coverage")
        if notes:
            for evidence in doc["evidence"]:
                page = page_map[evidence["page"]]
                text = page["text"]
                quote = evidence["quote"]
                if not quote or text.count(quote) != 1:
                    raise ValueError("Authored quotation is missing or ambiguous")
                start = text.index(quote)
                evidence.update(char_start=start, char_end=start + len(quote),
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    quote_basis="ocr_text_verbatim_not_error_free" if page["ocr_used"] else "native_text_verbatim",
                    visual_reviewed=True)
        evidence_count = 0
        for evidence in quotations(doc):
            evidence_count += 1
            page = page_map[evidence["page"]]
            check_quote(page, evidence)
            if evidence.get("visual_reviewed") is True and evidence["page"] not in visual:
                raise ValueError("Visual quotation page not included in declared coverage")
        if not evidence_count:
            raise ValueError("Full-reading decision requires source evidence")
        doc["reviewer"] = review["reviewer"]
    if not seen:
        raise ValueError("Empty review")
    return review


def register_render_inventory(doc, inputs):
    """Hash all explicitly attested renders without reinterpreting their text."""
    for rendered in doc.get("rendered_pages", []):
        image = ROOT / rendered["render_path"]
        digest = sha(image)
        if rendered.get("render_sha256", digest) != digest:
            raise ValueError("Rendered image changed")
        rendered["render_sha256"] = digest
        inputs[str(image.resolve())] = digest


def apply_first_pass(rows, reviews):
    result = deepcopy(rows)
    if len(result) != 50 or len({r["opportunity_id"] for r in result}) != 50:
        raise ValueError("Preserve all fifty selected opportunities")
    by_id, seen = {r["opportunity_id"]: r for r in result}, set()
    for review in reviews:
        for doc in review["documents"]:
            ocid = doc["opportunity_id"]
            if ocid in seen:
                raise ValueError("Multiple readings cannot silently become two-pass agreement")
            seen.add(ocid)
            row = by_id[ocid]
            if (row["source_status"] != "downloaded_unique_pdf" or row["physical_pages"] != doc["total_pages"]
                or row["sample_number"] != doc["sample_number"]):
                raise ValueError("Progress source differs")
            if row["second_pass_status"] != "not_performed" or row["reference_locked"] is not False:
                raise ValueError("Do not replace a second-pass or locked judgment")
            if row["predictions_generated"] is not False or row["historical_ranking_eligible"] is not False:
                raise ValueError("First-source progress must precede predictions and ranking use")
            row["previous_first_pass_status"] = row["first_pass_status"]
            row["previous_first_pass_decision"] = row["first_pass_decision"]
            row.update(first_pass_status="full_supplied_pdf", first_pass_decision=doc["decision"],
                latest_full_first_pass_reviewer=doc["reviewer"],
                latest_full_first_pass_condition_count=len(doc["bidder_experience_conditions"]),
                source_open_issues=doc.get("open_issues", doc.get("unknowns", [])))
    return result


def merge_source_reviews(previous, reviews):
    """Retain earlier first-pass evidence without counting it as a new reading."""
    by_id = {d["opportunity_id"]: deepcopy(d) for d in previous}
    if len(by_id) != len(previous):
        raise ValueError("Duplicate documents in previous source inventory")
    seen = set()
    for review in reviews:
        for doc in review["documents"]:
            ocid = doc["opportunity_id"]
            if ocid in seen:
                raise ValueError("Duplicate new first-pass document")
            seen.add(ocid)
            by_id[ocid] = deepcopy(doc)
    return sorted(by_id.values(), key=lambda d: d["sample_number"])


def current_block_counts(reviews):
    """Do not inherit prior partial-pass activity as current-block work."""
    return {"completed_first_readings_in_this_block":sum(len(r["documents"]) for r in reviews),
        "partial_readings_recorded_in_this_block":0, "new_second_passes_in_this_block":0}


def preserved_auxiliary_inventories(base):
    """Keep partial readings and second-pass notes as evidence history.

    A completed reading supersedes its partial status, not the earlier notes.
    Copy only known source-review inventories, never a previous manifest.
    """
    return {name: json.loads((base / name).read_text())
            for name in ("partial_source_reviews.json", "second_source_review.json")
            if (base / name).exists()}


def build(base, source_dir, review_paths, output):
    if output.exists():
        raise ValueError("Use a new output directory")
    inputs = dict(protected_outputs())
    for directory in (base, source_dir):
        verify_artifact(directory, inputs)
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    sources = {d["opportunity_id"]: d for d in json.loads((source_dir / "source_documents.json").read_text())}
    reviews = []
    for path in review_paths:
        inputs[str(path)] = sha(path)
        review = prepare_review(json.loads(path.read_text()), sources)
        reviews.append(review)
        for doc in review["documents"]:
            inputs[str(Path(doc["pdf_path"]).resolve())] = doc["pdf_sha256"]
            register_render_inventory(doc, inputs)
            for evidence in quotations(doc):
                if evidence.get("visual_reviewed") is True:
                    image = ROOT / evidence["render_path"]
                    inputs[str(image)] = sha(image)
    rows = apply_first_pass(json.loads((base / "review_progress.json").read_text()), reviews)
    summary = json.loads((base / "summary.json").read_text())
    # Preserve earlier disagreements as history, without presenting the old
    # "reviewer has not saved its reading" wording as the current state.
    if "prior_cross_review_issues" not in summary:
        summary["prior_cross_review_issues"] = summary.get("unresolved_cross_review_issues", [])
    summary["unresolved_cross_review_issues"] = [
        {"sample_number": row["sample_number"], "second_full_source_pass_pending": True,
         "source_warnings": row["source_open_issues"]}
        for row in rows if row.get("latest_full_first_pass_reviewer")
        and row["second_pass_status"] == "not_performed"]
    summary.update(first_pass_full_supplied_pdf=sum(r["first_pass_status"] == "full_supplied_pdf" for r in rows),
        first_pass_partial=sum(r["first_pass_status"] == "partial" for r in rows),
        downloaded_sources_not_read=sum(r["source_status"] == "downloaded_unique_pdf" and r["first_pass_status"] == "not_performed" for r in rows),
        completed_first_readings_in_this_block=sum(len(r["documents"]) for r in reviews),
        second_pass_full_supplied_pdf=sum(r["second_pass_status"] == "full_supplied_pdf" for r in rows),
        new_second_passes_in_this_block=0, previous_progress_preserved=True,
        source_clauses_not_merged_as_reference=True)
    summary.update(current_block_counts(reviews))
    previous_path = base / "compiled_source_reviews.json"
    previous_docs = json.loads(previous_path.read_text())["documents"] if previous_path.exists() else []
    combined = {"review_kind": "automatic_source_only_first_pass_extended", "reviewer": "Document-specific automatic reviewers",
        **{flag: False for flag in FLAGS}, "documents": merge_source_reviews(previous_docs, reviews)}
    auxiliary = preserved_auxiliary_inventories(base)
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "compiled_source_reviews.json", combined)
    dump(output / "review_progress.json", rows)
    dump(output / "second_pass_evidence.json", json.loads((base / "second_pass_evidence.json").read_text()))
    dump(output / "summary.json", summary)
    for name, inventory in auxiliary.items():
        dump(output / name, inventory)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--review", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base.resolve(), args.source.resolve(), [p.resolve() for p in args.review], args.output.resolve())))
