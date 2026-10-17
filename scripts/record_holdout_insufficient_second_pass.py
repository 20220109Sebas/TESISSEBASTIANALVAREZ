"""Append agreed insufficient-source rereadings without producing negative labels."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]


def validate(review, sources, first):
    if review.get("review_kind") != "automatic_source_only_insufficient_second_pass":
        raise ValueError("Unexpected review kind")
    for flag in ("human_annotation", "expert_annotation", "predictions_consulted", "model_rules_consulted", "reference_locked", "independent_reviewer"):
        if review.get(flag) is not False:
            raise ValueError("Require disclosed automatic non-blind source review")
    seen = set()
    for doc in review["documents"]:
        key = doc["opportunity_id"]
        if key in seen:
            raise ValueError("Duplicate document")
        seen.add(key)
        source, prior = sources[key], first[key]
        pages = {p["page"] for p in source["pages"]}
        if (doc["sample_number"] != source["source"]["sample_number"] or doc["sample_number"] != prior["sample_number"]
                or doc["pdf_sha256"] != source["source"]["sha256"]):
            raise ValueError("Source identity mismatch")
        if (doc.get("full_supplied_document_read") is not True or prior["coverage"].get("full_pdf_semantically_read") is not True
                or doc.get("entire_procurement_file_reviewed") is not False):
            raise ValueError("Both supplied-PDF readings must be complete")
        for name in ("pages_read", "complete_rendered_pages_read"):
            values = doc[name]
            if any(type(p) is not int for p in values) or len(set(values)) != len(values) or not set(values) <= pages:
                raise ValueError("Invalid coverage")
        if set(doc["pages_read"]) != pages:
            raise ValueError("Incomplete reading")
        if (doc["decision"] != "abstain_missing_specific_terms" or prior["decision"] != doc["decision"]
                or doc["bidder_experience_conditions"] != [] or not doc.get("source_warnings")):
            raise ValueError("Only agreed insufficient-source decisions are accepted")
        if not doc.get("evidence"):
            raise ValueError("Evidence required")
        for evidence in doc["evidence"]:
            page = next(p for p in source["pages"] if p["page"] == evidence["page"])
            if evidence["page"] not in doc["complete_rendered_pages_read"] or page["text"].count(evidence["quote"]) != 1:
                raise ValueError("Quote must be unique and visually checked")


def build(base, source_dir, first_path, review_path, output):
    if output.exists():
        raise ValueError("Use a fresh output directory")
    inputs = dict(protected_outputs())
    for directory in (base, source_dir):
        verify_artifact(directory, inputs)
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    review = json.loads(review_path.read_text())
    first = {d["opportunity_id"]: d for d in json.loads(first_path.read_text())["documents"]}
    sources = {d["opportunity_id"]: d for d in json.loads((source_dir / "source_documents.json").read_text())}
    validate(review, sources, first)
    inputs[str(review_path)], inputs[str(first_path)] = sha(review_path), sha(first_path)
    rows = json.loads((base / "review_progress.json").read_text())
    if len(rows) != 50 or len({r["opportunity_id"] for r in rows}) != 50:
        raise ValueError("Preserve fifty-case denominator")
    by_id = {r["opportunity_id"]: r for r in rows}
    evidence_rows = json.loads((base / "second_pass_evidence.json").read_text())
    for doc in review["documents"]:
        row, source = by_id[doc["opportunity_id"]], sources[doc["opportunity_id"]]
        if row["second_pass_status"] != "not_performed":
            raise ValueError("Cannot replace previous second pass")
        row.update(second_pass_status="full_supplied_pdf", two_pass_agreement=doc["decision"],
            source_insufficient_confirmed_two_pass=True, second_pass_reviewer=review["reviewer"],
            second_pass_independent_reviewer=False, second_pass_independence_note=review["independence_note"],
            second_pass_source_warnings=doc["source_warnings"])
        for number in doc["complete_rendered_pages_read"]:
            image = ROOT / f"tmp/pdfs/holdout_case{doc['sample_number']}_second_p{number}.png"
            inputs[str(image)] = sha(image)
        for item in doc["evidence"]:
            page = next(p for p in source["pages"] if p["page"] == item["page"])
            start = page["text"].index(item["quote"])
            image = ROOT / f"tmp/pdfs/holdout_case{doc['sample_number']}_second_p{item['page']}.png"
            evidence_rows.append({"opportunity_id": doc["opportunity_id"], "page": item["page"], "quote": item["quote"],
                "char_start": start, "char_end": start + len(item["quote"]),
                "page_text_sha256": hashlib.sha256(page["text"].encode()).hexdigest(),
                "pdf_sha256": doc["pdf_sha256"], "image_path": str(image), "image_sha256": sha(image),
                "method": page["extraction_method"] + "_and_full_page_visual_review",
                "human_or_expert": False, "entire_procurement_file_reviewed": False})
    summary = json.loads((base / "summary.json").read_text())
    summary.update(second_pass_full_supplied_pdf=sum(r["second_pass_status"] == "full_supplied_pdf" for r in rows),
        insufficient_source_agreements=sum(bool(r.get("source_insufficient_confirmed_two_pass")) for r in rows),
        completed_first_readings_in_this_block=0, partial_readings_recorded_in_this_block=0,
        new_second_passes_in_this_block=len(review["documents"]))
    verify_hashes(inputs)
    output.mkdir(parents=True)
    for filename in ("compiled_source_reviews.json", "partial_source_reviews.json"):
        shutil.copyfile(base / filename, output / filename)
    dump(output / "review_progress.json", rows)
    dump(output / "second_pass_evidence.json", evidence_rows)
    dump(output / "summary.json", summary)
    dump(output / "second_source_review.json", review)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return {k: summary[k] for k in ("first_pass_full_supplied_pdf", "second_pass_full_supplied_pdf", "insufficient_source_agreements", "f1_estimable")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "source", "first", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(*(getattr(args, name).resolve() for name in ("base", "source", "first", "review", "output")))))
