"""Append source rereadings with explicit render provenance and branch activation checks.

Keeps the v1 registrar intact for reproduction of previous checkpoints.
Image inventories describe actual viewing, not an automatically inferred reading.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import validate_source_document, verify_artifact
from scripts import record_holdout_condition_second_pass as v1
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]


def compare_conditions(prior, current):
    comparisons = v1.compare_conditions(prior, current)
    left, right = v1.keyed_conditions(prior), v1.keyed_conditions(current)
    for comparison in comparisons:
        key = tuple(comparison["identity"][field] for field in v1.IDENTITY)
        a, b = left.get(key), right.get(key)
        x, y = a.get("activation_condition") if a else None, b.get("activation_condition") if b else None
        status = "unspecified_in_both" if a is not None and b is not None and x is None and y is None else (
            "match" if x is not None and y is not None and x == y else "unknown" if x is None or y is None else "disagreement")
        comparison["fields"]["activation_condition"] = {"first": x, "second": y, "status": status}
    return comparisons


def validate(review, sources, first):
    v1.validate(review, sources, first)
    for doc in review["documents"]:
        source = sources[doc["opportunity_id"]]
        validate_source_document(source)
        if len(source["pages"]) != doc["physical_pages"]:
            raise ValueError("Physical page count mismatch")
        native = {p["page"] for p in source["pages"] if not p["ocr_used"]}
        ocr = {p["page"] for p in source["pages"] if p["ocr_used"]}
        if set(doc["native_pages_read"]) != native or set(doc["ocr_pages_read"]) != ocr:
            raise ValueError("Text method coverage differs from source")
        renders = doc["rendered_pages"]
        if len(renders) != len(doc["complete_rendered_pages_read"]) or {r["page"] for r in renders} != set(doc["complete_rendered_pages_read"]):
            raise ValueError("Each viewed page needs its render")
        paths = []
        for rendered in renders:
            path = (ROOT / rendered["path"]).resolve()
            if not path.is_relative_to(ROOT / "tmp/pdfs") or not path.is_file() or path.suffix != ".png" or not rendered.get("renderer"):
                raise ValueError("Render path or renderer invalid")
            paths.append(path)
        if len(set(paths)) != len(paths):
            raise ValueError("Do not reuse one image for multiple pages")
        if doc["decision"] == "source_conditions_agree_with_warnings_pending_reference_lock":
            comparisons = compare_conditions(first[doc["opportunity_id"]]["bidder_experience_conditions"], doc["bidder_experience_conditions"])
            if any(c["fields"]["activation_condition"]["status"] not in ("match", "unspecified_in_both") for c in comparisons):
                raise ValueError("Branch activation does not agree")


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
    for path in (review_path, Path(v1.__file__).resolve(), Path(__file__).resolve()):
        inputs[str(path)] = sha(path)
    rows = json.loads((base / "review_progress.json").read_text())
    if len(rows) != 50 or len({r["opportunity_id"] for r in rows}) != 50:
        raise ValueError("Preserve fifty-case denominator")
    summary = json.loads((base / "summary.json").read_text())
    if any(summary.get(flag) is not False for flag in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified")) or summary.get("positive_document_references_complete") != 0:
        raise ValueError("This registrar cannot alter an evaluated or locked reference")
    by_id = {r["opportunity_id"]: r for r in rows}
    evidence_rows = json.loads((base / "second_pass_evidence.json").read_text())
    for doc in review["documents"]:
        row, source = by_id[doc["opportunity_id"]], sources[doc["opportunity_id"]]
        if (row["second_pass_status"] != "not_performed" or row["first_pass_status"] != "full_supplied_pdf"
                or row["source_status"] != "downloaded_unique_pdf" or row["physical_pages"] != doc["physical_pages"]
                or row["reference_locked"] is not False or row["predictions_generated"] is not False
                or row["historical_ranking_eligible"] is not False):
            raise ValueError("Cannot replace readings or unlock protected use")
        row.update(second_pass_status="full_supplied_pdf", two_pass_agreement=doc["decision"],
            second_pass_reviewer=review["reviewer"], second_pass_independent_reviewer=False,
            second_pass_independence_note=review["independence_note"], second_pass_source_warnings=doc["source_warnings"],
            reference_adjudication_pending=True, second_pass_field_comparison=compare_conditions(
                first[doc["opportunity_id"]]["bidder_experience_conditions"], doc["bidder_experience_conditions"]))
        renders = {r["page"]: r for r in doc["rendered_pages"]}
        for rendered in renders.values():
            path = (ROOT / rendered["path"]).resolve()
            inputs[str(path)] = sha(path)
        for item in doc["evidence"]:
            page = source["pages"][item["page"] - 1]
            start = page["text"].index(item["quote"])
            image = (ROOT / renders[item["page"]]["path"]).resolve()
            evidence_rows.append({"opportunity_id": doc["opportunity_id"], "evidence_id": item["id"],
                "page": item["page"], "quote": item["quote"], "char_start": start, "char_end": start + len(item["quote"]),
                "page_text_sha256": hashlib.sha256(page["text"].encode()).hexdigest(), "pdf_sha256": doc["pdf_sha256"],
                "image_path": str(image), "image_sha256": sha(image), "renderer": renders[item["page"]]["renderer"],
                "method": page["extraction_method"] + "_and_full_page_visual_review", "human_or_expert": False,
                "independent_reviewer": False, "entire_procurement_file_reviewed": False})
    summary.update(second_pass_full_supplied_pdf=sum(r["second_pass_status"] == "full_supplied_pdf" for r in rows),
        condition_source_agreements_pending_reference=sum(r.get("two_pass_agreement") == "source_conditions_agree_with_warnings_pending_reference_lock" for r in rows),
        completed_first_readings_in_this_block=0, partial_readings_recorded_in_this_block=0,
        new_second_passes_in_this_block=len(review["documents"]))
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
    for filename, value in (("review_progress.json", rows), ("second_pass_evidence.json", evidence_rows), ("summary.json", summary)):
        dump(output / filename, value)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return {k: summary[k] for k in ("first_pass_full_supplied_pdf", "second_pass_full_supplied_pdf", "condition_source_agreements_pending_reference", "f1_estimable")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "source", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(*(getattr(args, name).resolve() for name in ("base", "source", "review", "output")))))
