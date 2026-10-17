"""Add actual source-only scope rereadings, without changing prior artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]


def validate_scope_review(review, sources, first):
    if review.get("review_kind") != "automatic_source_only_scope_second_pass":
        raise ValueError("Unexpected scope review kind")
    for flag in ("human_annotation", "expert_annotation", "predictions_consulted", "model_rules_consulted", "reference_locked"):
        if review.get(flag) is not False:
            raise ValueError("Scope review is automatic, source-only and unlocked")
    seen = set()
    for doc in review["documents"]:
        ocid = doc["opportunity_id"]
        if ocid in seen:
            raise ValueError("Duplicate scope review")
        seen.add(ocid)
        source, previous = sources[ocid], first[ocid]
        expected = {p["page"] for p in source["pages"]}
        if (doc["sample_number"] != source["source"]["sample_number"] or doc["sample_number"] != previous["sample_number"]
            or doc.get("full_supplied_document_read") is not True
            or previous["coverage"].get("full_pdf_semantically_read") is not True):
            raise ValueError("Both passes must cover the same supplied document")
        for values in (doc["pages_read"], doc["complete_rendered_pages_read"]):
            if any(type(p) is not int for p in values) or len(set(values)) != len(values) or not set(values) <= expected:
                raise ValueError("Invalid physical-page coverage")
        if set(doc["pages_read"]) != expected or doc.get("entire_procurement_file_reviewed") is not False:
            raise ValueError("Incomplete PDF or overstated procurement-file coverage")
        if (doc["decision"] != "outside_business_services_domain" or previous["decision"] != doc["decision"]
            or previous["domain"] != "consultor_individual" or doc["bidder_experience_conditions"] != []):
            raise ValueError("This extension is only for agreed individual-professional scope decisions")
        academic = any("BACHILLER" in e["quote"] or "Perfil Académico" in e["quote"] for e in doc["evidence"])
        individual_cv = (any("método de Consultores Individuales" in e["quote"] for e in doc["evidence"])
            and any("comparación de Currículos Vitae" in e["quote"] for e in doc["evidence"]))
        if not (academic or individual_cv):
            raise ValueError("Person status alone cannot prove exclusion from enterprise services")
        for evidence in doc["evidence"]:
            page = next(p for p in source["pages"] if p["page"] == evidence["page"])
            if evidence["page"] not in doc["complete_rendered_pages_read"] or page["text"].count(evidence["quote"]) != 1:
                raise ValueError("Scope quotation missing, ambiguous or not visually reviewed")
        if "rendered_pages" in doc:
            renders = doc["rendered_pages"]
            visual = set(doc["complete_rendered_pages_read"])
            if (len(renders) != len(visual) or any(type(r.get("page")) is not int for r in renders)
                    or {r["page"] for r in renders} != visual
                    or any(not r.get("render_path") for r in renders)):
                raise ValueError("Each visual page needs its saved render")
            by_page = {r["page"]: r["render_path"] for r in renders}
            if any(e["image"] != by_page[e["page"]] for e in doc["evidence"]):
                raise ValueError("Quote image differs from render inventory")


def update_second_pass_summary(summary, rows, review):
    """Keep source warnings while retiring only the completed pending-pass marker."""
    completed = {d["sample_number"]: d for d in review["documents"]}
    summary.update(second_pass_full_supplied_pdf=sum(r["second_pass_status"] == "full_supplied_pdf" for r in rows),
        outside_business_scope_agreements=sum(r.get("outside_business_scope_confirmed_two_pass", False) for r in rows),
        completed_first_readings_in_this_block=0, partial_readings_recorded_in_this_block=0,
        new_second_passes_in_this_block=len(completed), previous_progress_preserved=True)
    for item in summary.get("unresolved_cross_review_issues", []):
        doc = completed.get(item.get("sample_number"))
        if doc is None:
            continue
        item["second_full_source_pass_pending"] = False
        item["source_scope_reread_completed"] = True
        item["independent_reviewer"] = review.get("independent_reviewer")
        # Prior issues remain inspectable in the base artifact and first-pass notes.
        item["source_warnings"] = doc.get("source_warnings", item.get("source_warnings", []))


def build(base, source_dir, first_path, review_path, output):
    if output.exists():
        raise ValueError("Use a new progress directory")
    inputs = dict(protected_outputs())
    for directory in (base, source_dir):
        verify_artifact(directory, inputs)
    # Include all pre-existing input locks and source decisions in the new manifest.
    for directory in (base, source_dir):
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    first, review = json.loads(first_path.read_text()), json.loads(review_path.read_text())
    inputs[str(first_path)], inputs[str(review_path)] = sha(first_path), sha(review_path)
    sources = {d["opportunity_id"]: d for d in json.loads((source_dir / "source_documents.json").read_text())}
    first_by_id = {d["opportunity_id"]: d for d in first["documents"]}
    validate_scope_review(review, sources, first_by_id)
    rows = json.loads((base / "review_progress.json").read_text())
    if len(rows) != 50 or len({r["opportunity_id"] for r in rows}) != 50:
        raise ValueError("Preserve the fifty-case denominator")
    by_id = {r["opportunity_id"]: r for r in rows}
    evidence_rows = json.loads((base / "second_pass_evidence.json").read_text())
    for doc in review["documents"]:
        row, source = by_id[doc["opportunity_id"]], sources[doc["opportunity_id"]]
        if row["second_pass_status"] != "not_performed":
            raise ValueError("Do not replace an existing second-pass judgment")
        row.update(second_pass_status="full_supplied_pdf", two_pass_agreement=doc["decision"],
            outside_business_scope_confirmed_two_pass=True,
            second_pass_reviewer=review["reviewer"],
            second_pass_independent_reviewer=review.get("independent_reviewer"),
            second_pass_independence_note=review.get("independence_note"),
            second_pass_source_warnings=doc.get("source_warnings", []))
        for rendered in doc.get("rendered_pages", []):
            image = ROOT / rendered["render_path"]
            inputs[str(image)] = sha(image)
        for evidence in doc["evidence"]:
            page = next(p for p in source["pages"] if p["page"] == evidence["page"])
            start = page["text"].index(evidence["quote"])
            image = ROOT / evidence["image"]
            inputs[str(image)] = sha(image)
            evidence_rows.append({"opportunity_id": doc["opportunity_id"], "page": page["page"],
                "quote": evidence["quote"], "char_start": start, "char_end": start + len(evidence["quote"]),
                "page_text_sha256": hashlib.sha256(page["text"].encode()).hexdigest(),
                "pdf_sha256": source["source"]["sha256"], "image_path": str(image), "image_sha256": sha(image),
                "method": page["extraction_method"] + "_and_full_page_visual_review",
                "human_or_expert": False, "entire_procurement_file_reviewed": False})
    summary = json.loads((base / "summary.json").read_text())
    update_second_pass_summary(summary, rows, review)
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "review_progress.json", rows)
    dump(output / "second_pass_evidence.json", evidence_rows)
    dump(output / "summary.json", summary)
    dump(output / "second_source_review.json", review)
    # An append-only second pass must not drop the current first-pass inventories.
    for filename in ("compiled_source_reviews.json", "partial_source_reviews.json"):
        if (base / filename).exists():
            shutil.copyfile(base / filename, output / filename)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "source", "first", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(*(getattr(args, name).resolve() for name in ("base", "source", "first", "review", "output")))))
