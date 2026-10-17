"""Verify an automatic review's quotations, not its semantic correctness.

Verifying hashes and offsets does not count as the second source-reading pass.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]


def quotations(value):
    if isinstance(value, dict):
        if "quote" in value and "char_start" in value:
            yield value
        for child in value.values():
            yield from quotations(child)
    elif isinstance(value, list):
        for child in value:
            yield from quotations(child)


def check_quote(page, evidence):
    basis = evidence["quote_basis"]
    if basis == "native_text_verbatim":
        text = page["native_text"]
    elif basis == "ocr_text_verbatim_not_error_free" and page["ocr_used"]:
        text = page["text"]
    else:
        raise ValueError("Unknown or unavailable quotation basis")
    start, end = evidence["char_start"], evidence["char_end"]
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(text):
        raise ValueError("Invalid quotation span")
    if hashlib.sha256(text.encode()).hexdigest() != evidence["text_sha256"] or text[start:end] != evidence["quote"]:
        raise ValueError("Quotation differs from page text, hash or offsets")


def build(review_path, source_dir, output):
    if output.exists():
        raise ValueError("Use a new verification directory")
    inputs = dict(protected_outputs())
    verify_artifact(source_dir, inputs)
    review = json.loads(review_path.read_text())
    for field in ("human_annotation", "expert_annotation", "predictions_consulted", "model_rules_consulted",
                  "second_pass_performed", "reference_locked", "evaluation_ready"):
        if review.get(field) is not False:
            raise ValueError("Unexpected first-pass automatic-review attestation")
    inputs[str(review_path)] = sha(review_path)
    by_id = {d["opportunity_id"]: d for d in json.loads((source_dir / "source_documents.json").read_text())}
    checked, coverage = [], []
    seen = set()
    for doc in review["documents"]:
        if doc["opportunity_id"] in seen:
            raise ValueError("Duplicate review document")
        seen.add(doc["opportunity_id"])
        source = by_id[doc["opportunity_id"]]
        pdf_path = Path(doc["pdf_path"]).resolve()
        if pdf_path != Path(source["source"]["file_path"]).resolve() or doc["pdf_sha256"] != source["source"]["sha256"]:
            raise ValueError("Review PDF lineage differs")
        inputs[str(pdf_path)] = doc["pdf_sha256"]
        if doc["total_pages"] != len(source["pages"]):
            raise ValueError("Review physical page count differs")
        page_map = {p["page"]: p for p in source["pages"]}
        claimed = doc["coverage"]
        examined = set(claimed["native_pages_read"]) | set(claimed["complete_rendered_pages_read"]) | set(claimed.get("ocr_pages_read", []))
        if not examined <= set(page_map) or set(claimed["unreviewed_pages"]) != set(page_map) - examined:
            raise ValueError("Declared coverage and unreviewed pages do not reconcile")
        if claimed["full_pdf_semantically_read"] and examined != set(page_map):
            raise ValueError("Full document claimed with uncovered pages")
        for number, evidence in enumerate(quotations(doc), 1):
            page = page_map[evidence["page"]]
            check_quote(page, evidence)
            if evidence["visual_reviewed"]:
                image = ROOT / evidence["render_path"]
                inputs[str(image)] = sha(image)
            checked.append({"opportunity_id": doc["opportunity_id"], "evidence_number": number,
                "physical_page": evidence["page"], "quote_basis": evidence["quote_basis"],
                "literal_hash_offsets_verified": True,
                "visual_review_claim_recorded": evidence["visual_reviewed"],
                "semantic_correctness_independently_proven": False})
        coverage.append({"opportunity_id": doc["opportunity_id"], "sample_number": doc["sample_number"],
            "total_pages": len(page_map), "first_pass_pages_declared": len(examined),
            "first_pass_full_read_declared": claimed["full_pdf_semantically_read"],
            "second_pass_complete": False, "reference_complete": False,
            "scope_decision_from_first_pass": doc["decision"],
            "conditions_in_first_pass_inventory": len(doc["bidder_experience_conditions"])})
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "verified_quotations.json", checked)
    dump(output / "review_coverage.json", coverage)
    summary = {"documents": len(coverage), "literal_quotations_verified": len(checked),
        "first_pass_full_documents_declared": sum(r["first_pass_full_read_declared"] for r in coverage),
        "conditions_in_first_pass_inventory": sum(r["conditions_in_first_pass_inventory"] for r in coverage),
        "second_source_pass_performed_by_this_verifier": False,
        "reference_locked": False, "model_accuracy_evaluated": False,
        "verification_scope": "literal_provenance_and_coverage_consistency_not_semantic_gold"}
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("review", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.review.resolve(), args.source.resolve(), args.output.resolve())))
