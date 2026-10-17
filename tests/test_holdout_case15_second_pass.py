import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v2 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v33"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case15_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case15_complete_reading_deduplicates_repeated_qualification():
    review, sources, first = inputs()
    validate(review, sources, first)
    doc = review["documents"][0]
    assert doc["pages_read"] == doc["complete_rendered_pages_read"] == list(range(1, 72))
    assert doc["ocr_pages_read"] == list(range(26, 45))
    assert doc["native_pages_read"] == list(range(1, 26)) + list(range(45, 72))
    assert len(doc["rendered_pages"]) == 71
    assert len(doc["bidder_experience_conditions"]) == 1
    condition = doc["bidder_experience_conditions"][0]
    assert (condition["branch"], condition["billed_amount"], condition["currency"]) == ("general", "700000.00", "PEN")
    assert condition["lookback_years"] == 8
    assert condition["window_anchor"] == "conformidad_o_emision_del_comprobante"
    assert condition["reference_date"] == "presentacion_de_ofertas"
    assert condition["item"] is None
    assert condition["same_condition_repeated_on_pages"] == [42, 45]
    assert condition["company_applicability"] == "not_determined"
    assert condition["document_reference_complete"] is False
    evidence = {e["id"]: e for e in doc["evidence"]}
    assert evidence["native_amount_window"]["page"] == 45
    assert evidence["ocr_repeated_amount_window"]["page"] == 42
    assert evidence["worker_experience"]["page"] == 36
    assert "US$ 10,000.00" in evidence["insurance_amount"]["quote"]
    assert "veinte (20)" in evidence["max_contracts"]["quote"]
    assert "cancelación" in evidence["annex_cancelled_date"]["quote"]
    assert review["independent_reviewer"] is False


def test_case15_legacy_schema_is_not_reported_as_full_agreement():
    review, _, first = inputs()
    doc = review["documents"][0]
    prior = first[doc["opportunity_id"]]["bidder_experience_conditions"]
    current = doc["bidder_experience_conditions"]
    for field in ("billed_amount", "currency", "lookback_years", "item"):
        assert prior[0][field] == current[0][field]
    assert "section_role" not in prior[0]
    assert "window_anchor" not in prior[0]
    assert "reference_date" not in prior[0]
    assert "similar_services" not in prior[0]
    assert prior[0]["branch"] == "primary_paragraph"
    assert prior[0]["lookback_anchor"] == "before_offer_submission; conformity_or_payment_document_issue"
    comparison = compare_conditions(prior, current)
    assert len(comparison) == 2
    assert all(c["first_pass_present"] != c["second_pass_present"] for c in comparison)
    assert all(v["status"] == "unknown" for c in comparison for v in c["fields"].values())
    assert doc["decision"] == "source_conditions_discrepant_pending_reference_lock"
    warnings = {w["code"] for w in doc["source_warnings"]}
    assert {"legacy_schema_gap", "duplicate_qualification_clause", "security_cleaning_source_mismatch",
            "date_anchor_difference", "mype_no_reduced_branch_observed", "reference_adjudication_pending"} <= warnings


@pytest.mark.parametrize("mutation", [
    "amount_format", "false_agreement", "personnel_subject", "invented_quote",
    "duplicated_condition", "missing_last_page", "missing_scan_image", "fake_ocr", "eligibility_claim",
])
def test_unsafe_case15_registration_changes_are_rejected(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    condition = doc["bidder_experience_conditions"][0]
    if mutation == "amount_format": condition["billed_amount"] = "700,000"
    elif mutation == "false_agreement": doc["decision"] = "source_conditions_agree_with_warnings_pending_reference_lock"
    elif mutation == "personnel_subject": condition["subject"] = "personnel"
    elif mutation == "invented_quote": doc["evidence"].append({"id": "invented", "page": 71, "quote": "NO EXISTE EXPERIENCIA"})
    elif mutation == "duplicated_condition": doc["bidder_experience_conditions"].append(copy.deepcopy(condition))
    elif mutation == "missing_last_page": doc["pages_read"].remove(71)
    elif mutation == "missing_scan_image": doc["complete_rendered_pages_read"].remove(42)
    elif mutation == "fake_ocr":
        doc["native_pages_read"].remove(45)
        doc["ocr_pages_read"].append(45)
    elif mutation == "eligibility_claim": condition["company_applicability"] = "eligible"
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_build_appends_fourteenth_rereading_preserving_prior_inventories(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 14
    assert result["condition_source_agreements_pending_reference"] == 4
    assert result["f1_estimable"] is False
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    prior = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == prior
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    assert sum(r["first_pass_status"] == "full_supplied_pdf" and r["second_pass_status"] == "not_performed" for r in rows) == 13
    case = next(r for r in rows if r["sample_number"] == 15)
    assert case["second_pass_status"] == "full_supplied_pdf"
    assert case["reference_adjudication_pending"] is True
    assert len(case["second_pass_field_comparison"]) == 2
    assert all(r["historical_ranking_eligible"] is False and r["reference_locked"] is False for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
