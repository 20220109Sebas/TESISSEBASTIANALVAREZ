import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v2 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v30"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case8_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case8_retains_subjects_and_template_uncertainty():
    review, sources, first = inputs()
    validate(review, sources, first)
    doc = review["documents"][0]
    assert doc["pages_read"] == doc["native_pages_read"] == doc["complete_rendered_pages_read"] == list(range(1, 61))
    assert doc["ocr_pages_read"] == []
    assert len(doc["bidder_experience_conditions"]) == 1
    condition = doc["bidder_experience_conditions"][0]
    assert condition["billed_amount"] == "100000.00"
    assert condition["currency"] == "PEN"
    assert condition["lookback_years"] == 10
    assert condition["branch"] == "primary_paragraph"
    assert condition["window_anchor"] == "conformidad_o_emision_del_comprobante"
    assert condition["reference_date"] == "presentacion_de_ofertas"
    assert condition["evidence_ids"] == ["bidder_amount_and_window", "similar_services"]
    evidence = {e["id"]: e for e in doc["evidence"]}
    assert "OCHO AÑOS" in evidence["template_annex_window"]["quote"]
    assert "cuatro (04)" in evidence["key_personnel_years"]["quote"]
    assert "50,000.00" in evidence["contract_amount"]["quote"]
    assert "máximo de veinte" in evidence["max_contracts"]["quote"]
    assert "dos entregables" in evidence["deliverables_intro"]["quote"]
    assert "Tercer Entregable" in evidence["third_deliverable"]["quote"]
    assert condition["company_applicability"] == "not_determined"
    assert condition["document_reference_complete"] is False
    assert review["independent_reviewer"] is False


def test_schema_difference_cannot_be_reported_as_complete_agreement():
    review, _, first = inputs()
    doc = review["documents"][0]
    previous = first[doc["opportunity_id"]]["bidder_experience_conditions"]
    assert previous[0]["billed_amount"] == doc["bidder_experience_conditions"][0]["billed_amount"]
    assert previous[0]["lookback_years_explicit_in_qualification"] == 10
    assert previous[0]["lookback_years_for_locked_evaluation"] is None
    comparison = compare_conditions(previous, doc["bidder_experience_conditions"])
    assert any(not c["first_pass_present"] or not c["second_pass_present"] for c in comparison)
    assert doc["decision"] == "source_conditions_discrepant_pending_reference_lock"


@pytest.mark.parametrize("mutation", [
    "false_agreement", "personnel_subject", "invented_empty_page_quote", "duplicated_condition",
    "missing_last_page", "missing_evidence_image", "fake_ocr", "eligibility_claim", "locked_reference",
])
def test_unsafe_case8_registration_changes_are_rejected(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    condition = doc["bidder_experience_conditions"][0]
    if mutation == "false_agreement": doc["decision"] = "source_conditions_agree_with_warnings_pending_reference_lock"
    elif mutation == "personnel_subject": condition["subject"] = "personnel"
    elif mutation == "invented_empty_page_quote": doc["evidence"].append({"id": "invented", "page": 60, "quote": "SIN REQUISITOS EMPRESARIALES"})
    elif mutation == "duplicated_condition": doc["bidder_experience_conditions"].append(copy.deepcopy(condition))
    elif mutation == "missing_last_page": doc["pages_read"].remove(60)
    elif mutation == "missing_evidence_image": doc["complete_rendered_pages_read"].remove(59)
    elif mutation == "fake_ocr":
        doc["native_pages_read"].remove(59)
        doc["ocr_pages_read"].append(59)
    elif mutation == "eligibility_claim": condition["company_applicability"] = "eligible"
    elif mutation == "locked_reference": condition["document_reference_complete"] = True
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_build_appends_eleventh_rereading_and_preserves_previous_inventories(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 11
    assert result["condition_source_agreements_pending_reference"] == 3
    assert result["f1_estimable"] is False
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    prior = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == prior
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    assert sum(r["first_pass_status"] == "full_supplied_pdf" and r["second_pass_status"] == "not_performed" for r in rows) == 16
    case = next(r for r in rows if r["sample_number"] == 8)
    assert case["second_pass_status"] == "full_supplied_pdf"
    assert case["reference_adjudication_pending"] is True
    assert all(r["historical_ranking_eligible"] is False and r["reference_locked"] is False for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
