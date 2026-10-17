import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v2 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v31"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case33_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case33_preserves_both_conditional_branches_and_page_provenance():
    review, sources, first = inputs()
    validate(review, sources, first)
    doc = review["documents"][0]
    assert doc["pages_read"] == doc["complete_rendered_pages_read"] == list(range(1, 70))
    assert doc["ocr_pages_read"] == list(range(23, 34))
    assert doc["native_pages_read"] == list(range(1, 23)) + list(range(34, 70))
    assert len(doc["rendered_pages"]) == 69
    general, mype = doc["bidder_experience_conditions"]
    assert (general["branch"], general["billed_amount"]) == ("general", "480000.00")
    assert (mype["branch"], mype["billed_amount"]) == ("mype", "100000.00")
    assert mype["activation_condition"] == "Declaración MYPE en Anexo 1. Todos los integrantes deben ser MYPE en consorcio."
    for condition in (general, mype):
        assert condition["lookback_years"] == 15
        assert condition["currency"] == "PEN"
        assert condition["window_anchor"] == "conformidad_o_emision_del_comprobante"
        assert condition["reference_date"] == "presentacion_de_ofertas"
        assert condition["item"] is None
        assert condition["company_applicability"] == "not_determined"
        assert condition["document_reference_complete"] is False
    evidence = {e["id"]: e for e in doc["evidence"]}
    assert "veinte (20)" in evidence["max_contracts"]["quote"]
    assert evidence["guarantee_amount"]["page"] == 56
    assert evidence["personnel_score"]["page"] == 34
    assert evidence["empty_mype"]["page"] == 45
    assert "cancelación" in evidence["annex_date"]["quote"]
    assert "consorcios, todos los integrantes" in evidence["mype_window_and_consortium"]["quote"]
    assert review["independent_reviewer"] is False


def test_case33_agreement_is_field_and_activation_scoped_not_a_locked_reference():
    review, _, first = inputs()
    doc = review["documents"][0]
    comparison = compare_conditions(first[doc["opportunity_id"]]["bidder_experience_conditions"], doc["bidder_experience_conditions"])
    assert len(comparison) == 2
    assert all(c["first_pass_present"] and c["second_pass_present"] for c in comparison)
    assert all(v["status"] == "match" for c in comparison for k, v in c["fields"].items() if k != "activation_condition")
    assert comparison[0]["fields"]["activation_condition"]["status"] == "unspecified_in_both"
    assert comparison[1]["fields"]["activation_condition"]["status"] == "match"
    assert doc["decision"] == "source_conditions_agree_with_warnings_pending_reference_lock"
    assert "reference_adjudication_pending" in {w["code"] for w in doc["source_warnings"]}


@pytest.mark.parametrize("mutation", [
    "amount_change", "activation_change", "personnel_subject", "invented_quote",
    "duplicated_condition", "missing_last_page", "missing_scan_image", "fake_ocr", "eligibility_claim",
])
def test_unsafe_case33_registration_changes_are_rejected(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    condition = doc["bidder_experience_conditions"][0]
    if mutation == "amount_change": condition["billed_amount"] = "100000.00"
    elif mutation == "activation_change": doc["bidder_experience_conditions"][1]["activation_condition"] = "Todo postor."
    elif mutation == "personnel_subject": condition["subject"] = "personnel"
    elif mutation == "invented_quote": doc["evidence"].append({"id": "invented", "page": 69, "quote": "NO EXISTE EXPERIENCIA"})
    elif mutation == "duplicated_condition": doc["bidder_experience_conditions"].append(copy.deepcopy(condition))
    elif mutation == "missing_last_page": doc["pages_read"].remove(69)
    elif mutation == "missing_scan_image": doc["complete_rendered_pages_read"].remove(30)
    elif mutation == "fake_ocr":
        doc["native_pages_read"].remove(61)
        doc["ocr_pages_read"].append(61)
    elif mutation == "eligibility_claim": condition["company_applicability"] = "eligible"
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_build_appends_twelfth_rereading_without_changing_previous_inventories(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 12
    assert result["condition_source_agreements_pending_reference"] == 4
    assert result["f1_estimable"] is False
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    prior = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == prior
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    assert sum(r["first_pass_status"] == "full_supplied_pdf" and r["second_pass_status"] == "not_performed" for r in rows) == 15
    case = next(r for r in rows if r["sample_number"] == 33)
    assert case["second_pass_status"] == "full_supplied_pdf"
    assert case["reference_adjudication_pending"] is True
    assert all(r["historical_ranking_eligible"] is False and r["reference_locked"] is False for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
