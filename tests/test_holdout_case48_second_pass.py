import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v2 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v40"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case48_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case48_two_branches_preserve_scoped_amounts_and_window():
    review, sources, first = inputs()
    validate(review, sources, first)
    doc = review["documents"][0]
    assert doc["pages_read"] == list(range(1, 89))
    assert len(doc["native_pages_read"]) == 61 and len(doc["ocr_pages_read"]) == 27
    assert doc["ocr_pages_read"] == [3, *range(35, 60), 64]
    assert len(doc["complete_rendered_pages_read"]) == len(doc["rendered_pages"]) == 46
    assert set(doc["ocr_pages_read"]) <= set(doc["complete_rendered_pages_read"])
    assert doc["all_pages_visually_reviewed"] is False
    assert doc["all_content_proven_recovered"] is False
    assert len(doc["native_pages_not_rendered"]) == 42
    conditions = {c["branch"]: c for c in doc["bidder_experience_conditions"]}
    assert set(conditions) == {"general", "mype"}
    assert conditions["general"]["billed_amount"] == "1200000.00"
    assert conditions["mype"]["billed_amount"] == "118000.00"
    assert "Todos los integrantes" in conditions["mype"]["activation_condition"]
    assert conditions["general"]["activation_condition"] is None
    for condition in conditions.values():
        assert condition["currency"] == "PEN" and condition["lookback_years"] == 8
        assert condition["reference_date"] == "presentacion_de_ofertas"
        assert condition["window_anchor"] == "conformidad_o_emision_del_comprobante_de_pago"
        assert condition["company_applicability"] == "not_determined"
        assert condition["document_reference_complete"] is False
        assert len(condition["similar_services"]) == 5
        assert "ACOMETIDA" in condition["similar_services"][2]
        assert "UNIVERSIDADES" in condition["similar_services"][-1]
        assert not any("REHABILITACION" in service for service in condition["similar_services"])
    evidence = {e["id"]: e for e in doc["evidence"]}
    assert evidence["general_qualification_start"]["page"] == 61
    assert evidence["general_qualification_continuation"]["page"] == 62
    assert evidence["general_qualification_continuation"]["quote"].startswith("Doscientos Mil")
    assert doc["document_level_lookback_years"] == 8 and doc["conflicting_source_periods"] == []


def test_case48_personnel_forms_and_source_identity_stay_separate():
    review, _, _ = inputs()
    doc = review["documents"][0]
    evidence = {e["id"]: e for e in doc["evidence"]}
    assert len(evidence) == 30
    assert "CINCO (05)" in evidence["electrical_personnel"]["quote"]
    assert "TRES (03)" in evidence["safety_personnel"]["quote"]
    assert "veinticinco (25)" in evidence["personnel_window"]["quote"]
    assert "REHABILITACION" in evidence["personnel_similar_services"]["quote"]
    assert "ocho (8)" in evidence["annex_window"]["quote"]
    assert "cancelación" in evidence["annex_dates"]["quote"]
    assert "máximo de veinte" in evidence["accreditation"]["quote"]
    accreditation = doc["accreditation_observations"]
    assert accreditation["maximum_contracts"] == 20 and accreditation["minimum_contracts"] is None
    assert accreditation["private_contracts_require_payment_evidence"] is None
    assert accreditation["self_placed_paid_stamp_sufficient"] is False
    assert accreditation["consortium_only_executing_members"] is True
    consortium = doc["consortium_observations"]
    assert consortium["maximum_members"] is consortium["minimum_each_percent"] is None
    assert consortium["minimum_most_experienced_percent"] is None
    assert consortium["blank_two_member_template_is_not_maximum"] is True
    assert "20.09.2012" in evidence["legacy_consortium_rule"]["quote"]
    assert "SEGUNDA CONVOCATORIA" in evidence["proforma_identity"]["quote"]
    assert "TERCERA CONVOCATORIA" in evidence["annex_identity"]["quote"]
    assert all(r["number_matches"] and r["procedure_stage_conflict"] and not r["resolved"] for r in doc["annex_cross_references"])
    assert "[CONSIGNAR MONEDA Y MONTO ]" in evidence["blank_contract_amount"]["quote"]
    assert all(v["replaces_archived_text"] is False for v in doc["visual_observations"])
    excluded = {e["evidence_id"] for e in doc["excluded_as_bidder_experience"]}
    assert {"electrical_personnel", "safety_personnel", "personnel_window", "personnel_similar_services",
            "mype_bonus", "blank_contract_amount", "supply_improvement", "consortium_template"} <= excluded


def test_case48_partial_field_matches_do_not_claim_full_agreement():
    review, _, first = inputs()
    doc = review["documents"][0]
    prior = first[doc["opportunity_id"]]["bidder_experience_conditions"]
    before = copy.deepcopy(prior)
    comparisons = compare_conditions(prior, doc["bidder_experience_conditions"])
    assert prior == before and len(comparisons) == 2
    for comparison in comparisons:
        assert comparison["first_pass_present"] and comparison["second_pass_present"]
        assert all(comparison["fields"][field]["status"] == "match" for field in (
            "billed_amount", "currency", "lookback_years", "window_anchor", "reference_date"))
        assert comparison["fields"]["similar_services"]["status"] == "unknown"
    mype = next(c for c in comparisons if c["identity"]["branch"] == "mype")
    assert mype["fields"]["activation_condition"]["status"] == "unknown"
    assert "applicability" in prior[1] and "activation_condition" not in prior[1]
    assert doc["decision"] == "source_conditions_discrepant_pending_reference_lock"


@pytest.mark.parametrize("mutation", [
    "false_agreement", "missing_last_page", "missing_ocr_image", "fake_native",
    "personnel_subject", "invented_quote", "eligibility_claim", "reference_lock",
    "wrong_pdf", "bad_currency", "duplicate_render", "unknown_evidence", "duplicate_condition",
])
def test_case48_registration_rejects_unsafe_changes(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    condition = doc["bidder_experience_conditions"][0]
    if mutation == "false_agreement":
        doc["decision"] = "source_conditions_agree_with_warnings_pending_reference_lock"
    elif mutation == "missing_last_page":
        doc["pages_read"].remove(88)
    elif mutation == "missing_ocr_image":
        doc["complete_rendered_pages_read"].remove(59)
    elif mutation == "fake_native":
        doc["ocr_pages_read"].remove(59)
        doc["native_pages_read"].append(59)
    elif mutation == "personnel_subject":
        condition["subject"] = "personnel"
    elif mutation == "invented_quote":
        doc["evidence"].append({"id": "fake", "page": 88, "quote": "SIN EXPERIENCIA EXIGIDA"})
    elif mutation == "eligibility_claim":
        condition["company_applicability"] = "eligible"
    elif mutation == "reference_lock":
        review["reference_locked"] = True
    elif mutation == "wrong_pdf":
        doc["pdf_sha256"] = "0" * 64
    elif mutation == "bad_currency":
        condition["currency"] = "S/"
    elif mutation == "duplicate_render":
        doc["rendered_pages"][1]["path"] = doc["rendered_pages"][0]["path"]
    elif mutation == "unknown_evidence":
        condition["evidence_ids"].append("missing")
    elif mutation == "duplicate_condition":
        doc["bidder_experience_conditions"].append(copy.deepcopy(condition))
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_case48_build_appends_twenty_first_reading_without_unlocking_evaluation(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 21
    assert result["condition_source_agreements_pending_reference"] == 4
    assert result["f1_estimable"] is False
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    previous = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == previous and current[-1] == json.loads(REVIEW.read_text())
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    pending = [r["sample_number"] for r in rows if r["first_pass_status"] == "full_supplied_pdf" and r["second_pass_status"] == "not_performed"]
    assert pending == [1, 11, 18, 21, 34, 40]
    case = next(r for r in rows if r["sample_number"] == 48)
    assert case["second_pass_status"] == "full_supplied_pdf" and case["reference_adjudication_pending"] is True
    assert len(case["second_pass_field_comparison"]) == 2
    assert all(not r["historical_ranking_eligible"] and not r["reference_locked"] for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
    evidence = [e for e in json.loads((output / "second_pass_evidence.json").read_text()) if e["opportunity_id"] == case["opportunity_id"]]
    assert len(evidence) == 30
    assert all(e["image_sha256"] and not e["human_or_expert"] and not e["independent_reviewer"] for e in evidence)
    assert all(e["method"] == "native_pdf_text_and_full_page_visual_review" for e in evidence)
