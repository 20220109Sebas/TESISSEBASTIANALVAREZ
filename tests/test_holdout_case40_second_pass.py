import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v5 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v44"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case40_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case40_actual_coverage_and_disclosed_limits():
    review, sources, first = inputs()
    validate(review, sources, first)
    d = review["documents"][0]
    assert d["pages_read"] == list(range(1, 95))
    assert d["ocr_pages_read"] == list(range(42, 47))
    assert len(d["native_pages_read"]) == 89
    assert len(d["complete_rendered_pages_read"]) == len(d["rendered_pages"]) == 63
    assert len(d["native_pages_not_rendered"]) == 31
    assert not d["all_pages_visually_reviewed"] and not d["all_content_proven_recovered"]
    assert not d["new_ocr_executed"] and not d["entire_procurement_file_reviewed"]
    assert len(d["evidence"]) == 23 and d["unreviewed_pages"] == []
    assert all(not o["replaces_archived_text"] for o in d["visual_observations"])
    assert d["formal_services_evaluation_eligible"] is True
    assert d["domain"] == "servicios_transporte_documentos_confidenciales"
    assert all(not d[k] for k in ("historical_content_proven", "ranking_eligible", "predictions_generated"))


def test_case40_two_company_branches_not_four_repeated_conditions():
    review, _, _ = inputs()
    d = review["documents"][0]
    conditions = {c["branch"]: c for c in d["bidder_experience_conditions"]}
    assert set(conditions) == {"general", "mype"}
    assert conditions["general"]["billed_amount"] == "600000.00"
    assert conditions["mype"]["billed_amount"] == "50000.00"
    assert conditions["general"]["activation_condition"] is None
    assert conditions["mype"]["activation_condition"] == "Declaración MYPE en Anexo 1. Todos los integrantes del consorcio deben ser MYPE."
    for c in conditions.values():
        assert c["subject"] == "bidder" and c["section_role"] == "qualification"
        assert c["currency"] == "PEN" and c["lookback_years"] == 15
        assert c["reference_date"] == "presentacion_de_ofertas"
        assert c["window_anchor"] == "conformidad_o_emision_del_comprobante_de_pago"
        assert "educativos" in c["similar_services"] and "electorales" in c["similar_services"]
        assert "reservado o confidencial" in c["similar_services"]
        assert c["maximum_contracts_for_accreditation"] == 20 and c["minimum_contracts"] is None
        assert c["company_applicability"] == "not_determined" and not c["document_reference_complete"]
    repeat = d["cross_section_repetitions"][0]
    assert repeat["tdr_pages"] == [50, 51] and repeat["qualification_page"] == 54
    assert repeat["billed_amounts_match"] and repeat["lookback_years_match"]
    assert not repeat["counted_as_additional_conditions"] and not repeat["precedence_adjudicated"]


def test_case40_personnel_qualification_factor_and_warning_are_distinct():
    review, _, _ = inputs()
    d = review["documents"][0]
    qualification, factor, conductor, trainer = d["personnel_experience_observations"]
    assert qualification["minimum_years"] == 2 and qualification["lookback_years"] == 25
    assert qualification["overlap_counted_once"]
    assert factor["section_role"] == "evaluation_factor" and factor["lookback_years"] == 25
    assert [(b["lower_exclusive"], b["upper_inclusive"], b["points"]) for b in factor["score_bands"]] == [(7, None, 25), (4, 7, 15), (2, 4, 5)]
    assert conductor["minimum_license_age_years"] == 2 and conductor["minimum_experience_years"] is None
    assert conductor["license_class"] == "A-IIb" and not conductor["applied_to_bidder"]
    assert trainer["section_role"] == "template_warning" and trainer["minimum_years"] is None
    assert trainer["maximum_experience_years_permitted_in_warning"] == 2
    assert not trainer["explicit_concrete_minimum_selected"]
    assert d["document_level_lookback_years"] is None


def test_case40_other_money_does_not_fill_budget_or_experience():
    review, _, _ = inputs()
    d = review["documents"][0]
    by_role = {o["role"]: o for o in d["non_experience_amount_observations"]}
    assert by_role["procurement_budget"]["amount"] is None
    assert not by_role["procurement_budget"]["disclosed"]
    assert by_role["insurance_coverage"]["amount"] == "800000.00"
    assert by_role["estimated_cargo_value"]["amount"] == "500000.00"
    guarantee = by_role["conditional_guarantee_template"]
    assert guarantee["amount"] == "480000.00"
    assert guarantee["minimum_service_days"] == 60 and guarantee["minimum_payments"] == 2
    assert not guarantee["activated_from_template"]
    assert all(not o["applied_to_experience"] for role, o in by_role.items() if role != "procurement_budget")


def test_case40_conditional_forms_and_date_roles_do_not_become_history():
    review, _, _ = inputs()
    d = review["documents"][0]
    admission, annex = d["other_bidder_experience_observations"]
    assert admission["lookback_years"] == 2 and admission["activation_condition"]
    assert [c["minimum_contracts"] for c in admission["alternatives"]] == [1, 4]
    assert admission["consecutive_years_for_services"] and not admission["applied_to_general_specialty"]
    assert annex["rows_up_to"] == 20 and annex["lookback_years"] == 15
    assert annex["blank_template"] and not annex["actual_company_history_proven"]
    assert not annex["applied_as_new_qualification"]
    assert d["date_roles"]["qualification_anchor"] != d["date_roles"]["fx_and_annex_receipt_anchor"]
    assert not d["date_roles"]["collapsed"] and not d["date_roles"]["precedence_adjudicated"]
    accreditation = d["accreditation_observations"]
    assert accreditation["private_requires_payment_evidence"] and accreditation["multiple_receipts_require_contract_link"]
    assert accreditation["consortium_only_joint_executors"] and accreditation["reorganization_requires_support"]
    assert accreditation["reorganization_annex_number"] == accreditation["actual_annex_number"] == 14


def test_case40_missing_structured_first_pass_fields_stay_unknown():
    review, _, first = inputs()
    d = review["documents"][0]
    prior = first[d["opportunity_id"]]["bidder_experience_conditions"]
    before = copy.deepcopy(prior)
    comparisons = compare_conditions(prior, d["bidder_experience_conditions"])
    assert prior == before and len(comparisons) == 2
    for c in comparisons:
        assert c["fields"]["similar_services"]["status"] == "unknown"
        assert all(c["fields"][f]["status"] == "match" for f in ("billed_amount", "currency", "lookback_years", "window_anchor", "reference_date"))
    mype = next(c for c in comparisons if c["identity"]["branch"] == "mype")
    assert mype["fields"]["activation_condition"]["status"] == "unknown"
    assert prior[1]["applicability"] == d["first_pass_structural_gaps"]["first_pass_mype_applicability_retained"]
    assert not d["first_pass_structural_gaps"]["source_contradiction_in_main_amounts_or_window_observed"]
    assert d["decision"] == "source_conditions_discrepant_pending_reference_lock"


@pytest.mark.parametrize("mutation", [
    "agreement", "missing_page", "fake_ocr", "missing_image", "personnel",
    "invented_quote", "eligibility", "lock", "pdf", "currency", "duplicate_image",
    "evidence", "duplicate_condition", "numeric_amount", "zero_window",
    "boolean_window", "annex_role", "section_missing_page", "hidden_unrendered",
    "false_visual_claim", "historical_use",
])
def test_case40_registrar_rejects_unsafe_structure(mutation):
    review, sources, first = inputs()
    d = review["documents"][0]
    c = d["bidder_experience_conditions"][0]
    if mutation == "agreement":
        d["decision"] = "source_conditions_agree_with_warnings_pending_reference_lock"
    elif mutation == "missing_page":
        d["pages_read"].remove(94)
    elif mutation == "fake_ocr":
        d["native_pages_read"].remove(94)
        d["ocr_pages_read"].append(94)
    elif mutation == "missing_image":
        d["complete_rendered_pages_read"].remove(54)
    elif mutation == "personnel":
        c["subject"] = "personnel"
    elif mutation == "invented_quote":
        d["evidence"].append({"id": "fake", "page": 54, "quote": "SIN EXPERIENCIA EXIGIDA"})
    elif mutation == "eligibility":
        c["company_applicability"] = "eligible"
    elif mutation == "lock":
        review["reference_locked"] = True
    elif mutation == "pdf":
        d["pdf_sha256"] = "0" * 64
    elif mutation == "currency":
        c["currency"] = "S/"
    elif mutation == "duplicate_image":
        d["rendered_pages"][1]["path"] = d["rendered_pages"][0]["path"]
    elif mutation == "evidence":
        c["evidence_ids"].append("missing")
    elif mutation == "duplicate_condition":
        d["bidder_experience_conditions"].append(copy.deepcopy(c))
    elif mutation == "numeric_amount":
        c["billed_amount"] = 600000
    elif mutation in ("zero_window", "boolean_window"):
        c["lookback_years"] = 0 if mutation == "zero_window" else True
    elif mutation == "annex_role":
        c["section_role"] = "annex"
    elif mutation == "section_missing_page":
        d["reading_sections"][-1]["pages"].remove(94)
    elif mutation == "hidden_unrendered":
        d["native_pages_not_rendered"].pop()
    elif mutation == "false_visual_claim":
        d["all_pages_visually_reviewed"] = True
    elif mutation == "historical_use":
        d["ranking_eligible"] = True
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_case40_checkpoint_append_preserves_history_and_metric_locks(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 25
    assert result["condition_source_agreements_pending_reference"] == 4
    assert not result["f1_estimable"]
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    previous = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == previous and current[-1] == json.loads(REVIEW.read_text())
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    assert [r["sample_number"] for r in rows if r["first_pass_status"] == "full_supplied_pdf" and r["second_pass_status"] == "not_performed"] == [1, 11]
    case = next(r for r in rows if r["sample_number"] == 40)
    assert case["reference_adjudication_pending"] and case["second_pass_status"] == "full_supplied_pdf"
    assert all(not r["historical_ranking_eligible"] and not r["reference_locked"] for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
    evidence = [e for e in json.loads((output / "second_pass_evidence.json").read_text()) if e["opportunity_id"] == case["opportunity_id"]]
    assert len(evidence) == 23
    assert all(not e["human_or_expert"] and not e["independent_reviewer"] for e in evidence)
    manifest = json.loads((output / "manifest.json").read_text())
    for rendered in json.loads(REVIEW.read_text())["documents"][0]["rendered_pages"]:
        path = (ROOT / rendered["path"]).resolve()
        assert manifest["input_sha256"][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
