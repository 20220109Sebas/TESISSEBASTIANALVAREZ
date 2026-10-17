import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v5 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v45"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case1_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case1_actual_reading_and_prior_consultation_disclosed():
    review, sources, first = inputs()
    validate(review, sources, first)
    d = review["documents"][0]
    assert d["pages_read"] == list(range(1, 96))
    assert d["ocr_pages_read"] == list(range(26, 42))
    assert len(d["native_pages_read"]) == 79
    assert len(d["complete_rendered_pages_read"]) == len(d["rendered_pages"]) == 39
    assert len(d["native_pages_not_rendered"]) == 56
    assert len(d["evidence"]) == 15
    assert not d["all_pages_visually_reviewed"] and not d["all_content_proven_recovered"]
    assert not d["new_ocr_executed"] and not d["entire_procurement_file_reviewed"]
    assert "antes de esta lectura" in review["independence_note"]
    assert review["first_pass_notes_consulted"] and not review["independent_reviewer"]
    assert all(not o["replaces_archived_text"] for o in d["visual_observations"])


def test_case1_relative_condition_is_not_absolute_or_generic_service():
    review, _, _ = inputs()
    d = review["documents"][0]
    assert len(d["bidder_experience_conditions"]) == 1
    c = d["bidder_experience_conditions"][0]
    assert c["billed_amount"] is None and c["currency"] == "PEN"
    assert c["relative_threshold"]["multiplier"] == "1"
    assert c["relative_threshold"]["declared_basis_amount"] == "170755.01"
    assert c["lookback_years"] == 20 and d["document_level_lookback_years"] is None
    assert c["window_anchor"] == "conformidad_o_emision_del_comprobante_de_pago_final"
    assert c["reference_date"] == "presentacion_de_ofertas"
    assert "público" in c["similar_services"] and "Supervisiones" in c["similar_services"]
    assert c["maximum_contracts_for_accreditation"] == 20 and c["minimum_contracts"] is None
    assert d["domain"] == "consultoria_de_obra_supervision_edificacion_educativa"
    assert not d["formal_services_evaluation_eligible"] and d["final_scope_reconciliation_pending"]
    assert c["company_applicability"] == "not_determined" and not c["document_reference_complete"]
    assert not d["mype_observation"]["reduced_experience_threshold_identified"]


def test_case1_personnel_consortium_and_annex_roles_stay_distinct():
    review, _, _ = inputs()
    d = review["documents"][0]
    qualification, factor = d["personnel_experience_observations"]
    assert qualification["minimum_months"] == 24 and qualification["lookback_years"] == 25
    assert qualification["starts_from"] == "colegiatura" and qualification["overlap_counted_once"]
    assert factor["additional_years"] == 1 and factor["maximum_points"] == 40
    assert all(not p["applied_to_bidder"] for p in (qualification, factor))
    assert d["consortium_observations"]["maximum_members"] == 2
    assert d["consortium_observations"]["highest_experience_member_minimum_percent"] == 50
    a = d["accreditation_observations"]
    assert a["qualification_experience_annex_number"] == 12 and a["actual_experience_annex_number"] == 11
    assert a["reorganization_annex_number"] == 13 and a["actual_reorganization_annex_number"] == 12
    assert not a["numbering_adjudicated"] and a["private_requires_payment_evidence"]
    assert not d["date_roles"]["collapsed"] and not d["date_roles"]["precedence_adjudicated"]
    conditional, annex = d["other_bidder_experience_observations"]
    assert conditional["lookback_years"] == 2
    assert [a["minimum_contracts"] for a in conditional["alternatives"]] == [1, 4]
    assert not conditional["applied_to_general_specialty"]
    assert annex["blank_template"] and not annex["actual_company_history_proven"]


def test_case1_structural_unknowns_do_not_manufacture_agreement():
    review, _, first = inputs()
    d = review["documents"][0]
    prior = first[d["opportunity_id"]]["bidder_experience_conditions"]
    before = copy.deepcopy(prior)
    comparison = compare_conditions(prior, d["bidder_experience_conditions"])[0]
    assert prior == before
    assert comparison["fields"]["billed_amount"]["status"] == "unknown"
    assert comparison["fields"]["similar_services"]["status"] == "unknown"
    assert all(comparison["fields"][f]["status"] == "match" for f in ("currency", "lookback_years", "reference_date", "window_anchor"))
    assert prior[0]["relative_threshold"] == d["bidder_experience_conditions"][0]["relative_threshold"]
    assert not d["first_pass_structural_gaps"]["relative_threshold_compared_by_registrar"]
    assert d["decision"] == "source_conditions_discrepant_pending_reference_lock"


@pytest.mark.parametrize("mutation", [
    "agreement", "missing_page", "fake_ocr", "missing_image", "personnel",
    "invented_quote", "eligibility", "lock", "pdf", "duplicate_image",
    "section_missing_page", "false_visual_claim", "historical_use", "independence",
])
def test_case1_registrar_rejects_unsafe_claims(mutation):
    review, sources, first = inputs()
    d = review["documents"][0]
    c = d["bidder_experience_conditions"][0]
    if mutation == "agreement":
        d["decision"] = "source_conditions_agree_with_warnings_pending_reference_lock"
    elif mutation == "missing_page":
        d["pages_read"].remove(95)
    elif mutation == "fake_ocr":
        d["native_pages_read"].remove(95)
        d["ocr_pages_read"].append(95)
    elif mutation == "missing_image":
        d["complete_rendered_pages_read"].remove(38)
    elif mutation == "personnel":
        c["subject"] = "personnel"
    elif mutation == "invented_quote":
        d["evidence"][0]["quote"] = "SIN EXPERIENCIA EXIGIDA"
    elif mutation == "eligibility":
        c["company_applicability"] = "eligible"
    elif mutation == "lock":
        review["reference_locked"] = True
    elif mutation == "pdf":
        d["pdf_sha256"] = "0" * 64
    elif mutation == "duplicate_image":
        d["rendered_pages"][1]["path"] = d["rendered_pages"][0]["path"]
    elif mutation == "section_missing_page":
        d["reading_sections"][-1]["pages"].remove(95)
    elif mutation == "false_visual_claim":
        d["all_pages_visually_reviewed"] = True
    elif mutation == "historical_use":
        d["ranking_eligible"] = True
    elif mutation == "independence":
        review["independent_reviewer"] = True
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_case1_append_preserves_previous_reviews_and_metric_gates(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 26
    assert result["condition_source_agreements_pending_reference"] == 4
    assert not result["f1_estimable"]
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    previous = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == previous and current[-1] == json.loads(REVIEW.read_text())
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    assert [r["sample_number"] for r in rows if r["first_pass_status"] == "full_supplied_pdf" and r["second_pass_status"] == "not_performed"] == [11]
    assert all(not r["historical_ranking_eligible"] and not r["reference_locked"] for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    evidence = [e for e in json.loads((output / "second_pass_evidence.json").read_text()) if e["opportunity_id"] == current[-1]["documents"][0]["opportunity_id"]]
    assert len(evidence) == 15
    assert all(not e["human_or_expert"] and not e["independent_reviewer"] for e in evidence)
    manifest = json.loads((output / "manifest.json").read_text())
    for rendered in current[-1]["documents"][0]["rendered_pages"]:
        path = (ROOT / rendered["path"]).resolve()
        assert manifest["input_sha256"][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
