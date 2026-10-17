import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v2 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v38"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case2_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case2_full_ocr_and_visual_reading_preserves_conflicting_windows():
    review, sources, first = inputs()
    validate(review, sources, first)
    doc = review["documents"][0]
    pages = list(range(1, 86))
    assert doc["pages_read"] == doc["ocr_pages_read"] == doc["complete_rendered_pages_read"] == pages
    assert doc["native_pages_read"] == [] and len(doc["rendered_pages"]) == 85
    assert doc["all_pages_visually_reviewed"] is True
    assert doc["all_content_proven_recovered"] is False
    assert doc["document_level_lookback_years"] is None
    assert doc["conflicting_source_periods"] == [8, 15]
    assert [(x["value"], x["page"]) for x in doc["lookback_candidates"]] == [(8, 38), (15, 46), (15, 77)]
    general, mype = doc["bidder_experience_conditions"]
    assert general["branch"] == "general" and general["billed_amount"] == "416600.00"
    assert mype["branch"] == "declared_micro_small_enterprise" and mype["billed_amount"] == "52000.00"
    for c in (general, mype):
        assert c["lookback_years"] == 15  # Clause-specific, not a resolved global window.
        assert c["lookback_basis"] == "qualification_clause_only_not_document_level_resolution"
        assert c["document_level_period_conflict"] is True
        assert c["currency"] == "PEN" and c["item"] is None
        assert c["window_anchor"] == "conformidad_o_emision_del_comprobante"
        assert c["reference_date"] == "presentacion_de_ofertas"
        assert c["document_reference_complete"] is False
        assert c["company_applicability"] == "not_determined"
    assert mype["all_consortium_members_must_be_micro_small"] is True
    assert "todos_los_integrantes_mype" in mype["activation_condition"]
    relative, = doc["additional_formulations"]
    assert relative["relative_factor"] == 2 and relative["relative_base"] == "estimated_value"
    assert relative["billed_amount"] is None and relative["currency"] is None
    assert relative["lookback_years"] == 8 and relative["page"] == 38
    assert relative["converted_to_absolute"] is False
    assert relative["independent_additional_positive"] is False


def test_case2_visual_observations_do_not_replace_archived_ocr():
    review, sources, _ = inputs()
    doc = review["documents"][0]
    source = sources[doc["opportunity_id"]]
    observations = {v["page"]: v for v in doc["visual_observations"]}
    assert all(v["replaces_archived_text"] is False for v in observations.values())
    assert "ocho (8)" in observations[38]["visual_transcription"]
    assert "ocho (8)" not in source["pages"][37]["text"]
    assert observations[49]["structured_observation"] == {
        "maximum_members": 2, "minimum_each_percent": 30, "minimum_most_experienced_percent": 70,
    }
    assert "70%" not in source["pages"][48]["text"]
    evidence = {e["id"]: e for e in doc["evidence"]}
    assert "DIECIOCHO (18) MESES" in evidence["supervisor_qualification"]["quote"]
    assert evidence["supervisor_qualification"]["page"] == 47
    assert "veinticinco (25)" in evidence["personnel_window"]["quote"]
    assert "máximo de veinte (20)" in evidence["accreditation"]["quote"]
    assert "cancelación" in evidence["annex_dates"]["quote"]
    assert "no se da a conocer" in evidence["undisclosed_contract_value"]["quote"]
    assert "[CONSIGNAR MONEDA Y MONTO]" == evidence["empty_contract_amount"]["quote"]
    excluded = {e["evidence_id"] for e in doc["excluded_as_bidder_experience"]}
    assert {"supervisor_qualification", "personnel_window", "guarantee_not_experience", "tax_percentage", "mype_bonus", "redam_not_experience"} <= excluded
    accreditation = doc["accreditation_observations"]
    assert accreditation["maximum_contracts"] == 20 and accreditation["minimum_contracts"] is None
    assert accreditation["private_contracts_require_payment_evidence"] is True
    assert accreditation["self_placed_paid_stamp_sufficient"] is False


def test_case2_partial_field_matches_are_not_complete_agreement():
    review, _, first = inputs()
    doc = review["documents"][0]
    prior = first[doc["opportunity_id"]]["bidder_experience_conditions"]
    before = copy.deepcopy(prior)
    comparisons = compare_conditions(prior, doc["bidder_experience_conditions"])
    assert prior == before and len(comparisons) == 2
    for comparison in comparisons:
        assert comparison["first_pass_present"] and comparison["second_pass_present"]
        fields = comparison["fields"]
        assert all(fields[k]["status"] == "match" for k in (
            "billed_amount", "currency", "lookback_years", "window_anchor", "reference_date",
        ))
        assert fields["similar_services"]["status"] == "unknown"
        assert fields["activation_condition"]["status"] == "unknown"
    assert doc["decision"] == "source_conditions_discrepant_pending_reference_lock"


@pytest.mark.parametrize("mutation", [
    "false_agreement", "missing_last_page", "missing_ocr_image", "fake_native",
    "personnel_subject", "invented_quote", "eligibility_claim", "reference_lock",
    "wrong_pdf", "bad_currency", "duplicate_render", "unknown_evidence", "duplicate_condition",
])
def test_case2_registration_rejects_unsafe_changes(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    c = doc["bidder_experience_conditions"][0]
    if mutation == "false_agreement": doc["decision"] = "source_conditions_agree_with_warnings_pending_reference_lock"
    elif mutation == "missing_last_page": doc["pages_read"].remove(85)
    elif mutation == "missing_ocr_image": doc["complete_rendered_pages_read"].remove(38)
    elif mutation == "fake_native":
        doc["ocr_pages_read"].remove(38)
        doc["native_pages_read"].append(38)
    elif mutation == "personnel_subject": c["subject"] = "personnel"
    elif mutation == "invented_quote": doc["evidence"].append({"id": "fake", "page": 85, "quote": "SIN EXPERIENCIA EXIGIDA"})
    elif mutation == "eligibility_claim": c["company_applicability"] = "eligible"
    elif mutation == "reference_lock": review["reference_locked"] = True
    elif mutation == "wrong_pdf": doc["pdf_sha256"] = "0" * 64
    elif mutation == "bad_currency": c["currency"] = "S/"
    elif mutation == "duplicate_render": doc["rendered_pages"][1]["path"] = doc["rendered_pages"][0]["path"]
    elif mutation == "unknown_evidence": c["evidence_ids"].append("missing")
    elif mutation == "duplicate_condition": doc["bidder_experience_conditions"].append(copy.deepcopy(c))
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_case2_build_appends_nineteenth_reading_without_unlocking_evaluation(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 19
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
    assert pending == [1, 11, 18, 21, 34, 40, 48, 50]
    case = next(r for r in rows if r["sample_number"] == 2)
    assert case["second_pass_status"] == "full_supplied_pdf" and case["reference_adjudication_pending"] is True
    assert len(case["second_pass_field_comparison"]) == 2
    assert all(r["historical_ranking_eligible"] is False and r["reference_locked"] is False for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
    evidence = [e for e in json.loads((output / "second_pass_evidence.json").read_text()) if e["opportunity_id"] == case["opportunity_id"]]
    assert len(evidence) == 24
    assert all(e["image_sha256"] and e["human_or_expert"] is False and e["independent_reviewer"] is False for e in evidence)
    assert all(e["method"].endswith("_and_full_page_visual_review") for e in evidence)
