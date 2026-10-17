import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v2 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v39"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case50_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case50_full_reading_preserves_relative_amount_and_scoped_window():
    review, sources, first = inputs()
    validate(review, sources, first)
    doc = review["documents"][0]
    pages = list(range(1, 86))
    assert doc["pages_read"] == doc["ocr_pages_read"] == doc["complete_rendered_pages_read"] == pages
    assert doc["native_pages_read"] == [] and len(doc["rendered_pages"]) == 85
    assert doc["all_pages_visually_reviewed"] is True
    assert doc["all_content_proven_recovered"] is False
    assert all("Motor del renderizado original no restablecido" in r["renderer"] for r in doc["rendered_pages"])
    condition, = doc["bidder_experience_conditions"]
    assert condition["branch"] == "general" and condition["item"] is None
    assert condition["billed_amount"] is None and condition["converted_to_absolute"] is False
    assert condition["amount_expression"] == "Una (01) vez la cuantía"
    assert condition["quantia_multiplier"] == 1 and condition["currency"] == "PEN"
    assert "contexto" in condition["currency_basis"]
    assert condition["lookback_years"] == doc["document_level_lookback_years"] == 20
    assert doc["conflicting_source_periods"] == []
    assert condition["reference_date"] == "presentacion_de_ofertas"
    assert condition["window_anchor"] == "conformidad_o_emision_del_comprobante_de_pago_final"
    assert condition["company_applicability"] == "not_determined"
    assert condition["document_reference_complete"] is False
    assert condition["similar_services"] == [
        "Supervisión de Obras", "Saneamiento y afines", "Infraestructura para agua potable",
        "Infraestructura para alcantarillado", "Obras rurales",
    ]


def test_case50_distinguishes_personnel_forms_and_source_cross_references():
    review, _, _ = inputs()
    doc = review["documents"][0]
    evidence = {e["id"]: e for e in doc["evidence"]}
    assert len(evidence) == 25
    assert "veinte (20)" in evidence["bidder_requirement"]["quote"]
    assert "veinticinco" in evidence["personnel_window"]["quote"]
    assert "quince (15) años para consultoría o veinte (20) años para consultoría de obra" in evidence["annex_window"]["quote"]
    assert "cancelación" in evidence["annex_dates"]["quote"]
    assert "máximo de veinte" in evidence["accreditation"]["quote"]
    assert doc["accreditation_observations"]["maximum_contracts"] == 20
    assert doc["accreditation_observations"]["minimum_contracts"] is None
    assert doc["accreditation_observations"]["private_contracts_require_payment_evidence"] is True
    assert doc["accreditation_observations"]["self_placed_paid_stamp_sufficient"] is False
    assert [(r["referred_number"], r["actual_number"]) for r in doc["annex_cross_references"]] == [(12, 11), (13, 12)]
    assert all(r["resolved"] is False for r in doc["annex_cross_references"])
    assert doc["consortium_observations"] == {
        "maximum_members": 2, "minimum_each_percent": 30,
        "minimum_most_experienced_percent": 50, "blank_third_member_is_not_permission": True,
    }
    visual = {v["page"]: v for v in doc["visual_observations"]}
    assert all(v["replaces_archived_text"] is False for v in visual.values())
    assert "317580.35" in visual[39]["visual_transcription"]
    assert visual[47]["structured_observation"]["personnel_lookback_years"] == 25
    excluded = {e["evidence_id"] for e in doc["excluded_as_bidder_experience"]}
    assert {"supervisor_experience", "personnel_window", "additional_personnel_experience",
            "offer_weights", "kinship_exception", "tax_annex_actual", "mype_bonus"} <= excluded


def test_case50_unknown_amount_does_not_count_as_full_agreement():
    review, _, first = inputs()
    doc = review["documents"][0]
    prior = first[doc["opportunity_id"]]["bidder_experience_conditions"]
    before = copy.deepcopy(prior)
    comparison, = compare_conditions(prior, doc["bidder_experience_conditions"])
    assert prior == before
    assert comparison["first_pass_present"] and comparison["second_pass_present"]
    fields = comparison["fields"]
    assert all(fields[k]["status"] == "match" for k in (
        "currency", "lookback_years", "window_anchor", "reference_date",
    ))
    assert fields["billed_amount"]["status"] == fields["similar_services"]["status"] == "unknown"
    assert fields["activation_condition"]["status"] == "unspecified_in_both"
    assert doc["decision"] == "source_conditions_discrepant_pending_reference_lock"


@pytest.mark.parametrize("mutation", [
    "false_agreement", "missing_last_page", "missing_ocr_image", "fake_native",
    "personnel_subject", "invented_quote", "eligibility_claim", "reference_lock",
    "wrong_pdf", "bad_currency", "duplicate_render", "unknown_evidence", "duplicate_condition",
])
def test_case50_registration_rejects_unsafe_changes(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    condition = doc["bidder_experience_conditions"][0]
    if mutation == "false_agreement":
        doc["decision"] = "source_conditions_agree_with_warnings_pending_reference_lock"
    elif mutation == "missing_last_page":
        doc["pages_read"].remove(85)
    elif mutation == "missing_ocr_image":
        doc["complete_rendered_pages_read"].remove(44)
    elif mutation == "fake_native":
        doc["ocr_pages_read"].remove(44)
        doc["native_pages_read"].append(44)
    elif mutation == "personnel_subject":
        condition["subject"] = "personnel"
    elif mutation == "invented_quote":
        doc["evidence"].append({"id": "fake", "page": 85, "quote": "SIN EXPERIENCIA EXIGIDA"})
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


def test_case50_build_appends_twentieth_reading_without_unlocking_evaluation(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 20
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
    assert pending == [1, 11, 18, 21, 34, 40, 48]
    case = next(r for r in rows if r["sample_number"] == 50)
    assert case["second_pass_status"] == "full_supplied_pdf" and case["reference_adjudication_pending"] is True
    assert len(case["second_pass_field_comparison"]) == 1
    assert all(r["historical_ranking_eligible"] is False and r["reference_locked"] is False for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
    evidence = [e for e in json.loads((output / "second_pass_evidence.json").read_text()) if e["opportunity_id"] == case["opportunity_id"]]
    assert len(evidence) == 25
    assert all(e["image_sha256"] and e["human_or_expert"] is False and e["independent_reviewer"] is False for e in evidence)
    assert all(e["method"].endswith("_and_full_page_visual_review") for e in evidence)
