import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v2 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v36"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case16_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case16_complete_reading_and_visual_amount_provenance():
    review, sources, first = inputs()
    validate(review, sources, first)
    doc = review["documents"][0]
    assert doc["pages_read"] == list(range(1, 84))
    assert doc["native_pages_read"] == list(range(1, 21)) + list(range(41, 83))
    assert doc["ocr_pages_read"] == list(range(21, 41)) + [83]
    assert len(doc["rendered_pages"]) == len(doc["complete_rendered_pages_read"]) == 35
    assert set(doc["ocr_pages_read"]) <= set(doc["complete_rendered_pages_read"])
    assert doc["all_pages_visually_reviewed"] is False
    assert doc["entire_procurement_file_reviewed"] is False
    condition, = doc["bidder_experience_conditions"]
    assert condition["item"] is None and condition["branch"] == "general"
    assert condition["billed_amount"] == "3000000.00" and condition["currency"] == "PEN"
    assert condition["lookback_years"] == 15
    assert condition["window_anchor"] == "conformidad_o_emision_del_comprobante_de_pago"
    assert condition["reference_date"] == "presentacion_de_ofertas"
    assert condition["company_applicability"] == "not_determined"
    assert condition["document_reference_complete"] is False
    assert condition["amount_basis"] == "visual_page_36_ocr_omits_amount"
    evidence = {e["id"]: e for e in doc["evidence"]}
    visual = {e["id"]: e for e in doc["visual_observations"]}
    amount = visual[condition["visual_observation_id"]]
    assert amount["page"] == evidence[amount["related_evidence_id"]]["page"] == 36
    assert amount["replaces_archived_text"] is False
    assert amount["billed_amount"] == condition["billed_amount"]
    assert amount["transcription"] == "S/ 3'000,000.00 (TRES MILLONES CON 00/100 SOLES)"
    assert amount["transcription"] not in sources[doc["opportunity_id"]]["pages"][35]["text"]
    assert evidence["bidder_clause_ocr"]["quote_basis"] == "archived_ocr_verbatim_not_visual_transcription"
    assert evidence["bidder_clause_continuation"]["page"] == 37
    assert "máximo de veinte" in evidence["bidder_clause_continuation"]["quote"]
    assert "cincuenta (50%)" in evidence["consortium_limits"]["quote"]
    assert "cancelación" in evidence["annex_experience_dates"]["quote"]
    assert "veinticinco" in evidence["personnel_requirements"]["quote"]
    assert "NO APLICA" in evidence["mype_bonus_not_applicable"]["quote"]
    assert all(v["replaces_archived_text"] is False for v in visual.values())
    warnings = {w["code"] for w in doc["source_warnings"]}
    assert {"visual_amount_missing_from_ocr", "representation_comparison_pending",
            "similar_services_cross_page", "no_reduced_mype_branch_identified",
            "historical_bytes_unproven", "reference_adjudication_pending"} <= warnings


def test_case16_comparison_keeps_type_difference_and_missing_field():
    review, _, first = inputs()
    doc = review["documents"][0]
    prior = first[doc["opportunity_id"]]["bidder_experience_conditions"]
    current = doc["bidder_experience_conditions"]
    before = copy.deepcopy((prior, current))
    comparison, = compare_conditions(prior, current)
    assert (prior, current) == before
    assert comparison["first_pass_present"] is True and comparison["second_pass_present"] is True
    fields = comparison["fields"]
    assert fields["billed_amount"] == {"first": 3000000, "second": "3000000.00", "status": "disagreement"}
    assert fields["similar_services"]["status"] == "unknown"
    assert all(fields[k]["status"] == "match" for k in ("currency", "lookback_years", "window_anchor", "reference_date"))
    assert fields["activation_condition"]["status"] == "unspecified_in_both"
    assert doc["decision"] == "source_conditions_discrepant_pending_reference_lock"
    assert review["independent_reviewer"] is False


@pytest.mark.parametrize("mutation", [
    "amount_format", "false_agreement", "personnel_subject", "invented_quote",
    "duplicated_condition", "missing_last_page", "missing_scan_image", "fake_native",
    "eligibility_claim", "reference_lock", "wrong_pdf", "bad_currency",
    "duplicate_render", "unknown_evidence",
])
def test_case16_registration_rejects_unsafe_changes(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    condition = doc["bidder_experience_conditions"][0]
    if mutation == "amount_format": condition["billed_amount"] = "3,000,000"
    elif mutation == "false_agreement": doc["decision"] = "source_conditions_agree_with_warnings_pending_reference_lock"
    elif mutation == "personnel_subject": condition["subject"] = "personnel"
    elif mutation == "invented_quote": doc["evidence"].append({"id": "invented", "page": 83, "quote": "SIN EXPERIENCIA EXIGIDA"})
    elif mutation == "duplicated_condition": doc["bidder_experience_conditions"].append(copy.deepcopy(condition))
    elif mutation == "missing_last_page": doc["pages_read"].remove(83)
    elif mutation == "missing_scan_image": doc["complete_rendered_pages_read"].remove(83)
    elif mutation == "fake_native":
        doc["ocr_pages_read"].remove(36)
        doc["native_pages_read"].append(36)
    elif mutation == "eligibility_claim": condition["company_applicability"] = "eligible"
    elif mutation == "reference_lock": review["reference_locked"] = True
    elif mutation == "wrong_pdf": doc["pdf_sha256"] = "0" * 64
    elif mutation == "bad_currency": condition["currency"] = "S/"
    elif mutation == "duplicate_render": doc["rendered_pages"][1]["path"] = doc["rendered_pages"][0]["path"]
    elif mutation == "unknown_evidence": condition["evidence_ids"].append("not_recorded")
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_case16_build_appends_seventeenth_reading_preserving_reference_and_rankings(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 17
    assert result["condition_source_agreements_pending_reference"] == 4
    assert result["f1_estimable"] is False
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    prior = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == prior
    assert current[-1] == json.loads(REVIEW.read_text())
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    assert sum(r["first_pass_status"] == "full_supplied_pdf" and r["second_pass_status"] == "not_performed" for r in rows) == 10
    case = next(r for r in rows if r["sample_number"] == 16)
    assert case["second_pass_status"] == "full_supplied_pdf" and case["reference_adjudication_pending"] is True
    assert len(case["second_pass_field_comparison"]) == 1
    assert all(r["historical_ranking_eligible"] is False and r["reference_locked"] is False for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
    evidence = json.loads((output / "second_pass_evidence.json").read_text())
    amount, = [e for e in evidence if e["opportunity_id"] == case["opportunity_id"] and e["evidence_id"] == "bidder_clause_ocr"]
    assert amount["page"] == 36 and amount["image_sha256"]
    assert amount["method"] == "apple_vision_local_ocr_and_full_page_visual_review"
    assert amount["human_or_expert"] is False
