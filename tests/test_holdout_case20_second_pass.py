import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v3 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v35"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case20_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case20_keeps_four_item_branches_and_unknown_amount():
    review, sources, first = inputs()
    validate(review, sources, first)
    doc = review["documents"][0]
    assert doc["pages_read"] == doc["native_pages_read"] == list(range(1, 83))
    assert doc["ocr_pages_read"] == []
    assert len(doc["rendered_pages"]) == len(doc["complete_rendered_pages_read"]) == 28
    assert doc["all_pages_visually_reviewed"] is False
    assert doc["entire_procurement_file_reviewed"] is False
    conditions = doc["bidder_experience_conditions"]
    assert [(c["item"], c["branch"], c["billed_amount"]) for c in conditions] == [
        (1, "general", None), (1, "mype", "14000.00"),
        (2, "general", "300000.00"), (2, "mype", "39000.00")]
    assert conditions[0]["numeric_candidate"]["status"] == "visual_candidate_only_not_reference_value"
    assert conditions[0]["amount_digits_raw"] == "S/ 1 00 000"
    assert conditions[0]["amount_digits_visual"] == "S/ 100 000"
    assert conditions[0]["amount_words_raw"] == "ciento \ncien mil y 00/100 soles"
    for condition in conditions:
        assert condition["lookback_years"] == 15
        assert condition["window_anchor"] == "conformidad_o_emision_del_comprobante"
        assert condition["reference_date"] == "presentacion_de_ofertas"
        assert condition["company_applicability"] == "not_determined"
        assert condition["document_reference_complete"] is False
        if condition["branch"] == "mype":
            assert condition["lookback_basis"] == "explicit_in_MYPE_clause"
            assert "todos_los_integrantes_del_consorcio_MYPE" in condition["activation_condition"]
    evidence = {e["id"]: e for e in doc["evidence"]}
    assert evidence["item1_general"]["page"] == 44
    assert evidence["item2_general"]["page"] == 45
    assert evidence["offer_price_rule"]["page"] == 70
    assert "precio de la oferta" in evidence["offer_price_rule"]["quote"]
    assert "cancelación" in evidence["annex_date_context"]["quote"]
    assert "veinticinco" in evidence["personnel_scoring"]["quote"]
    assert "DETECCIÓN DE INCENDIO" in evidence["confidentiality_entity"]["quote"]
    warnings = {w["code"] for w in doc["source_warnings"]}
    assert {"item1_amount_literal_anomaly", "legacy_identity_collision",
            "offer_price_rule_not_experience_resolution", "date_anchor_difference",
            "historical_bytes_unproven", "reference_adjudication_pending"} <= warnings


def test_case20_raw_legacy_rows_remain_unmapped_not_eight_requirements():
    review, _, first = inputs()
    doc = review["documents"][0]
    prior = first[doc["opportunity_id"]]["bidder_experience_conditions"]
    current = doc["bidder_experience_conditions"]
    before = copy.deepcopy((prior, current))
    comparison = compare_conditions(prior, current)
    assert (prior, current) == before
    assert len(prior) == len(current) == 4
    assert len(comparison) == 8
    old = [c for c in comparison if c["identity_basis"] == "unmapped_legacy_condition_id"]
    assert {c["legacy_condition_id"] for c in old} == {
        "20-item1-general", "20-item1-mype", "20-item2-general", "20-item2-mype"}
    assert [c["first_pass_raw_condition"] for c in sorted(old, key=lambda c: c["legacy_condition_id"])] == prior
    assert all(c["first_pass_present"] != c["second_pass_present"] for c in comparison)
    assert all(v["status"] == "unknown" for c in comparison for v in c["fields"].values())
    assert prior[0]["amount"] is None and current[0]["billed_amount"] is None
    assert doc["decision"] == "source_conditions_discrepant_pending_reference_lock"
    assert review["independent_reviewer"] is False


@pytest.mark.parametrize("mutation", [
    "amount_format", "false_agreement", "personnel_subject", "invented_quote",
    "duplicated_condition", "missing_last_page", "missing_evidence_image", "fake_ocr",
    "eligibility_claim", "undisclosed_legacy_comparison", "reference_lock", "wrong_pdf",
])
def test_case20_registration_rejects_unsafe_changes(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    condition = doc["bidder_experience_conditions"][0]
    if mutation == "amount_format": condition["billed_amount"] = "100,000"
    elif mutation == "false_agreement": doc["decision"] = "source_conditions_agree_with_warnings_pending_reference_lock"
    elif mutation == "personnel_subject": condition["subject"] = "personnel"
    elif mutation == "invented_quote": doc["evidence"].append({"id": "invented", "page": 82, "quote": "SIN EXPERIENCIA EXIGIDA"})
    elif mutation == "duplicated_condition": doc["bidder_experience_conditions"].append(copy.deepcopy(condition))
    elif mutation == "missing_last_page": doc["pages_read"].remove(82)
    elif mutation == "missing_evidence_image": doc["complete_rendered_pages_read"].remove(44)
    elif mutation == "fake_ocr":
        doc["native_pages_read"].remove(73)
        doc["ocr_pages_read"].append(73)
    elif mutation == "eligibility_claim": condition["company_applicability"] = "eligible"
    elif mutation == "undisclosed_legacy_comparison": review.pop("legacy_identity_comparison")
    elif mutation == "reference_lock": review["reference_locked"] = True
    elif mutation == "wrong_pdf": doc["pdf_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate(review, sources, first)


@pytest.mark.parametrize("mutation", ["duplicate_id", "missing_id", "canonical_collision", "unknown_schema"])
def test_case20_legacy_storage_does_not_repair_invalid_identities(mutation):
    review, sources, first = inputs()
    prior = first[review["documents"][0]["opportunity_id"]]["bidder_experience_conditions"]
    if mutation == "duplicate_id": prior[1]["condition_id"] = prior[0]["condition_id"]
    elif mutation == "missing_id": prior[1].pop("condition_id")
    elif mutation == "canonical_collision":
        for c in prior:
            c.update(subject="bidder", section_role="qualification", branch="general")
    elif mutation == "unknown_schema": prior[1]["requirement_role"] = "evaluation"
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_case20_build_preserves_fifty_cases_and_prior_artifacts(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 16
    assert result["condition_source_agreements_pending_reference"] == 4
    assert result["f1_estimable"] is False
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    prior = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == prior
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    assert sum(r["first_pass_status"] == "full_supplied_pdf" and r["second_pass_status"] == "not_performed" for r in rows) == 11
    case = next(r for r in rows if r["sample_number"] == 20)
    assert case["second_pass_status"] == "full_supplied_pdf"
    assert case["reference_adjudication_pending"] is True
    assert len(case["second_pass_field_comparison"]) == 8
    assert all(r["historical_ranking_eligible"] is False and r["reference_locked"] is False for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
