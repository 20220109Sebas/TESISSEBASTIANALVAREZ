import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v3 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v34"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case13_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case13_complete_reading_preserves_mype_scope_uncertainty():
    review, sources, first = inputs()
    validate(review, sources, first)
    doc = review["documents"][0]
    assert doc["pages_read"] == doc["complete_rendered_pages_read"] == list(range(1, 83))
    assert doc["ocr_pages_read"] == list(range(23, 35))
    assert doc["native_pages_read"] == list(range(1, 23)) + list(range(35, 83))
    assert len(doc["rendered_pages"]) == 82
    general, mype = doc["bidder_experience_conditions"]
    assert general["billed_amount"] == "150000.00"
    assert general["lookback_years"] == 15
    assert general["window_anchor"] == "conformidad_o_emision_del_comprobante"
    assert general["reference_date"] == "presentacion_de_ofertas"
    assert mype["billed_amount"] == "37500.00"
    assert all(mype[k] is None for k in ("lookback_years", "window_anchor", "reference_date"))
    assert mype["activation_condition"] == "declarar_condicion_micro_o_pequena_empresa_en_anexo_1"
    assert mype["contextual_window_candidate"]["years"] == 15
    assert mype["contextual_window_candidate"]["status"] == "context_only_not_explicit_MYPE_field"
    for condition in (general, mype):
        assert condition["item"] is None
        assert condition["company_applicability"] == "not_determined"
        assert condition["document_reference_complete"] is False
    evidence = {e["id"]: e for e in doc["evidence"]}
    assert evidence["general_amount_window"]["page"] == 30
    assert evidence["periodic_services"]["page"] == 31
    assert evidence["personnel_window"]["page"] == 32
    assert evidence["annex_window_context"]["page"] == 72
    assert "veinticinco (25)" in evidence["personnel_window"]["quote"]
    assert "cancelación" in evidence["annex_date"]["quote"]
    assert review["independent_reviewer"] is False


def test_case13_legacy_branches_remain_unmapped_and_raw_inventory_is_preserved():
    review, _, first = inputs()
    doc = review["documents"][0]
    prior = first[doc["opportunity_id"]]["bidder_experience_conditions"]
    current = doc["bidder_experience_conditions"]
    saved = copy.deepcopy((prior, current))
    comparison = compare_conditions(prior, current)
    assert (prior, current) == saved
    assert len(comparison) == 4
    old = [c for c in comparison if c["identity_basis"] == "unmapped_legacy_condition_id"]
    assert {c["legacy_condition_id"] for c in old} == {"13-general", "13-mype"}
    assert [c["first_pass_raw_condition"] for c in sorted(old, key=lambda c: c["legacy_condition_id"])] == prior
    assert all(c["first_pass_present"] != c["second_pass_present"] for c in comparison)
    assert all(v["status"] == "unknown" for c in comparison for v in c["fields"].values())
    assert doc["decision"] == "source_conditions_discrepant_pending_reference_lock"
    assert prior[0]["amount"] == 150000
    assert prior[1]["amount"] == 37500
    assert prior[1]["lookback_years"] is None
    warnings = {w["code"] for w in doc["source_warnings"]}
    assert {"legacy_identity_collision", "mype_window_scope_pending", "package_not_two_experience_items",
            "date_anchor_difference", "reference_adjudication_pending"} <= warnings


@pytest.mark.parametrize("mutation", [
    "amount_format", "false_agreement", "personnel_subject", "invented_quote",
    "duplicated_condition", "missing_last_page", "missing_scan_image", "fake_ocr", "eligibility_claim",
    "undisclosed_legacy_comparison",
])
def test_unsafe_case13_registration_changes_are_rejected(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    condition = doc["bidder_experience_conditions"][0]
    if mutation == "amount_format": condition["billed_amount"] = "150,000"
    elif mutation == "false_agreement": doc["decision"] = "source_conditions_agree_with_warnings_pending_reference_lock"
    elif mutation == "personnel_subject": condition["subject"] = "personnel"
    elif mutation == "invented_quote": doc["evidence"].append({"id": "invented", "page": 82, "quote": "NO EXISTE EXPERIENCIA"})
    elif mutation == "duplicated_condition": doc["bidder_experience_conditions"].append(copy.deepcopy(condition))
    elif mutation == "missing_last_page": doc["pages_read"].remove(82)
    elif mutation == "missing_scan_image": doc["complete_rendered_pages_read"].remove(30)
    elif mutation == "fake_ocr":
        doc["native_pages_read"].remove(72)
        doc["ocr_pages_read"].append(72)
    elif mutation == "eligibility_claim": condition["company_applicability"] = "eligible"
    elif mutation == "undisclosed_legacy_comparison": review.pop("legacy_identity_comparison")
    with pytest.raises(ValueError):
        validate(review, sources, first)


@pytest.mark.parametrize("mutation", ["duplicate_id", "missing_id", "canonical_collision", "unknown_schema"])
def test_legacy_storage_identity_does_not_repair_invalid_duplicates(mutation):
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


def test_v3_keeps_strict_comparison_for_canonical_inputs():
    condition = {"subject": "bidder", "section_role": "qualification", "item": None,
                 "branch": "general", "billed_amount": "150000.00", "currency": "PEN",
                 "lookback_years": 15, "window_anchor": "a", "reference_date": "b", "similar_services": "c"}
    saved = copy.deepcopy(condition)
    comparison = compare_conditions([condition], [condition])
    assert condition == saved
    assert len(comparison) == 1
    assert all(v["status"] == "match" for k, v in comparison[0]["fields"].items() if k != "activation_condition")
    assert comparison[0]["fields"]["activation_condition"]["status"] == "unspecified_in_both"
    assert comparison[0]["identity_basis"] == "canonical_fields_as_stored"


def test_build_appends_fifteenth_rereading_preserving_prior_inventories(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 15
    assert result["condition_source_agreements_pending_reference"] == 4
    assert result["f1_estimable"] is False
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    prior = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == prior
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    assert sum(r["first_pass_status"] == "full_supplied_pdf" and r["second_pass_status"] == "not_performed" for r in rows) == 12
    case = next(r for r in rows if r["sample_number"] == 13)
    assert case["second_pass_status"] == "full_supplied_pdf"
    assert case["reference_adjudication_pending"] is True
    assert len(case["second_pass_field_comparison"]) == 4
    assert all(r["historical_ranking_eligible"] is False and r["reference_locked"] is False for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
