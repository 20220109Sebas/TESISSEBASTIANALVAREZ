import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v4 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v37"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case41_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case41_complete_source_reading_keeps_mype_amount_unknown():
    review, sources, first = inputs()
    validate(review, sources, first)
    doc = review["documents"][0]
    assert doc["pages_read"] == list(range(1, 85))
    assert doc["native_pages_read"] == list(range(1, 43)) + list(range(44, 85))
    assert doc["ocr_pages_read"] == [43]
    assert len(doc["rendered_pages"]) == len(doc["complete_rendered_pages_read"]) == 27
    assert doc["all_pages_visually_reviewed"] is False
    general, mype = doc["bidder_experience_conditions"]
    assert general["branch"] == "general" and general["billed_amount"] == "263940.00"
    assert mype["branch"] == "mype" and mype["billed_amount"] is None
    assert mype["amount_candidates"] == [
        {"value": "21995.00", "basis": "digits", "page": 48},
        {"value": "21995.50", "basis": "words", "page": 48},
    ]
    assert mype["amount_resolution"] == "unresolved_digits_words_conflict"
    for c, multiplier in [(general, "3"), (mype, "0.25")]:
        assert c["currency"] == "PEN" and c["lookback_years"] == 8
        assert c["window_anchor"] == "conformidad_o_emision_del_comprobante_de_pago"
        assert c["reference_date"] == "presentacion_de_ofertas"
        assert c["relative_requirement"] == {
            "multiplier": multiplier, "basis": "valor_estimado", "page": 35,
            "converted_to_absolute": False,
        }
        assert c["item"] is None and c["company_applicability"] == "not_determined"
        assert c["document_reference_complete"] is False
    assert "todos_los_integrantes_mype" in mype["activation_condition"]
    evidence = {e["id"]: e for e in doc["evidence"]}
    assert "21,995.00" in evidence["mype_qualification_conflict"]["quote"]
    assert "50/100" in evidence["mype_qualification_conflict"]["quote"]
    assert "máximo" in evidence["consortium_limits"]["quote"]
    assert "60%" in evidence["consortium_limits"]["quote"]
    assert "máximo" in evidence["accreditation"]["quote"]
    assert "veinte (20)" in evidence["accreditation"]["quote"]
    assert "cancelación" in evidence["experience_annex"]["quote"]
    assert "veinticinco" in evidence["personnel_requirements"]["quote"]
    assert "cinco por ciento" in evidence["mype_bonus"]["quote"]
    assert all(v["replaces_archived_text"] is False for v in doc["visual_observations"])


def test_case41_comparison_preserves_legacy_id_rows_without_semantic_pairing():
    review, _, first = inputs()
    doc = review["documents"][0]
    prior = first[doc["opportunity_id"]]["bidder_experience_conditions"]
    current = doc["bidder_experience_conditions"]
    original = copy.deepcopy((prior, current))
    comparisons = compare_conditions(prior, current)
    assert (prior, current) == original
    assert len(comparisons) == 4  # Storage entries, not four semantic requirements.
    legacy = [c for c in comparisons if c["identity_basis"] == "unmapped_legacy_condition_id"]
    assert {c["legacy_condition_id"] for c in legacy} == {"41-general", "41-mype"}
    assert all(c["legacy_identifier_field"] == "id" for c in legacy)
    assert [c["first_pass_raw_condition"] for c in legacy] == prior
    assert all(not (c["first_pass_present"] and c["second_pass_present"]) for c in comparisons)
    assert all(v["status"] == "unknown" for c in comparisons for v in c["fields"].values())
    assert doc["decision"] == "source_conditions_discrepant_pending_reference_lock"


@pytest.mark.parametrize("mutation", [
    "false_agreement", "missing_legacy_disclosure", "duplicated_legacy_id",
    "missing_legacy_id", "conflicting_legacy_ids", "canonical_duplicate",
    "personnel_subject", "invented_quote", "missing_last_page", "missing_ocr_image",
    "fake_native", "eligibility_claim", "reference_lock", "wrong_pdf",
    "bad_currency", "duplicate_render", "unknown_evidence",
])
def test_case41_registration_rejects_unsafe_changes(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    c = doc["bidder_experience_conditions"][0]
    prior = first[doc["opportunity_id"]]["bidder_experience_conditions"]
    if mutation == "false_agreement": doc["decision"] = "source_conditions_agree_with_warnings_pending_reference_lock"
    elif mutation == "missing_legacy_disclosure": review.pop("legacy_identity_comparison")
    elif mutation == "duplicated_legacy_id": prior[1]["id"] = prior[0]["id"]
    elif mutation == "missing_legacy_id": prior[0].pop("id")
    elif mutation == "conflicting_legacy_ids": prior[0]["condition_id"] = "conflicting"
    elif mutation == "canonical_duplicate": doc["bidder_experience_conditions"].append(copy.deepcopy(c))
    elif mutation == "personnel_subject": c["subject"] = "personnel"
    elif mutation == "invented_quote": doc["evidence"].append({"id": "fake", "page": 84, "quote": "SIN EXPERIENCIA EXIGIDA"})
    elif mutation == "missing_last_page": doc["pages_read"].remove(84)
    elif mutation == "missing_ocr_image": doc["complete_rendered_pages_read"].remove(43)
    elif mutation == "fake_native":
        doc["ocr_pages_read"].remove(43)
        doc["native_pages_read"].append(43)
    elif mutation == "eligibility_claim": c["company_applicability"] = "eligible"
    elif mutation == "reference_lock": review["reference_locked"] = True
    elif mutation == "wrong_pdf": doc["pdf_sha256"] = "0" * 64
    elif mutation == "bad_currency": c["currency"] = "S/"
    elif mutation == "duplicate_render": doc["rendered_pages"][1]["path"] = doc["rendered_pages"][0]["path"]
    elif mutation == "unknown_evidence": c["evidence_ids"].append("not_recorded")
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_legacy_condition_id_schema_still_supported_without_rewriting():
    review, _, first = inputs()
    key = review["documents"][0]["opportunity_id"]
    prior = first[key]["bidder_experience_conditions"]
    for row in prior:
        row["condition_id"] = row.pop("id")
    original = copy.deepcopy(prior)
    comparisons = compare_conditions(prior, review["documents"][0]["bidder_experience_conditions"])
    assert prior == original
    assert len(comparisons) == 4
    assert all(c["legacy_identifier_field"] == "condition_id" for c in comparisons if c["first_pass_present"])


def test_case41_build_appends_eighteenth_reading_preserving_reference_and_rankings(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 18
    assert result["condition_source_agreements_pending_reference"] == 4
    assert result["f1_estimable"] is False
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    prior = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == prior and current[-1] == json.loads(REVIEW.read_text())
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    pending = [r["sample_number"] for r in rows if r["first_pass_status"] == "full_supplied_pdf" and r["second_pass_status"] == "not_performed"]
    assert pending == [1, 2, 11, 18, 21, 34, 40, 48, 50]
    case = next(r for r in rows if r["sample_number"] == 41)
    assert case["second_pass_status"] == "full_supplied_pdf" and case["reference_adjudication_pending"] is True
    assert len(case["second_pass_field_comparison"]) == 4
    assert all(r["historical_ranking_eligible"] is False and r["reference_locked"] is False for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
    evidence = json.loads((output / "second_pass_evidence.json").read_text())
    mype, = [e for e in evidence if e["opportunity_id"] == case["opportunity_id"] and e["evidence_id"] == "mype_qualification_conflict"]
    assert mype["page"] == 48 and mype["image_sha256"]
    assert mype["method"] == "native_pdf_text_and_full_page_visual_review"
    assert mype["human_or_expert"] is False
