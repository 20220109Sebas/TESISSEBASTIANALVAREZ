import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v2 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v41"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case18_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case18_full_text_and_actual_image_coverage():
    review, sources, first = inputs()
    validate(review, sources, first)
    d = review["documents"][0]
    assert d["pages_read"] == d["native_pages_read"] == list(range(1, 95))
    assert d["ocr_pages_read"] == d["unreviewed_pages"] == []
    assert len(d["rendered_pages"]) == len(d["complete_rendered_pages_read"]) == 18
    assert {67, 68} <= set(d["complete_rendered_pages_read"])
    assert len(d["native_pages_not_rendered"]) == 76
    assert d["all_content_proven_recovered"] is d["all_pages_visually_reviewed"] is False
    assert all(v["replaces_archived_text"] is False for v in d["visual_observations"])
    assert d["formal_services_evaluation_eligible"] is False
    assert d["domain"] == "consultoria_de_obra_supervision_oxi"
    assert len(d["evidence"]) == 21


def test_case18_amount_abstention_and_roles_are_preserved():
    d = inputs()[0]["documents"][0]
    c = d["bidder_experience_conditions"][0]
    assert len(d["bidder_experience_conditions"]) == 1
    assert c["branch"] == "facturacion" and c["billed_amount"] is None
    assert c["amount_observation"]["resolved"] is False
    assert c["amount_observation"]["words_interpretation"] == "444002.87"
    assert c["amount_observation"]["budget_used_to_resolve"] is False
    assert c["currency"] == "PEN" and c["lookback_years"] == 10
    assert c["window_anchor"] == "no_explicitado"
    assert c["reference_date"] == "presentacion_de_ofertas"
    assert c["company_applicability"] == "not_determined"
    assert c["document_reference_complete"] is False
    a = d["accreditation_observations"]
    assert a["maximum_contracts"] is None and a["source_limits_conflict"] == [10, 20]
    obs = {o["branch"]: o for o in d["other_bidder_experience_observations"]}
    assert obs["proyectos_similares"]["minimum_projects"] == 2
    assert obs["proyectos_similares"]["subject"] == "bidder_or_linked_companies"
    assert obs["proyectos_similares"]["section_role"] == "admission"
    for key in ("actividad_anexo", "especialidad_anexo"):
        assert obs[key]["minimum_years"] == 4 and obs[key]["lookback_years"] is None
        assert obs[key]["section_role"] == "annex"
    assert d["optional_bidder_experience_observations"][0]["maximum_reference_multiplier"] == 3
    assert all(not o["applied_evaluation_factor"] for o in d["optional_bidder_experience_observations"])
    evidence = {e["id"]: e for e in d["evidence"]}
    assert "veinticinco (25)" in evidence["personnel_window"]["quote"]
    assert "3 años" in evidence["key_personnel"]["quote"]
    assert "cuatro (4) años como mínimo" in evidence["activity_four_years"]["quote"]
    assert "garantía de fiel cumplimiento" in evidence["mype_guarantee"]["quote"]
    assert d["document_level_lookback_years"] is None


def test_case18_comparison_does_not_flatten_prior_roles_or_claim_agreement():
    review, _, first = inputs()
    d = review["documents"][0]
    prior = first[d["opportunity_id"]]["bidder_experience_conditions"]
    before = copy.deepcopy(prior)
    comparisons = compare_conditions(prior, d["bidder_experience_conditions"])
    assert prior == before and len(comparisons) == 4
    c = next(c for c in comparisons if c["identity"]["branch"] == "facturacion")
    assert all(c["fields"][f]["status"] == "match" for f in (
        "currency", "lookback_years", "window_anchor", "reference_date"))
    assert c["fields"]["billed_amount"]["status"] == c["fields"]["similar_services"]["status"] == "unknown"
    others = [c for c in comparisons if c["identity"]["branch"] != "facturacion"]
    assert all(c["first_pass_present"] and not c["second_pass_present"] for c in others)
    assert {c["identity"]["section_role"] for c in others} == {"annex", "admission"}
    assert d["decision"] == "source_conditions_discrepant_pending_reference_lock"


@pytest.mark.parametrize("mutation", [
    "agreement", "missing_page", "fake_ocr", "missing_image", "personnel",
    "invented_quote", "eligibility", "lock", "pdf", "currency",
    "duplicate_image", "evidence", "duplicate_condition",
])
def test_case18_rejects_unsafe_registration(mutation):
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
        d["complete_rendered_pages_read"].remove(33)
    elif mutation == "personnel":
        c["subject"] = "personnel"
    elif mutation == "invented_quote":
        d["evidence"].append({"id": "fake", "page": 33, "quote": "NO EXPERIENCIA EXIGIDA"})
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
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_case18_build_preserves_sample_prior_inventories_and_closed_gates(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 22
    assert result["condition_source_agreements_pending_reference"] == 4
    assert result["f1_estimable"] is False
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    previous = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == previous and current[-1] == json.loads(REVIEW.read_text())
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    assert [r["sample_number"] for r in rows if r["first_pass_status"] == "full_supplied_pdf"
            and r["second_pass_status"] == "not_performed"] == [1, 11, 21, 34, 40]
    case = next(r for r in rows if r["sample_number"] == 18)
    assert case["second_pass_status"] == "full_supplied_pdf" and case["reference_adjudication_pending"] is True
    assert len(case["second_pass_field_comparison"]) == 4
    assert all(not r["historical_ranking_eligible"] and not r["reference_locked"] for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
    evidence = [e for e in json.loads((output / "second_pass_evidence.json").read_text()) if e["opportunity_id"] == case["opportunity_id"]]
    assert len(evidence) == 21
    assert all(e["image_sha256"] and not e["human_or_expert"] and not e["independent_reviewer"] for e in evidence)
    assert all(e["method"] == "native_pdf_text_and_full_page_visual_review" for e in evidence)
