import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v5 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v42"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case21_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case21_actual_text_and_image_coverage():
    review, sources, first = inputs()
    validate(review, sources, first)
    d = review["documents"][0]
    assert d["pages_read"] == list(range(1, 95))
    assert d["ocr_pages_read"] == list(range(23, 43))
    assert len(d["native_pages_read"]) == 74
    assert len(d["rendered_pages"]) == len(d["complete_rendered_pages_read"]) == 32
    assert set(d["ocr_pages_read"]) <= set(d["complete_rendered_pages_read"])
    assert len(d["native_pages_not_rendered"]) == 62
    assert d["unreviewed_pages"] == []
    assert d["all_content_proven_recovered"] is d["all_pages_visually_reviewed"] is False
    assert d["entire_procurement_file_reviewed"] is False
    assert all(v["replaces_archived_text"] is False for v in d["visual_observations"])
    assert len(d["evidence"]) == 21
    assert d["formal_services_evaluation_eligible"] is True
    assert d["domain"] == "servicios_instalacion_geosinteticos"
    assert all(d[k] is False for k in ("historical_content_proven", "ranking_eligible", "predictions_generated"))


def test_case21_count_keeps_supplier_tdr_role_and_unknown_fields():
    d = inputs()[0]["documents"][0]
    assert len(d["bidder_experience_conditions"]) == 1
    c = d["bidder_experience_conditions"][0]
    assert c["section_role"] == "tdr_supplier_profile"
    assert c["subject"] == "bidder" and c["minimum_similar_services"] == 3
    assert c["count_basis"] == "similar_services_executed"
    assert all(c[f] is None for f in ("billed_amount", "currency", "lookback_years", "reference_date", "window_anchor"))
    assert c["company_applicability"] == "not_determined" and not c["document_reference_complete"]
    observations = {o["branch"]: o for o in d["other_bidder_experience_observations"]}
    annex = observations["annex_specialty_template"]
    assert annex["lookback_years"] == 15 and annex["blank_template"]
    assert not annex["applied_to_supplier_profile"]
    conditional = observations["conditional_impediment_removal"]
    assert conditional["section_role"] == "conditional_admission"
    assert conditional["lookback_years"] == 2
    assert [a["minimum_contracts"] for a in conditional["alternatives"]] == [1, 4]
    assert conditional["activation_condition"] and not conditional["applied_to_supplier_profile"]
    personnel = d["personnel_experience_observations"]
    assert personnel[0]["engineer_service_counts"] == personnel[0]["technician_service_counts"] == [4, 3]
    assert not personnel[0]["source_conflict_resolved"]
    assert personnel[1]["lookback_years"] == 25 and personnel[2]["minimum_years"] == 2
    assert all(not p["applied_to_bidder"] for p in personnel)
    assert d["document_level_lookback_years"] is None


def test_case21_comparison_explicitly_checks_count_without_rewriting_first_pass():
    review, _, first = inputs()
    d = review["documents"][0]
    prior = first[d["opportunity_id"]]["bidder_experience_conditions"]
    before = copy.deepcopy(prior)
    comparisons = compare_conditions(prior, d["bidder_experience_conditions"])
    assert prior == before and len(comparisons) == 1
    c = comparisons[0]
    assert c["identity"]["section_role"] == "tdr_supplier_profile"
    assert c["fields"]["minimum_similar_services"] == {"first": 3, "second": 3, "status": "match"}
    assert all(c["fields"][f]["status"] == "unknown" for f in ("billed_amount", "currency", "lookback_years", "similar_services"))
    assert d["decision"] == "source_conditions_discrepant_pending_reference_lock"
    changed = copy.deepcopy(d["bidder_experience_conditions"])
    changed[0]["minimum_similar_services"] = 4
    assert compare_conditions(prior, changed)[0]["fields"]["minimum_similar_services"]["status"] == "disagreement"
    changed[0]["minimum_similar_services"] = None
    assert compare_conditions(prior, changed)[0]["fields"]["minimum_similar_services"]["status"] == "unknown"


@pytest.mark.parametrize("mutation", [
    "agreement", "missing_page", "fake_ocr", "missing_image", "personnel",
    "invented_quote", "eligibility", "lock", "pdf", "currency",
    "duplicate_image", "evidence", "duplicate_condition", "zero_count",
    "float_count", "boolean_count", "missing_basis", "annex_role",
    "section_missing_page", "hidden_unrendered", "false_visual_claim", "historical_use",
])
def test_case21_rejects_unsafe_registration(mutation):
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
        d["complete_rendered_pages_read"].remove(24)
    elif mutation == "personnel":
        c["subject"] = "personnel"
    elif mutation == "invented_quote":
        d["evidence"].append({"id": "fake", "page": 30, "quote": "NO EXPERIENCIA EXIGIDA"})
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
    elif mutation in ("zero_count", "float_count", "boolean_count"):
        c["minimum_similar_services"] = {"zero_count": 0, "float_count": 3.0, "boolean_count": True}[mutation]
    elif mutation == "missing_basis":
        c.pop("count_basis")
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


def test_case21_build_appends_without_changing_prior_inventories_or_gates(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 23
    assert result["condition_source_agreements_pending_reference"] == 4
    assert not result["f1_estimable"]
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    previous = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == previous and current[-1] == json.loads(REVIEW.read_text())
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    assert [r["sample_number"] for r in rows if r["first_pass_status"] == "full_supplied_pdf"
            and r["second_pass_status"] == "not_performed"] == [1, 11, 34, 40]
    case = next(r for r in rows if r["sample_number"] == 21)
    assert case["reference_adjudication_pending"] and case["second_pass_status"] == "full_supplied_pdf"
    assert len(case["second_pass_field_comparison"]) == 1
    assert case["second_pass_field_comparison"][0]["fields"]["minimum_similar_services"]["status"] == "match"
    for row in rows:
        assert not row["historical_ranking_eligible"] and not row["reference_locked"]
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
    evidence = [e for e in json.loads((output / "second_pass_evidence.json").read_text()) if e["opportunity_id"] == case["opportunity_id"]]
    assert len(evidence) == 21
    assert all(e["image_sha256"] and not e["human_or_expert"] and not e["independent_reviewer"] for e in evidence)
    assert {e["method"] for e in evidence} == {"native_pdf_text_and_full_page_visual_review", "apple_vision_local_ocr_and_full_page_visual_review"}
