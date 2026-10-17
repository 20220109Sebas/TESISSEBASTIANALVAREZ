import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v2 import build, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v29"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case36_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_review_preserves_one_repeated_condition_and_empty_page_provenance():
    review, sources, first = inputs()
    validate(review, sources, first)
    doc = review["documents"][0]
    assert doc["complete_rendered_pages_read"] == list(range(1, 63))
    assert doc["ocr_pages_read"] == list(range(1, 62))
    assert doc["native_pages_read"] == [62]
    last = sources[doc["opportunity_id"]]["pages"][-1]
    assert last["text"] == last["native_text"] == ""
    assert last["selected_ocr_unresolved"] is True
    assert last["ocr_used"] is False
    assert len(doc["bidder_experience_conditions"]) == 1
    condition = doc["bidder_experience_conditions"][0]
    assert condition["billed_amount"] == "5000000.00"
    assert condition["lookback_years"] == 8
    assert condition["evidence_ids"] == ["bidder_clause", "tdr_repeated_clause"]
    assert {e["page"] for e in doc["evidence"] if e["id"] in condition["evidence_ids"]} == {38, 43}
    assert all(e["page"] != 62 for e in doc["evidence"])


@pytest.mark.parametrize("mutation", [
    "personnel_years", "personnel_subject", "insurance_amount", "invented_mype",
    "duplicated_condition", "missing_last_page", "missing_scanned_visual",
    "false_native_last_page", "invented_empty_page_quote", "eligibility_claim",
])
def test_unsafe_case36_changes_are_rejected(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    condition = doc["bidder_experience_conditions"][0]
    if mutation == "personnel_years": condition["lookback_years"] = 25
    elif mutation == "personnel_subject": condition["subject"] = "personnel"
    elif mutation == "insurance_amount": condition["billed_amount"] = "10000.00"
    elif mutation == "invented_mype":
        extra = copy.deepcopy(condition)
        extra["branch"] = "mype"
        doc["bidder_experience_conditions"].append(extra)
    elif mutation == "duplicated_condition": doc["bidder_experience_conditions"].append(copy.deepcopy(condition))
    elif mutation == "missing_last_page": doc["pages_read"].remove(62)
    elif mutation == "missing_scanned_visual": doc["complete_rendered_pages_read"].remove(38)
    elif mutation == "false_native_last_page":
        doc["native_pages_read"] = []
        doc["ocr_pages_read"].append(62)
    elif mutation == "invented_empty_page_quote": doc["evidence"].append({"id": "empty", "page": 62, "quote": "sin requisitos"})
    elif mutation == "eligibility_claim": condition["company_applicability"] = "eligible"
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_build_appends_tenth_rereading_without_unlocking_experiment(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 10
    assert result["condition_source_agreements_pending_reference"] == 3
    assert result["f1_estimable"] is False
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    previous = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == previous
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    assert sum(r["first_pass_status"] == "full_supplied_pdf" and r["second_pass_status"] == "not_performed" for r in rows) == 17
    assert all(r["reference_locked"] is False and r["historical_ranking_eligible"] is False for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
