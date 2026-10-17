import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass import compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    review = json.loads((ROOT / "docs/holdout_case39_second_source_pass_v1.json").read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((ROOT / "artifacts/experience_holdout_source_view_v1/source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((ROOT / "artifacts/experience_holdout_review_progress_v27/compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_saved_review_is_valid():
    validate(*inputs())


@pytest.mark.parametrize("mutation", ["duplicate", "partial", "duplicate_page", "wrong_hash", "bad_quote", "unviewed", "predictions", "independent", "locked", "company", "false_agreement", "float_amount", "unknown_evidence"])
def test_reject_unsafe_record(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    condition = doc["bidder_experience_conditions"][0]
    if mutation == "duplicate": review["documents"].append(copy.deepcopy(doc))
    elif mutation == "partial": doc["pages_read"].pop()
    elif mutation == "duplicate_page": doc["native_pages_read"].append(1)
    elif mutation == "wrong_hash": doc["pdf_sha256"] = "wrong"
    elif mutation == "bad_quote": doc["evidence"][0]["quote"] = "invented"
    elif mutation == "unviewed": doc["complete_rendered_pages_read"].remove(27)
    elif mutation == "predictions": review["predictions_consulted"] = True
    elif mutation == "independent": review["independent_reviewer"] = True
    elif mutation == "locked": condition["document_reference_complete"] = True
    elif mutation == "company": condition["company_applicability"] = "eligible"
    elif mutation == "false_agreement": condition["lookback_years"] = 25
    elif mutation == "float_amount": condition["billed_amount"] = 700000.0
    elif mutation == "unknown_evidence": condition["evidence_ids"] = ["invented"]
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_matching_does_not_pair_by_amount():
    review, _, _ = inputs()
    conditions = review["documents"][0]["bidder_experience_conditions"]
    changed = copy.deepcopy(conditions)
    changed[0]["branch"] = "mype"
    comparisons = compare_conditions(conditions, changed)
    assert len(comparisons) == 2
    assert all(not (c["first_pass_present"] and c["second_pass_present"]) for c in comparisons)


def test_unknown_is_not_a_confirmed_value():
    review, _, _ = inputs()
    conditions = review["documents"][0]["bidder_experience_conditions"]
    changed = copy.deepcopy(conditions)
    changed[0]["lookback_years"] = None
    assert compare_conditions(conditions, changed)[0]["fields"]["lookback_years"]["status"] == "unknown"
