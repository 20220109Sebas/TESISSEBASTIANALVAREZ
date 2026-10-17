import copy
import json
from pathlib import Path
import pytest
from scripts.record_holdout_insufficient_second_pass import validate

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    review = json.loads((ROOT / "docs/holdout_oefa_second_pass_v1.json").read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((ROOT / "artifacts/experience_holdout_source_view_v1/source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((ROOT / "docs/reviewer_holdout_source_review_v1.json").read_text())["documents"]}
    return review, sources, first


def test_saved_review_is_valid():
    validate(*inputs())


@pytest.mark.parametrize("mutation", ["positive", "duplicate", "partial", "wrong_hash", "bad_quote", "unviewed", "predictions", "independent", "conditions"])
def test_reject_unsafe_record(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    if mutation == "positive": doc["decision"] = "experience_present"
    elif mutation == "duplicate": review["documents"].append(copy.deepcopy(doc))
    elif mutation == "partial": doc["pages_read"].pop()
    elif mutation == "wrong_hash": doc["pdf_sha256"] = "wrong"
    elif mutation == "bad_quote": doc["evidence"][0]["quote"] = "invented"
    elif mutation == "unviewed": doc["complete_rendered_pages_read"].remove(3)
    elif mutation == "predictions": review["predictions_consulted"] = True
    elif mutation == "independent": review["independent_reviewer"] = True
    elif mutation == "conditions": doc["bidder_experience_conditions"] = [{"years": 1}]
    with pytest.raises(ValueError): validate(review, sources, first)
