import copy
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v2 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v28"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case32_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_saved_review_is_valid():
    validate(*inputs())


@pytest.mark.parametrize("mutation", ["activation", "missing_activation", "page_count", "method", "missing_render", "duplicate_image", "outside_root", "bad_renderer", "unknown_quote", "false_mype_amount"])
def test_reject_unsafe_registration(mutation):
    review, sources, first = inputs()
    doc = review["documents"][0]
    if mutation == "activation": doc["bidder_experience_conditions"][1]["activation_condition"] = "Todos califican"
    elif mutation == "missing_activation": doc["bidder_experience_conditions"][1].pop("activation_condition")
    elif mutation == "page_count": doc["physical_pages"] = 56
    elif mutation == "method":
        doc["native_pages_read"].remove(1)
        doc["ocr_pages_read"].append(1)
        doc["complete_rendered_pages_read"].append(1)
    elif mutation == "missing_render": doc["rendered_pages"].pop()
    elif mutation == "duplicate_image": doc["rendered_pages"][1]["path"] = doc["rendered_pages"][0]["path"]
    elif mutation == "outside_root": doc["rendered_pages"][0]["path"] = "/private/tmp/missing.png"
    elif mutation == "bad_renderer": doc["rendered_pages"][0]["renderer"] = ""
    elif mutation == "unknown_quote": doc["evidence"][0]["quote"] = "inventado"
    elif mutation == "false_mype_amount": doc["bidder_experience_conditions"][1]["billed_amount"] = "160560.00"
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_activation_comparison_does_not_infer_eligibility():
    review, _, _ = inputs()
    conditions = review["documents"][0]["bidder_experience_conditions"]
    altered = copy.deepcopy(conditions)
    altered[1]["activation_condition"] = None
    result = compare_conditions(conditions, altered)
    assert result[0]["fields"]["activation_condition"]["status"] == "unspecified_in_both"
    assert result[1]["fields"]["activation_condition"]["status"] == "unknown"


def test_build_preserves_inventories_and_evaluation_gates(tmp_path):
    output = tmp_path / "progress"
    summary = build(BASE, SOURCE, REVIEW, output)
    assert summary["second_pass_full_supplied_pdf"] == 9
    assert summary["f1_estimable"] is False
    for filename in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (output / filename).read_bytes() == (BASE / filename).read_bytes()
    inventory = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert inventory[:-1] == json.loads((BASE / "condition_second_source_reviews.json").read_text())
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    assert all(r["reference_locked"] is False and r["historical_ranking_eligible"] is False for r in rows)
    with pytest.raises(ValueError, match="fresh"):
        build(BASE, SOURCE, REVIEW, output)
