import copy
import json
from pathlib import Path

import pytest

from scripts.evaluate_holdout_consolidated_predictions import (
    TARGET, build, consolidate, field_audit, validate_resolution,
)
from tesis_recommender.reference_evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "artifacts/experience_holdout_predictions_v1"
PLAN = ROOT / "artifacts/experience_holdout_inference_plan_v1"
DECISION = ROOT / "docs/holdout_case15_duplicate_resolution_v1.json"


def inputs():
    return (json.loads((PLAN / "projected_reference.json").read_text()),
            json.loads((PREDICTIONS / "prediction_streams.json").read_text()),
            json.loads(DECISION.read_text()))


def test_only_approved_repeat_is_consolidated_with_originals_preserved():
    ref, streams, decision = inputs()
    prior_ref, prior_streams = copy.deepcopy(ref), copy.deepcopy(streams)
    logical, gates, audit = consolidate(ref, streams, decision)
    assert ref == prior_ref and streams == prior_streams
    assert len(streams["accepted_branches"]) == 18
    assert len(logical["accepted_branches"]) == 17
    assert logical["additional_candidates"] == streams["additional_candidates"] == []
    rows = [r for r in logical["accepted_branches"] if r["opportunity_id"] == TARGET]
    assert len(rows) == 1
    assert [r["origin"]["page"] for r in rows[0]["original_emissions"]] == [42, 45]
    assert rows[0]["original_prediction_ids"] == decision["prediction_ids"]
    assert audit["original_emissions_preserved"] and not audit["reference_values_used_to_select_prediction"]
    assert all(g["evaluation_allowed"] for g in gates.values())
    assert decision["pre_inference_decision"] is False


@pytest.mark.parametrize("mutation", [
    "precommitted_claim", "no_confirmation", "wrong_case", "wrong_ids", "missing_emission",
    "conflicting_values", "wrong_origin", "wrong_identity", "new_duplicate",
    "missing_source_repeat_annotation", "reference_repair_permission",
])
def test_unauthorized_or_conflicting_consolidation_stops(mutation):
    ref, streams, decision = inputs()
    originals = [r for r in streams["accepted_branches"] if r["prediction_id"] in decision["prediction_ids"]]
    if mutation == "precommitted_claim":
        decision["pre_inference_decision"] = True
    elif mutation == "no_confirmation":
        decision["authorization"]["quote"] = ""
    elif mutation == "wrong_case":
        decision["target"]["sample_number"] = 20
    elif mutation == "wrong_ids":
        decision["prediction_ids"][0] = "other_case"
    elif mutation == "missing_emission":
        streams["accepted_branches"].remove(originals[1])
    elif mutation == "conflicting_values":
        originals[1]["fields"]["billed_amount"] = "800000.00"
    elif mutation == "wrong_origin":
        originals[1]["origin"]["page"] = 44
    elif mutation == "wrong_identity":
        originals[1]["item"] = 1
    elif mutation == "new_duplicate":
        extra = copy.deepcopy(streams["accepted_branches"][0])
        extra["prediction_id"] = "new_unapproved_duplicate"
        streams["accepted_branches"].append(extra)
    elif mutation == "missing_source_repeat_annotation":
        doc = next(d for d in ref["documents"] if d["sample_number"] == 15)
        doc["conditions"][0]["source_annotation"]["same_condition_repeated_on_pages"] = []
    elif mutation == "reference_repair_permission":
        decision["repair_or_fill_fields"] = True
    with pytest.raises(ValueError):
        consolidate(ref, streams, decision)


def test_equal_wrong_values_are_not_replaced_with_reference_values():
    ref, streams, decision = inputs()
    for row in streams["accepted_branches"]:
        if row["prediction_id"] in decision["prediction_ids"]:
            row["fields"]["billed_amount"] = "900000.00"
    logical, _, _ = consolidate(ref, streams, decision)
    row = next(r for r in logical["accepted_branches"] if r["opportunity_id"] == TARGET)
    assert row["fields"]["billed_amount"] == "900000.00"
    audit = field_audit(ref, logical["accepted_branches"])
    condition = next(c for c in audit if c["identity"]["opportunity_id"] == TARGET)
    assert condition["fields"]["billed_amount"]["status"] == "wrong_value"
    assert condition["fields"]["billed_amount"]["fp"] == condition["fields"]["billed_amount"]["fn"] == 1


def test_no_item_inference_or_unknown_reference_repair():
    ref, streams, decision = inputs()
    logical, _, _ = consolidate(ref, streams, decision)
    ocid20 = next(d["opportunity_id"] for d in ref["documents"] if d["sample_number"] == 20)
    assert all(r["item"] is None for r in logical["accepted_branches"] if r["opportunity_id"] == ocid20)
    rows = field_audit(ref, logical["accepted_branches"])
    assert len([r for r in rows if r["identity"]["opportunity_id"] == ocid20 and r["prediction_present"] and not r["reference_present"]]) == 2
    result = evaluate(ref, logical["accepted_branches"], ["billed_amount", "currency", "lookback_years"])
    assert result["fields"]["billed_amount"]["reference_unknown"] == 3
    assert result["fields"]["billed_amount"]["predictions_ungraded_unknown_reference"] == 1


@pytest.fixture(scope="module")
def evaluation(tmp_path_factory):
    output = tmp_path_factory.mktemp("consolidated-holdout") / "evaluation"
    summary = build(PREDICTIONS, PLAN, DECISION, output)
    return output, summary


def test_frozen_comparator_counts_and_denominators(evaluation):
    output, summary = evaluation
    assert summary["selected_cases"] == 50 and summary["evaluable_documents"] == 16
    assert summary["formal_reference_conditions"] == 25
    assert summary["raw_accepted_emissions"] == 18 and summary["logical_accepted_conditions"] == 17
    assert summary["explicitly_consolidated_repeat_groups"] == 1
    accepted = summary["field_metrics"]["accepted_branches"]
    for field, counts in {"billed_amount": (12, 4, 10), "currency": (15, 2, 9), "lookback_years": (15, 2, 8)}.items():
        assert tuple(accepted[field][k] for k in ("tp", "fp", "fn")) == counts
        tp, fp, fn = counts
        assert accepted[field]["precision"] == pytest.approx(tp / (tp + fp))
        assert accepted[field]["recall"] == pytest.approx(tp / (tp + fn))
        assert accepted[field]["f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    coverage = json.loads((output / "coverage_ledger.json").read_text())
    assert len(coverage) == len({r["opportunity_id"] for r in coverage}) == 50
    assert sum(r["formal_evaluation_eligible"] for r in coverage) == 16
    assert not summary["end_to_end_metric_fully_precommitted"]
    assert summary["post_inference_resolution_applied"] and summary["reference_frozen_before_inference"]


def test_zero_additional_candidates_does_not_mean_perfect_precision(evaluation):
    _, summary = evaluation
    assert summary["additional_candidates"] == 0
    additional = summary["field_metrics"]["additional_candidates"]
    for field, support in (("billed_amount", 22), ("currency", 24), ("lookback_years", 23)):
        assert additional[field]["tp"] == additional[field]["fp"] == 0
        assert additional[field]["fn"] == support
        assert additional[field]["precision"] is None
        assert additional[field]["recall"] == additional[field]["f1"] == 0


def test_no_rerun_or_frozen_changes_and_original_failure_retained(evaluation):
    _, summary = evaluation
    assert summary["f1_computed"] and summary["original_emissions_preserved"]
    assert all(not summary[f] for f in ("comparison_code_changed", "model_rules_changed", "predictions_recomputed",
        "reference_values_changed", "ranking_inputs_modified", "semantic_accuracy_independently_proven", "human_or_expert_reference"))
    original = json.loads((PREDICTIONS / "summary.json").read_text())
    assert original["comparison_status"] == "stopped_duplicate_prediction_identities" and not original["f1_computed"]
    assert json.loads((PREDICTIONS / "metrics.json").read_text()) is None


def test_evaluation_reproduces_all_outputs_and_rejects_overwrite(evaluation, tmp_path):
    original, _ = evaluation
    output = tmp_path / "reproduction"
    build(PREDICTIONS, PLAN, DECISION, output)
    assert sorted(p.name for p in original.iterdir()) == sorted(p.name for p in output.iterdir())
    assert all((output / p.name).read_bytes() == p.read_bytes() for p in original.iterdir())
    with pytest.raises(ValueError, match="fresh concordance directory"):
        build(PREDICTIONS, PLAN, DECISION, output)


def test_wrong_input_pin_aborts_without_an_output(tmp_path):
    _, _, decision = inputs()
    path = next(iter(decision["input_sha256"]))
    decision["input_sha256"][path] = "0" * 64
    altered = tmp_path / "bad-decision.json"
    altered.write_text(json.dumps(decision))
    with pytest.raises(ValueError):
        build(PREDICTIONS, PLAN, altered, tmp_path / "never-written")
    assert not (tmp_path / "never-written").exists()
