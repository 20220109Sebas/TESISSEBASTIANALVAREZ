import copy
import json
from pathlib import Path

import pytest

from scripts.run_holdout_frozen_extractors import (
    branch_label, canonical_money, check_policy, comparison_gate, infer,
    prepare, project_outputs, project_prediction, project_reference,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/experience_holdout_reference_lock_v1"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
POLICY = ROOT / "docs/holdout_prediction_projection_v1.json"


def policy():
    return json.loads(POLICY.read_text())


def test_reference_projection_changes_only_global_branch_labels():
    original = json.loads((REFERENCE / "reference.json").read_text())
    before = copy.deepcopy(original)
    projected = project_reference(original, policy())
    assert original == before
    assert projected["serialization_projection"] == policy()["version"]
    for a, b in zip(original["documents"], projected["documents"]):
        assert a.keys() == b.keys()
        for x, y in zip(a["conditions"], b["conditions"]):
            adjusted = copy.deepcopy(x)
            adjusted["branch"] = branch_label(x["branch"], policy())
            assert adjusted == y
    d = {d["sample_number"]: d for d in projected["documents"]}
    assert d[8]["conditions"][0]["branch"] == "general"
    assert d[2]["conditions"][1]["branch"] == "mype"
    assert d[20]["conditions"][0]["item"] == 1
    assert d[13]["conditions"][1]["fields"]["lookback_years"] is None


@pytest.mark.parametrize("flag", ["reference_value_changes_permitted", "model_rule_changes_permitted",
    "predictions_seen_when_policy_fixed", "historical_ranking_eligibility", "new_ocr_permitted"])
def test_policy_rejects_tuning_or_prior_prediction_selection(flag):
    p = policy()
    p[flag] = True
    with pytest.raises(ValueError):
        check_policy(p)


def test_arbitrary_reference_branch_alias_or_reference_collision_rejected():
    p = policy()
    p["branch_aliases"]["personal"] = "general"
    with pytest.raises(ValueError):
        check_policy(p)
    original = json.loads((REFERENCE / "reference.json").read_text())
    d = next(d for d in original["documents"] if d["sample_number"] == 8)
    extra = copy.deepcopy(d["conditions"][0])
    extra["branch"] = "general"
    d["conditions"].append(extra)
    with pytest.raises(ValueError, match="aliases collide"):
        project_reference(original, policy())


@pytest.mark.parametrize("amount", ["0", "-1", "100.001", "NaN", "Infinity"])
def test_money_projection_never_rounds_or_infers(amount):
    with pytest.raises(ValueError):
        canonical_money(amount)


def test_projection_does_not_infer_item_subject_role_or_relative_amount():
    raw = {"subject": "supplier", "section_role": "supplier_profile", "absolute_amount": None,
           "currency": None, "multiplier": 3, "status": "requires_separate_review"}
    row = project_prediction("synthetic", raw, "supplier_profile_tdr", policy(), "test1", {})
    assert row["subject"] == "supplier" and row["section_role"] == "supplier_profile"
    assert row["item"] is None
    assert row["fields"] == {"billed_amount": None, "currency": None, "lookback_years": None}
    assert row["raw_status"] == "requires_separate_review"
    assert canonical_money("100") == "100.00"


def test_all_requires_review_branches_and_candidates_are_preserved():
    doc = {"opportunity_id": "synthetic", "conditional_variables": [{
        "subject": "bidder", "section_role": "qualification", "status": "requires_review_of_unresolved_segment",
        "source_clause_page": 3, "source_clause_sha256": "x", "branches": [
            {"branch_id": "primary_paragraph", "billed_amount": "100", "currency": "PEN", "lookback_years": None},
            {"branch_id": "micro_small_declaration_1", "billed_amount": "25", "currency": "PEN", "lookback_years": 8},
        ]}], "recovery_candidates": {"records": [{"subject": "bidder", "section_role": "qualification",
            "kind": "literal_money_in_unresolved_bidder_context", "status": "requires_separate_review",
            "billed_amount": "99", "currency": "PEN", "evidence": {}}]},
        "formulation_candidates": {"records": [{"subject": "supplier", "section_role": "supplier_profile",
            "kind": "minimum_prior_service_count", "status": "literal_candidate_development_only",
            "minimum_services": 3, "evidence": {}}]}}
    streams = project_outputs([doc], policy())
    assert len(streams["accepted_branches"]) == len(streams["additional_candidates"]) == 2
    assert [r["branch"] for r in streams["accepted_branches"]] == ["general", "mype"]
    assert streams["additional_candidates"][1]["fields"]["billed_amount"] is None


def test_duplicate_identity_gate_keeps_even_exact_duplicate_values():
    ref = {"documents": [{"opportunity_id": "a", "scope": "enterprise_services"},
                         {"opportunity_id": "b", "scope": "insufficient_reference"}]}
    rows = [{"opportunity_id": ocid, "subject": "bidder", "section_role": "qualification", "item": None,
             "branch": "general", "fields": {"billed_amount": "100.00"}, "prediction_id": str(i), "origin": {}}
            for i, ocid in enumerate(("a", "a", "b", "b"))]
    before = copy.deepcopy(rows)
    gate = comparison_gate(ref, rows)
    assert rows == before and gate["records_removed"] == 0
    assert not gate["evaluation_allowed"]
    assert len(gate["duplicate_groups_formal"]) == 1 and len(gate["duplicate_groups_all"]) == 2
    assert gate["duplicate_groups_formal"][0]["prediction_ids"] == ["0", "1"]


@pytest.fixture(scope="module")
def run_result(tmp_path_factory):
    root = tmp_path_factory.mktemp("frozen-holdout")
    plan = root / "plan"
    prepared = prepare(REFERENCE, SOURCE, POLICY, plan)
    output = root / "predictions"
    summary = infer(plan, output)
    return plan, output, prepared, summary


def test_real_run_is_source_only_and_preserves_fixed_reference(run_result):
    plan, output, prepared, summary = run_result
    assert prepared["projection_locked_before_predictions"] and not prepared["predictions_generated"]
    assert summary["documents_processed"] == 27 and summary["selected_cases_retained"] == 50
    assert summary["physical_pages_processed"] == 1667
    assert summary["predictions_generated"] and summary["literal_provenance_verified"]
    assert all(summary[f] is False for f in ("reference_values_modified", "model_rules_modified",
        "new_ocr_performed", "ranking_inputs_modified", "semantic_accuracy_independently_proven"))
    assert summary["reference_and_adapter_locked_before_inference"] and summary["no_emitted_records_removed"]
    streams = json.loads((output / "prediction_streams.json").read_text())
    gates = json.loads((output / "comparison_gates.json").read_text())
    assert all(len(streams[name]) == gates[name]["emitted_records"] for name in streams)
    if any(not g["evaluation_allowed"] for g in gates.values()):
        assert not summary["f1_computed"]
        assert json.loads((output / "metrics.json").read_text()) is None
        assert summary["comparison_status"] == "stopped_duplicate_prediction_identities"
    raw = json.loads((output / "raw_extractor_outputs.json").read_text())
    assert len(raw) == 27
    assert summary["additional_candidates"] == sum(len(d[p]["records"]) for d in raw for p in ("recovery_candidates", "formulation_candidates"))
    run_plan = json.loads((plan / "run_plan.json").read_text())
    assert run_plan["parameters"]["segments_max_pages"] == 2


def test_inference_reproduces_all_outputs_and_rejects_overwrite(run_result, tmp_path):
    plan, original, _, _ = run_result
    output = tmp_path / "predictions"
    infer(plan, output)
    assert sorted(p.name for p in output.iterdir()) == sorted(p.name for p in original.iterdir())
    assert all((output / p.name).read_bytes() == p.read_bytes() for p in original.iterdir())
    with pytest.raises(ValueError, match="fresh prediction directory"):
        infer(plan, output)


def test_reference_or_adapter_tampering_stops_before_inference(run_result, tmp_path):
    plan, _, _, _ = run_result
    # Tamper only with a temporary test copy, never the actual frozen reference.
    import shutil
    altered = tmp_path / "plan-copy"
    shutil.copytree(plan, altered)
    path = altered / "projection_policy.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        infer(altered, tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()
