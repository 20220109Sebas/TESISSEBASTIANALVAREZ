"""Grade saved predictions after the explicitly authorized case-15 resolution.

No extraction is repeated, no model output is repaired and no frozen file changes.
Every original emission is retained; only one approved textual repeat is folded
into a logical condition for the existing exact-identity field comparator.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes
from scripts.run_holdout_frozen_extractors import comparison_gate
from tesis_recommender.reference_evaluation import evaluate, key

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("billed_amount", "currency", "lookback_years")
TARGET = "ocds-dgv273-seacev3-1101046"


def validate_resolution(resolution):
    if (resolution.get("version") != "holdout_explicit_duplicate_resolution_v1"
            or resolution.get("stream") != "accepted_branches"
            or resolution.get("remaining_duplicate_policy") != "stop"
            or resolution.get("authorization", {}).get("source") != "direct_user_confirmation_in_current_chat"
            or resolution["authorization"].get("choice") != "one_logical_condition_report_original_emissions_separately"
            or not resolution["authorization"].get("quote")):
        raise ValueError("Require explicit user-authorized condition consolidation")
    for flag in ("pre_inference_decision", "human_source_annotation", "expert_source_annotation",
                 "independent_validation", "selection_by_reference_values", "repair_or_fill_fields",
                 "item_assignment_from_reference", "model_changes", "reference_changes", "ranking_changes"):
        if resolution.get(flag) is not False:
            raise ValueError("No blind-decision claim, tuning, repair or company inference allowed")
    expected = {"sample_number": 15, "opportunity_id": TARGET, "subject": "bidder",
                "section_role": "qualification", "item": None, "branch": "general"}
    if resolution["target"] != expected or resolution["origin_pages"] != [42, 45]:
        raise ValueError("The authorization applies only to case15 pages42/45")
    if resolution["prediction_ids"] != [f"{TARGET}:accepted:0:0", f"{TARGET}:accepted:1:0"]:
        raise ValueError("Do not generalize this decision to other emissions")
    if len(resolution["origin_clause_sha256"]) != 2 or len(set(resolution["origin_clause_sha256"])) != 2:
        raise ValueError("Both original textual locations must be pinned")


def consolidate(reference, streams, resolution):
    validate_resolution(resolution)
    original = streams[resolution["stream"]]
    ids = resolution["prediction_ids"]
    matched = [r for r in original if r["prediction_id"] in ids]
    if len(matched) != 2 or {r["prediction_id"] for r in matched} != set(ids):
        raise ValueError("Require both unique saved emissions")
    selected = sorted(matched, key=lambda r: (r["origin"]["page"], r["prediction_id"]))
    expected_key = key(resolution["target"])
    if any(key(r) != expected_key for r in selected):
        raise ValueError("Identity differs; cannot consolidate by the amount")
    if selected[0]["fields"] != selected[1]["fields"]:
        raise ValueError("Conflicting fields require separate adjudication; no best-value selection")
    if ([r["origin"]["page"] for r in selected] != resolution["origin_pages"]
            or [r["origin"]["clause_sha256"] for r in selected] != resolution["origin_clause_sha256"]):
        raise ValueError("Source location differs from authorized repeated condition")
    doc = next((d for d in reference["documents"] if d["opportunity_id"] == TARGET), None)
    if not doc or doc["scope"] != "enterprise_services" or doc["sample_number"] != 15:
        raise ValueError("Case15 is not in the locked formal reference scope")
    conditions = [c for c in doc["conditions"] if key(c) == expected_key]
    if (len(conditions) != 1
            or conditions[0]["source_annotation"].get("same_condition_repeated_on_pages") != [42, 45]
            or conditions[0]["first_pass_condition"].get("condition_count") != 1):
        raise ValueError("Pre-inference source annotation must identify one repeated condition")
    # The two field dictionaries must agree with each other, not with reference
    # values: equal wrong predictions would remain wrong after consolidation.
    logical = copy.deepcopy(selected[0])
    logical.update(prediction_id=f"{TARGET}:logical:pages42_45",
                   consolidation="user_authorized_same_condition_in_two_source_locations",
                   original_prediction_ids=ids,
                   original_emissions=copy.deepcopy(selected),
                   original_origins=[copy.deepcopy(r["origin"]) for r in selected])
    projected, inserted = [], False
    for row in original:
        if row["prediction_id"] in ids:
            if not inserted:
                projected.append(logical)
                inserted = True
        else:
            projected.append(copy.deepcopy(row))
    result = copy.deepcopy(streams)
    result[resolution["stream"]] = projected
    gates = {name: comparison_gate(reference, rows) for name, rows in result.items()}
    if any(not gate["evaluation_allowed"] for gate in gates.values()):
        raise ValueError("An unapproved duplicate remains; stop before metrics")
    audit = {"sample_number": 15, "opportunity_id": TARGET, "original_emission_count": len(original),
             "logical_condition_count": len(projected), "textual_repetitions_consolidated": 1,
             "resolved_groups": 1, "original_emissions_preserved": True,
             "original_prediction_ids": ids, "original_emissions": copy.deepcopy(selected),
             "logical_prediction": logical, "decision": copy.deepcopy(resolution),
             "reference_values_used_to_select_prediction": False}
    return result, gates, audit


def field_audit(reference, predictions):
    eligible = {d["opportunity_id"] for d in reference["documents"] if d["scope"] == "enterprise_services"}
    truth = {key(c): c for d in reference["documents"] if d["opportunity_id"] in eligible for c in d["conditions"]}
    pred = {key(c): c for c in predictions if c["opportunity_id"] in eligible}
    cells = []
    for identity in sorted(set(truth) | set(pred), key=repr):
        actual, emitted = truth.get(identity), pred.get(identity)
        states = {}
        for field in FIELDS:
            known = actual is not None and actual["fields"].get(field) is not None
            present = emitted is not None and emitted["fields"].get(field) is not None
            ref_value = actual["fields"].get(field) if actual else None
            pred_value = emitted["fields"].get(field) if emitted else None
            if actual is not None and not known:
                status, tp, fp, fn = "reference_unknown_ungraded", 0, 0, 0
            elif known and present and ref_value == pred_value:
                status, tp, fp, fn = "exact_match", 1, 0, 0
            else:
                status = ("wrong_value" if known and present else "missing_known_value" if known
                          else "spurious_identity_value" if present else "no_supported_value")
                tp, fp, fn = 0, int(present), int(known)
            states[field] = {"reference_value": ref_value, "prediction_value": pred_value,
                             "status": status, "tp": tp, "fp": fp, "fn": fn}
        cells.append({"identity": dict(zip(("opportunity_id", "subject", "section_role", "item", "branch"), identity)),
                      "reference_present": actual is not None, "prediction_present": emitted is not None,
                      "prediction_id": emitted["prediction_id"] if emitted else None,
                      "prediction_origin": copy.deepcopy(emitted["origin"]) if emitted else None,
                      "fields": states})
    return cells


def build(prediction_dir, plan_dir, decision_path, output):
    if output.exists():
        raise ValueError("Use a fresh concordance directory; preserve original execution")
    inputs = dict(protected_outputs())
    for directory in (prediction_dir, plan_dir):
        verify_artifact(directory, inputs)
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    decision = json.loads(decision_path.read_text())
    validate_resolution(decision)
    expected_paths = {str(prediction_dir / name) for name in ("prediction_streams.json", "comparison_gates.json")}
    expected_paths.add(str(plan_dir / "projected_reference.json"))
    if set(decision["input_sha256"]) != expected_paths:
        raise ValueError("Resolution must pin these actual frozen inputs")
    verify_hashes(decision["input_sha256"])
    inputs.update(decision["input_sha256"])
    plan = json.loads((plan_dir / "run_plan.json").read_text())
    verify_hashes({plan["original_reference_path"]: plan["original_reference_sha256"],
                   **plan["frozen_model_code_sha256"], **plan["comparison_code_sha256"]})
    initial = json.loads((prediction_dir / "summary.json").read_text())
    if (initial["comparison_status"] != "stopped_duplicate_prediction_identities"
            or initial["predictions_generated"] is not True or initial["f1_computed"] is not False
            or json.loads((prediction_dir / "metrics.json").read_text()) is not None):
        raise ValueError("Original stopped execution must remain unchanged")
    reference = json.loads((plan_dir / "projected_reference.json").read_text())
    streams = json.loads((prediction_dir / "prediction_streams.json").read_text())
    before_reference, before_streams = copy.deepcopy(reference), copy.deepcopy(streams)
    resolved, gates, consolidation = consolidate(reference, streams, decision)
    assert reference == before_reference and streams == before_streams
    metrics = {name: evaluate(reference, rows, list(FIELDS)) for name, rows in resolved.items()}
    diagnostics = {name: field_audit(reference, rows) for name, rows in resolved.items()}
    for name, cells in diagnostics.items():
        for field in FIELDS:
            for count in ("tp", "fp", "fn"):
                if sum(c["fields"][field][count] for c in cells) != metrics[name]["fields"][field][count]:
                    raise ValueError("Field audit differs from unchanged frozen comparator")
    coverage = [{"sample_number": d["sample_number"], "opportunity_id": d["opportunity_id"],
                 "scope": d["scope"], "reason": d["reason"],
                 "reference_conditions": len(d["conditions"]),
                 "raw_accepted_emissions": sum(r["opportunity_id"] == d["opportunity_id"] for r in streams["accepted_branches"]),
                 "logical_accepted_conditions": sum(r["opportunity_id"] == d["opportunity_id"] for r in resolved["accepted_branches"]),
                 "additional_candidates": sum(r["opportunity_id"] == d["opportunity_id"] for r in streams["additional_candidates"]),
                 "formal_evaluation_eligible": d["scope"] == "enterprise_services",
                 "ranking_eligible": False} for d in reference["documents"]]
    if len(coverage) != 50 or len({r["opportunity_id"] for r in coverage}) != 50:
        raise ValueError("Preserve the fixed fifty-case denominator")
    summary = {"version": "holdout_concordance_after_explicit_resolution_v1",
        "selected_cases": len(coverage), "scope_counts": dict(Counter(r["scope"] for r in coverage)),
        "evaluable_documents": metrics["accepted_branches"]["evaluable_documents"],
        "formal_reference_conditions": metrics["accepted_branches"]["reference_conditions"],
        "raw_accepted_emissions": len(streams["accepted_branches"]),
        "logical_accepted_conditions": len(resolved["accepted_branches"]),
        "additional_candidates": len(streams["additional_candidates"]),
        "explicitly_consolidated_repeat_groups": 1, "original_emissions_preserved": True,
        "post_inference_resolution_applied": True, "end_to_end_metric_fully_precommitted": False,
        "reference_frozen_before_inference": True, "comparison_code_changed": False,
        "comparison_status": "completed_after_user_authorized_duplicate_resolution",
        "f1_computed": True, "reference_type": "automatic_source_review",
        "semantic_accuracy_independently_proven": False, "human_or_expert_reference": False,
        "model_rules_changed": False, "predictions_recomputed": False,
        "reference_values_changed": False, "ranking_inputs_modified": False,
        "interpretation": decision["metrics_interpretation"],
        "field_metrics": {name: result["fields"] for name, result in metrics.items()}}
    paths = (decision_path, Path(__file__).resolve(), ROOT / "tests/test_holdout_consolidated_concordance.py")
    for path in paths:
        inputs[str(path)] = sha(path)
    verify_hashes(inputs)
    output.mkdir(parents=True)
    for filename, value in (("consolidation_audit.json", consolidation), ("logical_prediction_streams.json", resolved),
                            ("comparison_gates.json", gates), ("metrics.json", metrics),
                            ("field_diagnostics.json", diagnostics), ("coverage_ledger.json", coverage),
                            ("summary.json", summary)):
        dump(output / filename, value)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "plan", "decision", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.predictions.resolve(), args.plan.resolve(), args.decision.resolve(), args.output.resolve())))
