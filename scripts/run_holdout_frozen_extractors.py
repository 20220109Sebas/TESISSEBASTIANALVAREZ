"""Run fixed extractors, preserve every emission and honor duplicate-identity gates.

Reference projection is a pre-inference, value-preserving serialization adapter.
Extraction receives only archived pages. It never sees reference values or notes.
"""
import argparse
import copy
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import validate_source_document, verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes
from scripts.compare_experience_segments import validate_result
from scripts.represent_experience_conditions import verify_variable
from scripts.extend_experience_formulations import verify_record, verify_ref
from scripts.recover_unresolved_experience import verify_candidate
from tesis_recommender.experience_segments import extract_experience_segments
from tesis_recommender.experience_conditions import represent_experience_clause
from tesis_recommender.experience_formulations import extract_formulations
from tesis_recommender.experience_recovery_candidates import recover_candidates
from tesis_recommender.reference_evaluation import evaluate, key

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("billed_amount", "currency", "lookback_years")


def check_policy(policy):
    expected = {"primary_paragraph": "general", "general": "general", "mype": "mype",
                "declared_micro_small_enterprise": "mype"}
    if (policy["version"] != "holdout_prediction_projection_v1"
            or policy["branch_aliases"] != expected
            or policy["model_mype_branch_prefix"] != "micro_small_declaration_"
            or policy["model_mype_branch_target"] != "mype"
            or policy["streams"] != ["accepted_branches", "additional_candidates"]
            or policy["identity_policy"] != "exact_identity_no_best_match_by_amount"):
        raise ValueError("Unexpected serialization policy")
    for flag in ("reference_value_changes_permitted", "model_rule_changes_permitted",
                 "predictions_seen_when_policy_fixed", "historical_ranking_eligibility", "new_ocr_permitted"):
        if policy[flag] is not False:
            raise ValueError("Adapter cannot tune, repair OCR or modify reference values")


def branch_label(label, policy):
    if isinstance(label, str) and label.startswith(policy["model_mype_branch_prefix"]):
        suffix = label[len(policy["model_mype_branch_prefix"]):]
        if not suffix.isdigit() or int(suffix) <= 0:
            raise ValueError("Unrecognized model MYPE branch suffix")
        return policy["model_mype_branch_target"]
    return policy["branch_aliases"].get(label, label)


def canonical_money(raw):
    if raw is None:
        return None
    value = Decimal(str(raw))
    if not value.is_finite() or value <= 0 or value != value.quantize(Decimal("0.01")):
        raise ValueError("Cannot round or infer a predicted amount")
    return format(value, ".2f")


def project_reference(original, policy):
    if (original["locked"] is not True or original["predictions_seen_before_lock"] is not False
            or original["reference_type"] != "automatic_source_review" or original["is_human_or_expert"] is not False):
        raise ValueError("Require existing source-only reference lock")
    projected = copy.deepcopy(original)
    for doc in projected["documents"]:
        for condition in doc["conditions"]:
            condition["branch"] = branch_label(condition["branch"], policy)
    projected["serialization_projection"] = policy["version"]
    identities = [key(c) for d in projected["documents"] if d["scope"] == "enterprise_services" for c in d["conditions"]]
    if len(identities) != len(set(identities)):
        raise ValueError("Branch aliases collide in reference; explicit pre-inference resolution required")
    return projected


def prepare(reference_dir, source_dir, policy_path, output):
    if output.exists():
        raise ValueError("Use a fresh pre-inference directory")
    inputs = dict(protected_outputs())
    for directory in (reference_dir, source_dir):
        verify_artifact(directory, inputs)
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    lock = json.loads((reference_dir / "reference_lock.json").read_text())
    verify_hashes({str(reference_dir / "reference.json"): lock["reference_sha256"],
                   lock["protocol_path"]: lock["protocol_sha256"],
                   **lock["comparison_code_sha256"], **lock["frozen_model_code_sha256"]})
    inputs.update(lock["comparison_code_sha256"])
    inputs.update(lock["frozen_model_code_sha256"])
    policy = json.loads(policy_path.read_text())
    check_policy(policy)
    reference = project_reference(json.loads((reference_dir / "reference.json").read_text()), policy)
    source_path = source_dir / "source_documents.json"
    for doc in json.loads(source_path.read_text()):
        validate_source_document(doc)
    helper_paths = (Path(__file__).resolve(), ROOT / "scripts/audit_holdout_evaluation_readiness.py",
                    ROOT / "scripts/register_browser_experience_sources.py",
                    ROOT / "tests/test_holdout_frozen_inference.py", policy_path)
    for path in helper_paths:
        inputs[str(path)] = sha(path)
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "projected_reference.json", reference)
    dump(output / "projection_policy.json", policy)
    dump(output / "run_plan.json", {
        "version": "holdout_frozen_inference_plan_v1", "original_reference_sha256": lock["reference_sha256"],
        "projected_reference_sha256": sha(output / "projected_reference.json"),
        "source_documents_path": str(source_path), "source_documents_sha256": sha(source_path),
        "original_reference_path": str(reference_dir / "reference.json"),
        "frozen_model_code_sha256": lock["frozen_model_code_sha256"],
        "comparison_code_sha256": lock["comparison_code_sha256"],
        "parameters": {"segments_max_pages": 2, "segments_max_chars": 5000,
                       "recovery_context_max_chars": 4000, "supplier_context_max_chars": 5000},
        "predictions_generated": False, "reference_values_changed": False,
        "alias_translation_applied_to_both_sides": True, "selection_by_reference_values": False,
        "duplicate_prediction_policy": "stop_metrics_preserve_emissions", "ranking_inputs_modified": False,
    })
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return {"projection_locked_before_predictions": True, "reference_values_changed": False, "predictions_generated": False}


def project_prediction(ocid, raw, branch, policy, prediction_id, origin):
    return {"opportunity_id": ocid, "subject": raw["subject"], "section_role": raw["section_role"],
            "item": raw.get("item"), "branch": branch_label(branch, policy),
            "fields": {"billed_amount": canonical_money(raw.get("billed_amount")),
                       "currency": raw.get("currency"), "lookback_years": raw.get("lookback_years")},
            "prediction_id": prediction_id, "origin": origin,
            "raw_status": raw.get("status"), "raw_branch": branch, "ranking_eligible": False}


def project_outputs(documents, policy):
    accepted, additional = [], []
    for doc in documents:
        ocid = doc["opportunity_id"]
        for i, variable in enumerate(doc["conditional_variables"]):
            for j, branch in enumerate(variable["branches"]):
                raw = {**branch, "subject": variable["subject"], "section_role": variable["section_role"],
                       "status": variable["status"]}
                accepted.append(project_prediction(ocid, raw, branch["branch_id"], policy,
                    f"{ocid}:accepted:{i}:{j}", {"kind": "accepted_branch", "variable_index": i,
                        "branch_index": j, "page": variable["source_clause_page"],
                        "clause_sha256": variable["source_clause_sha256"]}))
        for family, packet in (("recovery", doc["recovery_candidates"]), ("formulation", doc["formulation_candidates"])):
            for i, raw in enumerate(packet["records"]):
                branch = "supplier_profile_tdr" if raw["section_role"] == "supplier_profile" else "general"
                additional.append(project_prediction(ocid, raw, branch, policy,
                    f"{ocid}:{family}:{i}", {"kind": raw["kind"], "family": family,
                                            "record_index": i, "evidence": raw["evidence"]}))
    return {"accepted_branches": accepted, "additional_candidates": additional}


def comparison_gate(reference, predictions):
    eligible = {d["opportunity_id"] for d in reference["documents"] if d["scope"] == "enterprise_services"}
    all_groups, formal_groups = defaultdict(list), defaultdict(list)
    for row in predictions:
        all_groups[key(row)].append(row)
        if row["opportunity_id"] in eligible:
            formal_groups[key(row)].append(row)
    def collisions(groups):
        return [{"identity": dict(zip(("opportunity_id", "subject", "section_role", "item", "branch"), identity)),
                 "prediction_ids": [r["prediction_id"] for r in rows],
                 "fields": [r["fields"] for r in rows], "origins": [r["origin"] for r in rows]}
                for identity, rows in sorted(groups.items(), key=lambda kv: repr(kv[0])) if len(rows) > 1]
    formal, every = collisions(formal_groups), collisions(all_groups)
    return {"emitted_records": len(predictions), "records_in_formal_scope": sum(len(v) for v in formal_groups.values()),
            "records_outside_formal_scope": len(predictions) - sum(len(v) for v in formal_groups.values()),
            "duplicate_groups_formal": formal, "duplicate_groups_all": every,
            "evaluation_allowed": not formal, "records_removed": 0}


def infer(plan_dir, output):
    if output.exists():
        raise ValueError("Use a fresh prediction directory")
    inputs = {}
    verify_artifact(plan_dir, inputs)
    inputs.update(json.loads((plan_dir / "manifest.json").read_text())["input_sha256"])
    plan = json.loads((plan_dir / "run_plan.json").read_text())
    verify_hashes({plan["original_reference_path"]: plan["original_reference_sha256"],
                   plan["source_documents_path"]: plan["source_documents_sha256"],
                   **plan["frozen_model_code_sha256"], **plan["comparison_code_sha256"]})
    if plan["predictions_generated"] is not False or plan["selection_by_reference_values"] is not False:
        raise ValueError("Pre-inference plan must precede real predictions")
    policy = json.loads((plan_dir / "projection_policy.json").read_text())
    check_policy(policy)
    sources = json.loads(Path(plan["source_documents_path"]).read_text())
    # No reference object is passed to extraction. All 27 PDFs use identical rules.
    documents = []
    for doc in sources:
        validate_source_document(doc)
        pages = doc["pages"]
        result = extract_experience_segments(pages, max_pages=2, max_chars=5000)
        validate_result(result, pages)
        variables = []
        for category in ("qualification_clauses", "evaluation_factor_clauses"):
            for clause in result[category]:
                variable = represent_experience_clause(clause)
                verify_variable(variable, clause, pages)
                variables.append(variable)
        recovery = recover_candidates(pages, result)
        for row in recovery["records"]:
            verify_candidate(row, {p["page"]: p for p in pages})
        formulation = extract_formulations(pages, result["mentions"])
        for row in formulation["records"]:
            verify_record(row, {p["page"]: p for p in pages})
        for warning in formulation["warnings"]:
            verify_ref(warning["evidence"], {p["page"]: p for p in pages})
        documents.append({"opportunity_id": doc["opportunity_id"], "sample_number": doc["sample_number"],
            "source_pdf_sha256": doc["source"]["sha256"], "physical_pages_processed": len(pages),
            "segments": result, "conditional_variables": variables, "recovery_candidates": recovery,
            "formulation_candidates": formulation, "ranking_eligible": False})
    # Read the locked projected reference only after raw inference is finished.
    reference = json.loads((plan_dir / "projected_reference.json").read_text())
    predictions = project_outputs(documents, policy)
    gates = {name: comparison_gate(reference, rows) for name, rows in predictions.items()}
    blocked = any(not gate["evaluation_allowed"] for gate in gates.values())
    # Do not present one stream as a complete comparison while the other is blocked.
    metrics = None if blocked else {name: evaluate(reference, rows, list(FIELDS)) for name, rows in predictions.items()}
    summary = {"documents_processed": len(documents), "selected_cases_retained": len(reference["documents"]),
        "physical_pages_processed": sum(d["physical_pages_processed"] for d in documents),
        "qualification_clauses": sum(len(d["segments"]["qualification_clauses"]) for d in documents),
        "evaluation_factor_clauses": sum(len(d["segments"]["evaluation_factor_clauses"]) for d in documents),
        "accepted_branches": len(predictions["accepted_branches"]),
        "additional_candidates": len(predictions["additional_candidates"]),
        "candidate_kinds": dict(Counter(r["kind"] for d in documents for packet in
            (d["recovery_candidates"], d["formulation_candidates"]) for r in packet["records"])),
        "comparison_status": "stopped_duplicate_prediction_identities" if blocked else "completed_automatic_reference_concordance",
        "predictions_generated": True, "f1_computed": metrics is not None,
        "duplicate_groups_formal": {name: len(gate["duplicate_groups_formal"]) for name, gate in gates.items()},
        "literal_provenance_verified": True, "no_emitted_records_removed": True,
        "reference_values_modified": False, "model_rules_modified": False, "new_ocr_performed": False,
        "ranking_inputs_modified": False, "semantic_accuracy_independently_proven": False,
        "reference_and_adapter_locked_before_inference": True}
    verify_hashes(inputs)
    output.mkdir(parents=True)
    for name, value in (("raw_extractor_outputs.json", documents), ("prediction_streams.json", predictions),
                        ("comparison_gates.json", gates), ("metrics.json", metrics), ("summary.json", summary)):
        dump(output / name, value)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("prepare")
    for name in ("reference", "source", "policy", "output"):
        pre.add_argument("--" + name, type=Path, required=True)
    run = commands.add_parser("infer")
    for name in ("plan", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.reference.resolve(), args.source.resolve(), args.policy.resolve(), args.output.resolve())
    else:
        result = infer(args.plan.resolve(), args.output.resolve())
    print(json.dumps(result))
