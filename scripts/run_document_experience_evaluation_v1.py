"""Freeze new comparison gates; lock source-only reference before inference.

These commands do not select a sample or acquire/inspect PDFs. New artifacts are
append-only. Frozen historical locks, predictions and comparators stay intact.
"""
import argparse
import importlib
import json
import shutil
from collections import Counter
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.freeze_document_experience_candidate_v1 import validate_lock as validate_candidate
from scripts.freeze_document_source_design_v1 import validate_design as validate_source_design
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes
from tesis_recommender.document_experience_candidate_v1 import predict_documents
from tesis_recommender.document_experience_evaluation_v1 import (
    FIELDS, ROLES, adapt_predictions, build_reference, compare, source_population,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/document_experience_candidate_lock_v1"
SOURCE_DESIGN = ROOT / "artifacts/document_source_design_lock_v1"
FILES = ("src/tesis_recommender/document_experience_evaluation_v1.py",
         "scripts/run_document_experience_evaluation_v1.py",
         "tests/test_document_experience_evaluation_v1.py",
         "docs/document_experience_evaluation_design_v1.json")


def checked(directory, inputs):
    verify_artifact(directory, inputs)
    inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])


def save(output, files, inputs):
    if output.exists():
        raise ValueError("Use a fresh artifact directory; no overwrite or reference relock")
    verify_hashes(inputs)
    output.mkdir(parents=True)
    for name, value in files.items():
        dump(output / name, value)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return files["summary.json"]


def freeze(output):
    if output.exists():
        raise ValueError("Use a fresh evaluation-design directory")
    validate_candidate(CANDIDATE)
    validate_source_design(SOURCE_DESIGN, before_selection=True)
    inputs = dict(protected_outputs())
    for directory in (CANDIDATE, SOURCE_DESIGN):
        checked(directory, inputs)
    code = {str(ROOT / p): sha(ROOT / p) for p in FILES}
    inputs.update(code)
    protocol = json.loads((ROOT / FILES[-1]).read_text())
    if protocol["fields"] != list(FIELDS) or set(protocol["streams_reported_separately"]) != ROLES or protocol["sample_size"] != 50:
        raise ValueError("Comparison code and protocol differ")
    lock = {"version": "document_experience_evaluation_design_lock_v1", "code_sha256": code,
            "candidate_lock_path": str(CANDIDATE), "candidate_manifest_sha256": sha(CANDIDATE / "manifest.json"),
            "source_design_path": str(SOURCE_DESIGN), "source_design_manifest_sha256": sha(SOURCE_DESIGN / "manifest.json"),
            "comparison_adapter_locked": True, "reference_first_gates_locked": True,
            "new_sample_selected": False, "ready_to_select_new_evaluation_sample": False,
            "ready_to_compute_metrics": False,
            "pending": "integrated_metadata_selection_runner_and_preselection_end_to_end_audit"}
    summary = {"new_code_and_protocol_files_frozen": len(FILES), "comparison_adapter_locked": True,
               "reference_first_gates_locked": True, "new_sample_selected": False, "new_pdf_downloads": 0,
               "new_reference_created": False, "model_inference_on_new_sample": False, "metrics_computed": False,
               "ready_to_select_new_evaluation_sample": False, "ranking_inputs_modified": False}
    verify_hashes(inputs)
    output.mkdir(parents=True)
    for rel in FILES:
        target = output / "code_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    dump(output / "evaluation_design_lock.json", lock)
    dump(output / "protocol.json", protocol)
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return summary


def validate_design(directory):
    inputs = {}
    checked(directory, inputs)
    lock = json.loads((directory / "evaluation_design_lock.json").read_text())
    verify_hashes(lock["code_sha256"])
    for flag in ("comparison_adapter_locked", "reference_first_gates_locked"):
        if lock.get(flag) is not True:
            raise ValueError("Evaluation code gates are not fixed")
    for flag in ("new_sample_selected", "ready_to_select_new_evaluation_sample", "ready_to_compute_metrics"):
        if lock.get(flag) is not False:
            raise ValueError("Preparation cannot be relabelled as a completed/newly selected evaluation")
    candidate, source = Path(lock["candidate_lock_path"]), Path(lock["source_design_path"])
    verify_hashes({str(candidate / "manifest.json"): lock["candidate_manifest_sha256"],
                   str(source / "manifest.json"): lock["source_design_manifest_sha256"]})
    validate_candidate(candidate)
    validate_source_design(source)
    for name in ("document_experience_evaluation_v1", "reference_evaluation"):
        module = importlib.import_module("tesis_recommender." + name)
        if Path(module.__file__).resolve() != (ROOT / "src/tesis_recommender" / (name + ".py")).resolve():
            raise ValueError("Comparison code imported from a different path")
    if json.loads((directory / "protocol.json").read_text()) != json.loads((ROOT / FILES[-1]).read_text()):
        raise ValueError("Comparison protocol changed")
    return lock


def load_sources(source_dir, inputs, design_dir):
    checked(source_dir, inputs)
    design = json.loads((design_dir / "evaluation_design_lock.json").read_text())
    source_design = Path(design["source_design_path"])
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    if source_manifest["input_sha256"].get(str(source_design / "manifest.json")) != sha(source_design / "manifest.json"):
        raise ValueError("Source view lacks the frozen preparation-design prerequisite")
    documents = json.loads((source_dir / "source_documents.json").read_text())
    ledger = json.loads((source_dir / "coverage_ledger.json").read_text())
    source_population(documents, ledger)
    registry = json.loads((source_design / "exclusion_registry.json").read_text())
    if {r["opportunity_id"] for r in ledger} & set(registry["excluded_ocids"]):
        raise ValueError("Previously observed case cannot become a new validation case")
    if {d["source"]["sha256"] for d in documents} & set(registry["known_pdf_sha256"]):
        raise ValueError("Previously observed PDF cannot become an unseen source")
    summary = json.loads((source_dir / "summary.json").read_text())
    if summary.get("predictions_generated") is not False or summary.get("reference_created") is not False:
        raise ValueError("Use a preserved source-only combined view")
    return documents, ledger


def lock_reference(design_dir, source_dir, draft_path, output):
    validate_design(design_dir)
    inputs = dict(protected_outputs())
    checked(design_dir, inputs)
    documents, ledger = load_sources(source_dir, inputs, design_dir)
    inputs[str(draft_path)] = sha(draft_path)
    draft = json.loads(draft_path.read_text())
    locked = build_reference(draft, documents, ledger)
    inputs.update(locked["visual_evidence_files"])
    summary = {"stage": "automatic_reference_locked_before_runner_inference", "selected_cases": 50,
        "source_documents": len(documents), "scope_counts": dict(Counter(d["scope"] for d in locked["reference"]["documents"])),
        "source_dir": str(source_dir), "source_manifest_sha256": sha(source_dir / "manifest.json"),
        "evaluation_design_dir": str(design_dir), "evaluation_design_manifest_sha256": sha(design_dir / "manifest.json"),
        "predictions_generated": False, "metrics_computed": False, "independent_or_expert_review": False,
        "ranking_inputs_modified": False}
    return save(output, {"locked_reference.json": locked, "reference_input.json": draft, "summary.json": summary}, inputs)


def checked_reference(reference_dir, design_dir, source_dir, inputs):
    checked(reference_dir, inputs)
    summary = json.loads((reference_dir / "summary.json").read_text())
    if (summary["source_dir"] != str(source_dir) or summary["evaluation_design_dir"] != str(design_dir)
            or summary["source_manifest_sha256"] != sha(source_dir / "manifest.json")
            or summary["evaluation_design_manifest_sha256"] != sha(design_dir / "manifest.json")
            or summary["predictions_generated"] is not False):
        raise ValueError("Reference belongs to different sources/design or is not pre-inference")
    binding = {"reference_manifest_sha256": sha(reference_dir / "manifest.json"),
               "reference_sha256": sha(reference_dir / "locked_reference.json"),
               "source_manifest_sha256": sha(source_dir / "manifest.json"),
               "evaluation_design_manifest_sha256": sha(design_dir / "manifest.json")}
    return binding


def infer(design_dir, source_dir, reference_dir, output):
    validate_design(design_dir)
    inputs = dict(protected_outputs())
    checked(design_dir, inputs)
    documents, ledger = load_sources(source_dir, inputs, design_dir)
    # Check reference artifact hashes/summary only. No reference values are loaded
    # into this inference runner or passed to the fixed candidate interface.
    binding = checked_reference(reference_dir, design_dir, source_dir, inputs)
    if not documents:
        raise ValueError("No complete source documents: report coverage, not model metrics")
    candidate = predict_documents(documents)
    summary = {"stage": "fixed_candidate_inference_after_reference_lock", "binding": binding,
        "source_documents": len(documents), "selected_cases": len(ledger),
        "original_occurrences": len(candidate["occurrence_projection"]["occurrences"]),
        "identity_gate_passed": candidate["occurrence_projection"]["identity_gate"]["passed"],
        "reference_values_passed_to_model": False, "metrics_computed": False, "ranking_inputs_modified": False}
    return save(output, {"candidate_predictions.json": candidate, "summary.json": summary}, inputs)


def evaluate_run(design_dir, source_dir, reference_dir, inference_dir, output):
    validate_design(design_dir)
    inputs = dict(protected_outputs())
    checked(design_dir, inputs)
    documents, ledger = load_sources(source_dir, inputs, design_dir)
    binding = checked_reference(reference_dir, design_dir, source_dir, inputs)
    checked(inference_dir, inputs)
    inference_summary = json.loads((inference_dir / "summary.json").read_text())
    inference_manifest = json.loads((inference_dir / "manifest.json").read_text())
    # A self-reported bool/binding alone is insufficient: the reference itself
    # must be a hashed prerequisite recorded by inference.
    for path in (reference_dir / "manifest.json", reference_dir / "locked_reference.json",
                 source_dir / "manifest.json", design_dir / "manifest.json"):
        if inference_manifest["input_sha256"].get(str(path)) != sha(path):
            raise ValueError("Inference lacks the reference-first artifact prerequisite")
    if inference_summary.get("reference_values_passed_to_model") is not False:
        raise ValueError("Inference may not consult reference values")
    locked = json.loads((reference_dir / "locked_reference.json").read_text())
    rebuilt = build_reference(json.loads((reference_dir / "reference_input.json").read_text()), documents, ledger)
    if rebuilt != locked:
        raise ValueError("Locked reference differs from its archived source-only readings")
    candidate = json.loads((inference_dir / "candidate_predictions.json").read_text())
    result = compare(locked, candidate, documents, ledger, inference_summary["binding"], binding)
    projection = candidate["occurrence_projection"]
    summary = {"stage": result["stage"], "selected_cases": 50, "complete_source_documents": len(documents),
        "source_status_counts": dict(Counter(r["source_status"] for r in ledger)),
        "reference_scope_counts": dict(Counter(d["scope"] for d in locked["reference"]["documents"])),
        "original_occurrences": len(projection["occurrences"]), "candidate_units": len(projection["candidate_units"]),
        "blocked_units": len(projection["identity_gate"]["blocked_unit_ids"]),
        "null_fields_in_original_occurrences": {f: sum(o["record"]["fields"][f] is None for o in projection["occurrences"]) for f in FIELDS},
        "unresolved_ocr_pages": sum(len(c["unresolved_ocr_pages"]) for c in candidate["coverage"]),
        "metrics_computed": result["metrics_computed"], "expert_accuracy_proven": False,
        "post_inference_identity_resolution_used": False, "ranking_inputs_modified": False}
    return save(output, {"comparison.json": result, "summary.json": summary}, inputs)


def regression(design_dir, output):
    """Re-use known archived predictions to test the adapter, never new F1."""
    validate_design(design_dir)
    inputs = dict(protected_outputs())
    checked(design_dir, inputs)
    sources = ROOT / "artifacts/experience_holdout_source_view_v1"
    predictions = ROOT / "artifacts/document_experience_candidate_regression_v1"
    for directory in (sources, predictions):
        checked(directory, inputs)
    documents = json.loads((sources / "source_documents.json").read_text())
    candidate = json.loads((predictions / "candidate_outputs.json").read_text())
    adapted = adapt_predictions(candidate, documents)
    summary = {"stage": "observed_source_adapter_regression_only", "documents": len(documents),
        "original_occurrences": adapted["original_occurrences"], "candidate_units": adapted["candidate_units"],
        "blocked_units": len(adapted["blocked_unit_ids"]), "adapter_stage": adapted["stage"],
        "all_documents_previously_inspected": True, "new_sample_selected": False,
        "new_sample_inference_performed": False, "metrics_computed": False,
        "old_case15_resolution_generalized": False, "ranking_inputs_modified": False}
    return save(output, {"adapter_regression.json": adapted, "summary.json": summary}, inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "regression", "lock-reference", "infer", "evaluate"))
    for flag in ("design", "source", "draft", "reference", "inference", "output"):
        parser.add_argument("--" + flag, type=Path)
    args = parser.parse_args()
    required = {"freeze": ("output",), "regression": ("design", "output"), "lock-reference": ("design", "source", "draft", "output"),
                "infer": ("design", "source", "reference", "output"),
                "evaluate": ("design", "source", "reference", "inference", "output")}[args.command]
    if any(getattr(args, flag) is None for flag in required):
        parser.error("Required: " + ", ".join("--" + flag for flag in required))
    paths = [getattr(args, flag).resolve() for flag in required]
    function = {"freeze": freeze, "regression": regression, "lock-reference": lock_reference, "infer": infer, "evaluate": evaluate_run}[args.command]
    print(json.dumps(function(*paths), ensure_ascii=False))
