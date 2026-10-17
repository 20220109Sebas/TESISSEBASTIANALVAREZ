"""Snapshot a text-input model candidate, not a complete new evaluation."""
import argparse
import importlib
import json
import platform
import shutil
import sys
import unicodedata
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/document_experience_candidate_contract_v1.json"
POLICY = ROOT / "docs/document_experience_occurrence_policy_v1.md"
MODEL_FILES = (
    "src/tesis_recommender/__init__.py", "src/tesis_recommender/experience.py",
    "src/tesis_recommender/experience_conditions.py", "src/tesis_recommender/experience_segments.py",
    "src/tesis_recommender/document_experience_dev_v1.py", "src/tesis_recommender/document_experience_dev_v2.py",
    "src/tesis_recommender/document_experience_units_dev_v1.py", "src/tesis_recommender/document_experience_candidate_v1.py",
)
TEST_FILES = (
    "tests/test_document_experience_dev_v1.py", "tests/test_document_experience_dev_v2.py",
    "tests/test_document_experience_units_dev_v1.py", "tests/test_document_experience_candidate_v1.py",
)


def runtime_identity():
    return {"python_implementation": platform.python_implementation(), "python_version": sys.version,
            "unicode_database_version": unicodedata.unidata_version,
            "external_model_runtime_dependencies": [], "source_preparation_runtime_not_frozen": True}


def check_contract(contract):
    from tesis_recommender.document_experience_candidate_v1 import MAX_BYTES, MAX_DOCUMENTS, MAX_PAGES, METHODS
    fixed = json.loads(CONTRACT.read_text())
    if contract != fixed:
        raise ValueError("Lock contract differs from the fixed source contract")
    inputs = contract["input"]
    if (inputs["max_documents"] != MAX_DOCUMENTS or inputs["max_physical_pages_per_document"] != MAX_PAGES
            or inputs["max_pdf_bytes"] != MAX_BYTES or set(inputs["allowed_text_methods"]) != METHODS
            or contract["extraction"]["max_clause_pages"] != 2
            or contract["extraction"]["max_clause_characters"] != 6000):
        raise ValueError("Input/extraction contract differs from executable constants")


def freeze(output):
    if output.exists():
        raise ValueError("Use a new candidate freeze directory")
    inputs = dict(protected_outputs())
    for name in ("document_experience_units_development_v1", "document_experience_validation_readiness_v1"):
        directory = ROOT / "artifacts" / name
        verify_artifact(directory, inputs)
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    contract = json.loads(CONTRACT.read_text())
    check_contract(contract)
    model = {str(ROOT / rel): sha(ROOT / rel) for rel in MODEL_FILES}
    tests = {str(ROOT / rel): sha(ROOT / rel) for rel in TEST_FILES}
    code = (Path(__file__).resolve(), ROOT / "scripts/verify_document_experience_candidate_v1.py")
    inputs.update(model)
    inputs.update(tests)
    inputs.update({str(path): sha(path) for path in (*code, CONTRACT, POLICY)})
    compare = {str(ROOT / rel): sha(ROOT / rel) for rel in (
        "src/tesis_recommender/reference_evaluation.py", "tests/test_reference_evaluation.py")}
    inputs.update(compare)
    verify_hashes(inputs)
    output.mkdir(parents=True)
    snapshot_paths = sorted({*model, *tests, *compare, *(str(p) for p in (*code, CONTRACT, POLICY))})
    for name in snapshot_paths:
        path = Path(name)
        destination = output / "code_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    dump(output / "contract.json", contract)
    dump(output / "runtime.json", runtime_identity())
    dump(output / "model_lock.json", {"version": "document_experience_candidate_lock_v1",
        "model_code_sha256": model, "test_code_sha256": tests, "comparison_engine_sha256": compare,
        "contract_path": str(CONTRACT), "contract_sha256": sha(CONTRACT),
        "unit_policy_path": str(POLICY), "unit_policy_sha256": sha(POLICY),
        "runtime": runtime_identity(), "model_interface_locked": True,
        "sampling_design_locked": False, "source_acquisition_and_ocr_locked": False,
        "new_reference_locked": False, "new_sample_selected": False, "predictions_on_new_sample_generated": False,
        "ready_for_new_evaluation": False,
        "change_policy": "new_version_required_no_retuning_locked_new_test",
        "comparison_adapter_and_run_gates_pending": True})
    summary = {"model_code_files_frozen": len(model), "model_test_files_frozen": len(tests),
        "comparison_engine_files_pinned": len(compare), "model_interface_locked": True,
        "ready_for_new_evaluation": False, "new_sample_selected": False,
        "source_acquisition_or_ocr_executed": False, "metrics_computed": False,
        "ranking_inputs_modified": False, "stage": "text_input_model_candidate_frozen_not_end_to_end_protocol"}
    dump(output / "summary.json", summary)
    verify_hashes(inputs)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return summary


def validate_lock(directory):
    inputs = {}
    verify_artifact(directory, inputs)
    lock = json.loads((directory / "model_lock.json").read_text())
    if (lock.get("model_interface_locked") is not True or lock.get("ready_for_new_evaluation") is not False
            or lock.get("new_sample_selected") is not False):
        raise ValueError("Candidate lock cannot be relabelled as a completed evaluation")
    verify_hashes(lock["model_code_sha256"])
    verify_hashes(lock["test_code_sha256"])
    verify_hashes(lock["comparison_engine_sha256"])
    verify_hashes({lock["contract_path"]: lock["contract_sha256"], lock["unit_policy_path"]: lock["unit_policy_sha256"]})
    if lock["runtime"] != runtime_identity() or json.loads((directory / "runtime.json").read_text()) != lock["runtime"]:
        raise ValueError("Runtime changed; do not silently reuse the candidate lock")
    check_contract(json.loads((directory / "contract.json").read_text()))
    # Check import origins as well as disk-file hashes.
    for rel in MODEL_FILES:
        if rel.startswith("src/tesis_recommender/") and not rel.endswith("__init__.py"):
            name = "tesis_recommender." + Path(rel).stem
            module = importlib.import_module(name)
            if Path(module.__file__).resolve() != (ROOT / rel).resolve():
                raise ValueError("Imported model code came from a different path")
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output.resolve())))
