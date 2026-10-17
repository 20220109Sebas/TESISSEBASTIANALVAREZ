"""Fix source preparation and sampling design WITHOUT selecting new documents."""
import argparse
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.freeze_document_experience_candidate_v1 import validate_lock
from scripts.prepare_document_experience_validation_readiness_v1 import BASE_FILES, collect_exclusions, verify_listed_file
from scripts.register_browser_experience_sources import dump, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PYTHON = Path(os.environ.get("TESIS_SOURCE_PYTHON", sys.executable)).resolve()
VISION_BINARY = ROOT / "tmp/vision_ocr_build/vision-pdf-ocr"
CODE_FILES = (
    "src/tesis_recommender/document_source_preparation_v1.py", "src/tesis_recommender/document_sampling_v1.py",
    "src/tesis_recommender/annex_sources.py", "scripts/fetch_experience_documents.py",
    "scripts/prepare_document_sources_v1.py", "scripts/freeze_document_source_design_v1.py",
    "scripts/vision_pdf_ocr.swift", "tests/test_document_source_design_v1.py",
    "docs/document_source_design_v1.json", "docs/document_sampling_design_v1.json",
    "scripts/audit_holdout_evaluation_readiness.py", "scripts/register_browser_experience_sources.py",
    "scripts/freeze_document_experience_candidate_v1.py", "scripts/prepare_document_experience_validation_readiness_v1.py",
)


def source_runtime():
    result = subprocess.run([str(SOURCE_PYTHON), str(ROOT / "scripts/prepare_document_sources_v1.py"), "runtime"],
                            check=True, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def check_design_constants():
    from tesis_recommender.document_sampling_v1 import SAMPLE_SIZE, SEED, ROLES
    from tesis_recommender.document_source_preparation_v1 import (
        MAX_BYTES, MAX_PAGES, MAX_PAGE_CHARS, MAX_DOC_CHARS, OCR_WORD_THRESHOLD, IMAGE_PIXELS, FORM_DEPTH,
    )
    source = json.loads((ROOT / "docs/document_source_design_v1.json").read_text())
    sampling = json.loads((ROOT / "docs/document_sampling_design_v1.json").read_text())
    expected = (source["access"]["max_bytes_per_document"], source["native"]["max_physical_pages"],
                source["native"]["max_characters_per_page"], source["native"]["max_characters_per_document"],
                source["ocr"]["image_form_max_depth"])
    if expected != (MAX_BYTES, MAX_PAGES, MAX_PAGE_CHARS, MAX_DOC_CHARS, FORM_DEPTH):
        raise ValueError("Source design constants differ from executable code")
    if OCR_WORD_THRESHOLD != 50 or IMAGE_PIXELS != 600:
        raise ValueError("OCR screen differs")
    if sampling["seed"] != SEED or sampling["sample_size"] != SAMPLE_SIZE or tuple(sampling["source_priority"]) != ROLES:
        raise ValueError("Sampling design differs from executable code")


def freeze(output):
    if output.exists():
        raise ValueError("Use a new source-design directory")
    check_design_constants()
    candidate = ROOT / "artifacts/document_experience_candidate_lock_v1"
    validate_lock(candidate)
    inputs = {}
    verify_artifact(candidate, inputs)
    inputs.update(json.loads((candidate / "manifest.json").read_text())["input_sha256"])
    paths = sorted({*(ROOT / p for p in BASE_FILES), *(ROOT / "artifacts").glob("experience*/**/access_results.json")})
    payloads = []
    for path in paths:
        verify_listed_file(path, inputs)
        payloads.append((str(path.relative_to(ROOT)), json.loads(path.read_text())))
    exclusions = collect_exclusions(payloads)
    runtime = source_runtime()
    inputs.update(runtime["pypdf_code_sha256"])
    inputs[runtime["python_executable"]] = runtime["python_sha256"]
    inputs[str(VISION_BINARY)] = sha(VISION_BINARY)
    inputs.update({str(ROOT / p): sha(ROOT / p) for p in CODE_FILES})
    candidate_data = ROOT / "data/processed/contrast_candidates_v1/annual_v1/test_opportunities.csv"
    availability = candidate_data.parent / "opportunity_availability.csv"
    expected = json.loads((ROOT / "artifacts/experiment_annual_v1/manifest.json").read_text())["input_sha256"][str(candidate_data.relative_to(ROOT))]
    verify_hashes({str(candidate_data): expected})
    inputs.update({str(candidate_data): expected, str(availability): sha(availability)})
    verify_hashes(inputs)
    output.mkdir(parents=True)
    for rel in CODE_FILES:
        target = output / "code_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    dump(output / "exclusion_registry.json", exclusions)
    lock = {"version": "document_source_design_lock_v1", "code_sha256": {str(ROOT / p): sha(ROOT / p) for p in CODE_FILES},
        "source_python": str(SOURCE_PYTHON), "source_runtime": runtime, "vision_binary": str(VISION_BINARY),
        "vision_binary_sha256": sha(VISION_BINARY), "known_pdf_sha256": exclusions["known_pdf_sha256"],
        "exclusion_source_files": [str(p) for p in paths], "model_candidate_manifest_sha256": sha(candidate / "manifest.json"),
        "source_preparation_code_locked": True, "sampling_design_locked": True,
        "new_sample_selected": False, "new_reference_locked": False, "comparison_adapter_locked": False,
        "ready_to_select_new_evaluation_sample": False, "ready_to_compute_metrics": False,
        "interpretation": "source_and_sampling_design_fixed_adapter_and_final_reference_gates_still_pending"}
    dump(output / "source_design_lock.json", lock)
    summary = {"source_code_and_design_files_frozen": len(CODE_FILES), "excluded_ocids": len(exclusions["excluded_ocids"]),
        "known_pdf_hashes": len(exclusions["known_pdf_sha256"]), "pypdf_version": runtime["pypdf_version"],
        "sampling_design_locked": True, "source_preparation_code_locked": True,
        "new_sample_selected": False, "new_pdf_downloads": 0, "new_ocr_executed": False,
        "new_reference_created": False, "model_inference_performed": False, "metrics_computed": False,
        "ready_to_select_new_evaluation_sample": False, "ranking_inputs_modified": False}
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return summary


def validate_design(directory, *, before_selection=False):
    inputs = {}
    verify_artifact(directory, inputs)
    lock = json.loads((directory / "source_design_lock.json").read_text())
    verify_hashes(lock["code_sha256"])
    verify_hashes({lock["vision_binary"]: lock["vision_binary_sha256"]})
    if source_runtime() != lock["source_runtime"]:
        raise ValueError("Source runtime changed; use a new design version")
    current_paths = sorted({*(ROOT / p for p in BASE_FILES), *(ROOT / "artifacts").glob("experience*/**/access_results.json")})
    if before_selection and [str(p) for p in current_paths] != lock["exclusion_source_files"]:
        raise ValueError("Observed source inventory changed before selection; refresh exclusions in a new lock")
    if (lock["new_sample_selected"] is not False or lock["ready_to_select_new_evaluation_sample"] is not False
            or lock["ready_to_compute_metrics"] is not False):
        raise ValueError("Preparation lock cannot be relabelled as full evaluation readiness")
    check_design_constants()
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output.resolve())))
