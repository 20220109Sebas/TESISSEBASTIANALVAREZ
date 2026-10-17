"""Check saved-text replays and frozen files, without running extraction models."""
import argparse
import json
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ("experience_holdout_mixed_text_v1", "experience_holdout_mixed_text_v1_replay"),
    ("experience_holdout_sparse_text_v1", "experience_holdout_sparse_text_v1_replay"),
    ("experience_holdout_source_view_v1", "experience_holdout_source_view_v1_replay"),
    ("experience_holdout_reference_draft_v2", "experience_holdout_reference_draft_v2_replay"),
    ("experience_holdout_evaluation_preflight_v1", "experience_holdout_evaluation_preflight_v1_replay"),
    ("experience_holdout_source_reading_v1", "experience_holdout_source_reading_v1_replay"),
    ("experience_holdout_scope_observations_v2", "experience_holdout_scope_observations_v2_replay"),
    ("experience_holdout_reviewer_verification_v1", "experience_holdout_reviewer_verification_v1_replay"),
    ("experience_holdout_review_progress_v1", "experience_holdout_review_progress_v1_replay"),
    ("experience_holdout_review_progress_v2", "experience_holdout_review_progress_v2_replay"),
]


def audit(output):
    if output.exists():
        raise ValueError("Use a new verification directory")
    protected = protected_outputs()
    inputs = dict(protected)
    checked = []
    for original_name, replay_name in PAIRS:
        original, replay = ROOT / "artifacts" / original_name, ROOT / "artifacts" / replay_name
        for folder in (original, replay):
            verify_artifact(folder, inputs)
        expected = json.loads((original / "manifest.json").read_text())["output_sha256"]
        actual = json.loads((replay / "manifest.json").read_text())["output_sha256"]
        if expected != actual:
            raise ValueError(f"Nonidentical logical outputs: {original_name}")
        checked.append({"original": str(original), "replay": str(replay),
            "logical_output_files_identical": len(expected), "all_hashes_verified": True,
            "manifest_bytes_compared": False,
            "reason_manifest_bytes_not_compared": "Output directories differ; manifests preserve those absolute paths"})
    lock_records = []
    for path, key in (
        (ROOT / "artifacts/experience_substitute_plan_v1/pending_plan.json", "frozen_rule_and_runner_sha256"),
        (ROOT / "artifacts/experience_holdout_models_lock_v1/model_lock.json", "original_code_sha256"),
        (ROOT / "artifacts/experience_holdout_evaluation_preflight_v1/evaluation_lock.json", "comparison_code_sha256"),
    ):
        expected = json.loads(path.read_text())[key]
        verify_hashes(expected)
        inputs.update(expected)
        inputs[str(path)] = sha(path)
        lock_records.append({"lock": str(path), "files_verified": len(expected)})
    verify_hashes(inputs)
    summary = {"saved_text_replay_pairs_verified": len(checked),
        "logical_output_files_identical": sum(p["logical_output_files_identical"] for p in checked),
        "protected_outputs_verified": len(protected), "code_locks": lock_records,
        "ocr_rerun": False, "predictions_generated": False, "model_accuracy_estimated": False,
        "ranking_inputs_modified": False,
        "interpretation": "Reproducibility and integrity only, not proof of source semantics or model quality"}
    output.mkdir(parents=True)
    dump(output / "checks.json", checked)
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output.resolve())))
