"""Snapshot extraction code before generating predictions on the new sample."""
import argparse
import json
import shutil
from pathlib import Path
from scripts.register_browser_experience_sources import dump, sha, verify_hashes, protected_outputs

ROOT = Path(__file__).resolve().parents[1]


def freeze(plan_dir, output):
    if output.exists():
        raise ValueError("Use a new freeze directory")
    plan_manifest = json.loads((plan_dir / "manifest.json").read_text())
    inputs = dict(plan_manifest["input_sha256"])
    inputs.update({str(plan_dir / p): h for p, h in plan_manifest["output_sha256"].items()})
    inputs.update(protected_outputs())
    code = sorted((ROOT / "src/tesis_recommender").glob("experience*.py"))
    code += [ROOT / "scripts" / name for name in (
        "process_completed_experience_text.py", "recover_unresolved_experience.py",
        "compare_experience_segments.py", "represent_experience_conditions.py",
        "extend_experience_formulations.py")]
    code_hashes = {str(p): sha(p) for p in code}
    verify_hashes(inputs)
    output.mkdir(parents=True)
    for path in code:
        target = output / "code_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    dump(output / "model_lock.json", {"sample_selection_sha256": sha(plan_dir / "selection.json"),
        "original_code_sha256": code_hashes,
        "predictions_generated": False, "no_tuning_on_holdout": True,
        "ranking_inputs_modified": False, "reference_status": "not_complete",
        "comparison": "frozen_accepted_clauses_and_separate_recovery_candidates_reported_separately",
        "future_change_policy": "any_code_change_requires_new_development_version_not_refitting_this_test"})
    verify_hashes(inputs)
    verify_hashes(code_hashes)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return {"extraction_code_files_frozen": len(code), "predictions_generated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.plan.resolve(), args.output.resolve())))
