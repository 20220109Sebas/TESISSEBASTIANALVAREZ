"""Import the explicitly authorized Pasco retry without replacing prior access evidence."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.register_browser_experience_sources import dump, validate_observation, verify_hashes, protected_outputs
from scripts.fetch_experience_documents import sha
from scripts.fetch_pending_experience_sources import check_code

PASCO = "ocds-dgv273-seacev3-1116792"


def register(observations_path, output):
    if output.exists():
        raise ValueError("Use a new access directory")
    plan_path = ROOT / "artifacts/experience_substitute_plan_v1/pending_plan.json"
    manifest_path = plan_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_hashes({str(plan_path.parent / p): h for p, h in manifest["output_sha256"].items()})
    plan = json.loads(plan_path.read_text())
    check_code(plan)
    selected = [s for s in plan["pending_sources"] if s["opportunity_id"] == PASCO]
    observations = json.loads(observations_path.read_text())["observations"]
    if len(selected) != 1 or len(observations) != 1 or observations[0]["opportunity_id"] != PASCO:
        raise ValueError("Only the selected Pasco document may be retried here")
    source, observation = selected[0], observations[0]
    batch = plan["batches"][0]
    verify_hashes({batch["path"]: batch["sha256"]})
    limits = json.loads(Path(batch["path"]).read_text())["limits"]
    inputs = {str(plan_path): sha(plan_path), str(manifest_path): sha(manifest_path),
              str(observations_path): sha(observations_path), batch["path"]: batch["sha256"],
              source["current_record_file_path"]: source["current_record_sha256"]}
    inputs.update(plan["frozen_rule_and_runner_sha256"])
    inputs.update(protected_outputs())
    prior = ROOT / "artifacts/experience_substitute_browser_access_v1"
    prior_manifest = json.loads((prior / "manifest.json").read_text())
    inputs.update({str(prior / p): h for p, h in prior_manifest["output_sha256"].items()})
    inputs[str(prior / "manifest.json")] = sha(prior / "manifest.json")
    verify_hashes(inputs)
    record = json.loads(Path(source["current_record_file_path"]).read_text())["records"][0]["compiledRelease"]
    if record["ocid"] != PASCO:
        raise ValueError("Different saved process")
    validate_observation(observation, source, record["tender"], limits)
    path = Path(observation["download_path"])
    if observation["status"] != "browser_download_received" or path.stat().st_size > limits["max_bytes_per_document"]:
        raise ValueError("Missing or oversized retry download")
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("Not a PDF")
    digest = sha(path)
    inputs[str(path)] = digest
    row = dict(source, browser_observation=observation, status="downloaded_pdf", sha256=digest,
               bytes=path.stat().st_size, file_path=str(output / "pdfs" / (digest + ".pdf")),
               retrieval_transport="public_seace_browser_download_authorized_retry",
               original_document_replaced=False, human_annotation=False, http_status=None,
               historical_content_proven=False, ranking_eligible=False, absence_of_requirements_proven=False)
    verify_hashes(inputs)
    output.mkdir(parents=True)
    (output / "pdfs").mkdir()
    shutil.copyfile(path, row["file_path"])
    verify_hashes({row["file_path"]: digest})
    dump(output / "access_plan.json", dict(limits=limits, pending_sources=selected,
         purpose="authorized_same_document_retry_separate_development", ranking_eligible=False))
    dump(output / "access_results.json", [row])
    dump(output / "manifest.json", {"access_plan_path": str(output / "access_plan.json"),
        "access_plan_sha256": sha(output / "access_plan.json"), "input_sha256": inputs,
        "code_sha256": {str(Path(__file__)): sha(__file__),
                        str(ROOT / "scripts/register_browser_experience_sources.py"): sha(ROOT / "scripts/register_browser_experience_sources.py")},
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    verify_hashes(inputs)
    check_code(plan)
    return {"pdfs_saved": 1, "bytes": row["bytes"], "sha256": digest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(register(args.observations.resolve(), args.output.resolve())))
