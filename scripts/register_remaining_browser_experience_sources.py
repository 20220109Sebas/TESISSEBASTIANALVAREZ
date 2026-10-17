"""Register fixed remaining SEACE batches without changing prior runners or rules."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.register_browser_experience_sources import (
    dump, protected_outputs, validate_observation, verify_hashes,
)
from scripts.fetch_experience_documents import sha
from scripts.fetch_pending_experience_sources import check_code


def select_batch(plan, observations, batch_number):
    if batch_number not in (2, 3):
        raise ValueError("Only fixed remaining batches 2 and 3 are supported")
    first_count = plan["batches"][0]["sources"]
    if [o["opportunity_id"] for o in observations] != [
        s["opportunity_id"] for s in plan["pending_sources"][first_count:]
    ]:
        raise ValueError("Observations must follow the complete remaining queue")
    if len({o["opportunity_id"] for o in observations}) != len(observations):
        raise ValueError("Duplicate observed process")
    start = sum(b["sources"] for b in plan["batches"][:batch_number - 1])
    stop = start + plan["batches"][batch_number - 1]["sources"]
    return (plan["batches"][batch_number - 1], plan["pending_sources"][start:stop],
            observations[start - first_count:stop - first_count])


def register(observations_path, output, batch_number):
    if output.exists():
        raise ValueError("Use a new access directory")
    plan_path = ROOT / "artifacts/experience_substitute_plan_v1/pending_plan.json"
    manifest_path = plan_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_hashes({str(plan_path.parent / p): h for p, h in manifest["output_sha256"].items()})
    plan = json.loads(plan_path.read_text())
    check_code(plan)
    observations = json.loads(observations_path.read_text())["observations"]
    batch, selected, observed = select_batch(plan, observations, batch_number)
    verify_hashes({batch["path"]: batch["sha256"]})
    fixed_batch = json.loads(Path(batch["path"]).read_text())
    if fixed_batch["selected"] != selected:
        raise ValueError("Fixed batch selection differs from the queue")
    limits = fixed_batch["limits"]
    inputs = {str(observations_path): sha(observations_path), str(plan_path): sha(plan_path),
              str(manifest_path): sha(manifest_path), batch["path"]: batch["sha256"]}
    inputs.update(plan["frozen_rule_and_runner_sha256"])
    inputs.update(protected_outputs())
    ready = []
    for o, s in zip(observed, selected):
        inputs[s["current_record_file_path"]] = s["current_record_sha256"]
        verify_hashes({s["current_record_file_path"]: s["current_record_sha256"]})
        record = json.loads(Path(s["current_record_file_path"]).read_text())["records"][0]
        if record["compiledRelease"]["ocid"] != s["opportunity_id"]:
            raise ValueError("Saved record has different OCID")
        validate_observation(o, s, record["compiledRelease"]["tender"], limits)
        row = dict(s, browser_observation=o, retrieval_transport="public_seace_browser_download",
                   original_document_replaced=False, human_annotation=False, http_status=None,
                   historical_content_proven=False, ranking_eligible=False,
                   status=o["status"], absence_of_requirements_proven=False)
        if o["status"] == "browser_download_received":
            path = Path(o["download_path"])
            size = path.stat().st_size
            if size > limits["max_bytes_per_document"]:
                raise ValueError("Downloaded bytes exceed frozen limit")
            with path.open("rb") as stream:
                if stream.read(5) != b"%PDF-":
                    raise ValueError("Downloaded file is not a PDF")
            digest = sha(path)
            inputs[str(path)] = digest
            row.update(status="downloaded_pdf", sha256=digest, bytes=size,
                       file_path=str(output / "pdfs" / (digest + ".pdf")))
        ready.append(row)
    verify_hashes(inputs)
    output.mkdir(parents=True)
    (output / "pdfs").mkdir()
    for row in ready:
        if row["status"] == "downloaded_pdf":
            shutil.copyfile(row["browser_observation"]["download_path"], row["file_path"])
            verify_hashes({row["file_path"]: row["sha256"]})
    dump(output / "access_plan.json", dict(
        limits=limits, pending_sources=selected, fixed_batch_number=batch_number,
        purpose="separate_substitute_development_browser_recovery", ranking_eligible=False,
    ))
    dump(output / "access_results.json", ready)
    dump(output / "manifest.json", {
        "access_plan_path": str(output / "access_plan.json"),
        "access_plan_sha256": sha(output / "access_plan.json"),
        "input_sha256": inputs,
        "code_sha256": {str(Path(__file__)): sha(__file__),
                        str(ROOT / "scripts/register_browser_experience_sources.py"):
                            sha(ROOT / "scripts/register_browser_experience_sources.py")},
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    })
    check_code(plan)
    verify_hashes(inputs)
    return {"batch_number": batch_number, "fichas_verified": len(ready),
            "pdfs_saved": sum(r["status"] == "downloaded_pdf" for r in ready)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-number", type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    print(json.dumps(register(args.observations.resolve(), args.output.resolve(), args.batch_number)))
