"""Freeze the next unattempted five documents from the original 50-opportunity queue."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tesis_recommender.annex_queue import next_annex_batch
from tesis_recommender.annex_sources import safe_annex_url


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(queue_dir, access_dirs, output):
    if output.exists():
        raise ValueError("Use a new plan directory")
    inputs = {}
    def checked(directory, name):
        manifest = json.loads((directory / "manifest.json").read_text())
        path = directory / name
        if sha(path) != manifest["output_sha256"][name]:
            raise ValueError("Queue or access artifact changed")
        inputs[str(path)] = sha(path)
        inputs[str(directory / "manifest.json")] = sha(directory / "manifest.json")
        return json.loads(path.read_text())
    queue = checked(queue_dir, "queue.json")
    accesses = [r for directory in access_dirs for r in checked(directory, "access_results.json")]
    for row in accesses:
        if row["status"] == "downloaded_pdf":
            path = Path(row["file_path"])
            if sha(path) != row["sha256"]:
                raise ValueError("Cached PDF changed")
            inputs[str(path)] = sha(path)
    selection = next_annex_batch(queue, accesses)
    if any(not safe_annex_url(d["url"]) or d.get("temporal_metadata_status") != "declared_at_or_before_initial" for d in selection["selected"]):
        raise ValueError("Unexpected URL or temporal metadata in fixed queue")
    plan = {"purpose": "next_unattempted_bounded_document_batch_development_only",
            "selection": "fixed_queue_order_without_outcomes_no_implicit_retries",
            "limits": {"documents": 5, "max_bytes_per_document": 10 * 1024 * 1024,
                       "timeout_seconds": 30, "max_pages_per_document": 120, "workers": 1, "ocr": False},
            "allowed_host": "prod1.seace.gob.pe", "selected": selection["selected"], "input_sha256": inputs,
            "historical_content_proven": False, "ranking_eligible": False, "labels_read": False,
            "automatic_retries": False}
    summary = {k: v for k, v in selection.items() if k != "selected"}
    summary.update(sample_size=len(queue), next_batch=len(selection["selected"]))
    output.mkdir(parents=True)
    for name, value in (("access_plan.json", plan), ("summary.json", summary)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    code = [Path(__file__), ROOT / "src/tesis_recommender/annex_queue.py", ROOT / "src/tesis_recommender/annex_sources.py"]
    (output / "manifest.json").write_text(json.dumps({"input_sha256": inputs,
        "code_sha256": {str(p): sha(p) for p in code},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, default=ROOT / "artifacts/experience_annex_plan_v2")
    p.add_argument("--access", type=Path, action="append")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_annex_plan_v3")
    a = p.parse_args()
    dirs = a.access or [ROOT / "artifacts/experience_document_access_v2", ROOT / "artifacts/experience_document_access_v3"]
    print(json.dumps(prepare(a.queue, dirs, a.output), ensure_ascii=False))
