"""Run the fixed source batches sequentially, keeping failed accesses without retries."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_experience_documents import fetch, sha


def check_code(plan):
    for p, expected in plan["frozen_rule_and_runner_sha256"].items():
        if sha(p) != expected:
            raise ValueError("Rules or runners changed after source selection")


def run(plan_dir, output):
    if output.exists():
        raise ValueError("Choose a new access directory")
    path = plan_dir / "pending_plan.json"
    manifest = json.loads((plan_dir / "manifest.json").read_text())
    if sha(path) != manifest["output_sha256"][path.name]:
        raise ValueError("Pending plan changed")
    plan = json.loads(path.read_text())
    check_code(plan)
    output.mkdir(parents=True)
    rows = []
    for i, batch in enumerate(plan["batches"], 1):
        check_code(plan)
        if sha(batch["path"]) != batch["sha256"]:
            raise ValueError("Batch plan changed")
        folder = output / f"batch_{i:02d}"
        fetch(Path(batch["path"]), folder)
        rows.extend(json.loads((folder / "access_results.json").read_text()))
        print(json.dumps({"batch_complete": i, "attempted_so_far": len(rows)}), flush=True)
    check_code(plan)
    if [r["url"] for r in rows] != [r["url"] for r in plan["pending_sources"]]:
        raise ValueError("Accesses differ from fixed pending sources")
    summary = {"attempted_sources": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
               "rules_unchanged": True, "automatic_retries": False, "historical_content_proven": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"input_sha256": {str(path): sha(path)},
        "code_sha256": {str(Path(__file__)): sha(__file__)},
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.plan.resolve(), a.output.resolve())))
