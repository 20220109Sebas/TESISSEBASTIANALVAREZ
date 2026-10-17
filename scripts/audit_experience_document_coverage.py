"""Audit all 50 fixed source states, including failed and unattempted documents."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tesis_recommender.document_coverage import coverage_rows, coverage_summary


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(evidence_dir, access_dirs, output):
    if output.exists():
        raise ValueError("Use a new coverage directory")
    inputs = {}
    def checked(directory, name):
        p = directory / name
        m = json.loads((directory / "manifest.json").read_text())
        if sha(p) != m["output_sha256"][name]:
            raise ValueError("Coverage source artifact changed")
        inputs[str(p)] = sha(p)
        inputs[str(directory / "manifest.json")] = sha(directory / "manifest.json")
        return json.loads(p.read_text())
    queue = checked(ROOT / "artifacts/experience_annex_plan_v2", "queue.json")
    accesses = [r for d in access_dirs for r in checked(d, "access_results.json")]
    for r in accesses:
        if "file_path" in r:
            if sha(r["file_path"]) != r["sha256"]:
                raise ValueError("Complete or partial downloaded bytes changed")
            inputs[r["file_path"]] = r["sha256"]
    docs = checked(evidence_dir, "document_evidence.json")
    rows = coverage_rows(queue, accesses, docs)
    summary = coverage_summary(rows)
    for p, expected in inputs.items():
        if sha(p) != expected:
            raise ValueError("Coverage input changed")
    output.mkdir(parents=True)
    for name, value in (("opportunity_coverage.json", rows), ("summary.json", summary)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    code = [Path(__file__), ROOT / "src/tesis_recommender/document_coverage.py"]
    (output / "manifest.json").write_text(json.dumps({"input_sha256": inputs,
        "code_sha256": {str(p): sha(p) for p in code},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, default=ROOT / "artifacts/experience_document_evidence_v4")
    p.add_argument("--access", type=Path, action="append")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_document_coverage_v1")
    a = p.parse_args()
    access = a.access or [ROOT / f"artifacts/experience_document_access_v{i}" for i in (2, 3, 4, 5, 6, 7)]
    print(json.dumps(audit(a.evidence, access, a.output), ensure_ascii=False))
