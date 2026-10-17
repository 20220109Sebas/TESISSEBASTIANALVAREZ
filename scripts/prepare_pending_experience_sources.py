"""Fix all untouched source URLs and extraction code before retrieving more PDFs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tesis_recommender.annex_sources import safe_annex_url


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition_pending(queue, accesses):
    if len({r["url"] for r in queue}) != len(queue) or len({r["opportunity_id"] for r in queue}) != len(queue):
        raise ValueError("Duplicate fixed source")
    attempted = {r["url"] for r in accesses}
    pending = [r for r in queue if r["url"] not in attempted]
    return [pending[i:i + 5] for i in range(0, len(pending), 5)]


def prepare(output, expected_count):
    if output.exists():
        raise ValueError("Choose a new plan directory")
    inputs = {}
    def checked(directory, filename):
        manifest = directory / "manifest.json"
        p = directory / filename
        if sha(p) != json.loads(manifest.read_text())["output_sha256"][filename]:
            raise ValueError("Saved input changed")
        inputs[str(p)] = sha(p); inputs[str(manifest)] = sha(manifest)
        return json.loads(p.read_text())
    queue = checked(ROOT / "artifacts/experience_annex_plan_v2", "queue.json")
    accesses = [r for i in range(2, 10) for r in checked(ROOT / f"artifacts/experience_document_access_v{i}", "access_results.json")]
    batches = partition_pending(queue, accesses)
    pending = [r for batch in batches for r in batch]
    if len(queue) != 50 or len(pending) != expected_count or not pending:
        raise ValueError("Unexpected fixed sample or untouched source count")
    if any(not safe_annex_url(r["url"]) or r["temporal_metadata_status"] != "declared_at_or_before_initial" for r in pending):
        raise ValueError("Unexpected source URL or metadata")
    files = [ROOT / "src/tesis_recommender" / f for f in ["experience.py", "experience_segments.py", "experience_conditions.py", "experience_formulations.py", "ocr_evidence.py", "document_coverage.py"]]
    files += [ROOT / "scripts" / f for f in ["fetch_experience_documents.py", "inspect_experience_pdfs.py", "extend_experience_document_evidence.py", "extend_experience_formulations.py", "represent_experience_conditions.py", "compare_experience_segments.py", "audit_experience_document_coverage.py"]]
    frozen_code = {str(p): sha(p) for p in files}
    inputs.update(frozen_code)
    output.mkdir(parents=True)
    plan = {"purpose": "all_untouched_sources_with_rules_frozen_before_retrieval",
            "sample_size": len(queue), "pending_sources": pending,
            "frozen_rule_and_runner_sha256": frozen_code,
            "batches": [], "input_sha256": inputs, "labels_read": False,
            "automatic_retries": False, "historical_content_proven": False,
            "ranking_eligible": False}
    for i, batch in enumerate(batches, 1):
        folder = output / "batches" / f"batch_{i:02d}"
        folder.mkdir(parents=True)
        part = {"purpose": plan["purpose"], "selection": "fixed_queue_order_no_outcome_selection_no_retries",
                "selected": batch, "input_sha256": inputs, "allowed_host": "prod1.seace.gob.pe",
                "limits": {"documents": 5, "max_bytes_per_document": 10 * 1024 * 1024,
                           "timeout_seconds": 30, "max_pages_per_document": 120, "workers": 1, "ocr": False},
                "labels_read": False, "automatic_retries": False, "ranking_eligible": False,
                "historical_content_proven": False}
        path = folder / "access_plan.json"
        path.write_text(json.dumps(part, ensure_ascii=False, indent=2) + "\n")
        (folder / "manifest.json").write_text(json.dumps({"output_sha256": {path.name: sha(path)}}, indent=2) + "\n")
        plan["batches"].append({"path": str(path), "sha256": sha(path), "sources": len(batch)})
    path = output / "pending_plan.json"
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"input_sha256": inputs,
        "code_sha256": {str(Path(__file__)): sha(__file__)},
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}}, indent=2) + "\n")
    return {"untouched_sources_fixed": len(pending), "batches": len(batches), "rule_files_frozen": len(frozen_code)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-count", type=int, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.output.resolve(), a.expected_count)))
