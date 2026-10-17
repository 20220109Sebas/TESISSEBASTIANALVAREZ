"""Freeze the next five missing bases after title filtering; reuse checked PDF bytes."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tesis_recommender.annex_sources import select_annexes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output):
    if output.exists():
        raise ValueError("Use a new plan directory")
    inventory_dir = ROOT / "artifacts/experience_source_inventory_v1"
    access_dir = ROOT / "artifacts/experience_document_access_v2"
    inv_manifest = json.loads((inventory_dir / "manifest.json").read_text())
    access_manifest = json.loads((access_dir / "manifest.json").read_text())
    evidence = {}
    for path, expected in ((inventory_dir / "document_inventory.json", inv_manifest["output_sha256"]["document_inventory.json"]),
                           (access_dir / "access_results.json", access_manifest["output_sha256"]["access_results.json"])):
        if sha(path) != expected:
            raise ValueError("Source inventory or access results changed")
        evidence[str(path)] = expected
    inventory = json.loads((inventory_dir / "document_inventory.json").read_text())
    chosen, excluded, eligible = select_annexes(inventory)
    downloaded = {row["url"]: row for row in json.loads((access_dir / "access_results.json").read_text()) if row["status"] == "downloaded_pdf"}
    reused, pending = [], []
    for doc in chosen:
        old = downloaded.get(doc["url"])
        if old:
            path = Path(old["file_path"])
            if sha(path) != old["sha256"]:
                raise ValueError("Cached PDF changed")
            evidence[str(path)] = old["sha256"]
            reused.append(old)
        else:
            pending.append(doc)
    plan = {"purpose": "next_bounded_annex_access_after_document_role_filter_not_validation",
            "selection": "one_requirement_document_per_fixed_sample_opportunity_then_next_5_uncached_in_sample_order",
            "metadata_basis": "declared document date only, not historical content proof",
            "limits": {"documents": 5, "max_bytes_per_document": 10 * 1024 * 1024,
                       "timeout_seconds": 30, "max_pages_per_document": 120, "workers": 1, "ocr": False},
            "allowed_host": "prod1.seace.gob.pe", "selected": pending[:5],
            "input_sha256": evidence, "no_outcome_documents": True, "no_manual_annotations": True,
            "historical_content_proven": False, "ranking_eligible": False}
    summary = {"inventory_entries": len(inventory), "eligible_role_entries": len(eligible),
               "selected_opportunities": len(chosen), "reused_downloaded_pdfs": len(reused),
               "pending_opportunities": len(pending), "next_batch": len(plan["selected"]),
               "exclusion_reasons": dict(Counter(d["selection_reason"] for d in excluded)),
               "profile_or_label_selection": False}
    output.mkdir(parents=True)
    for name, value in (("access_plan.json", plan), ("summary.json", summary),
                        ("queue.json", chosen), ("excluded.json", excluded), ("reused.json", reused)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"input_sha256": evidence,
        "script_sha256": sha(Path(__file__)), "selector_sha256": sha(ROOT / "src/tesis_recommender/annex_sources.py"),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_annex_plan_v2")
    a = p.parse_args()
    print(json.dumps(prepare(a.output)))
