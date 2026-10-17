"""Assemble source-only review inputs, retaining all 50 selected opportunities."""
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.register_browser_experience_sources import dump, sha, verify_hashes, protected_outputs

ROOT = Path(__file__).resolve().parents[1]


def access_status(row, known_hashes, seen_hashes):
    if row["status"] != "downloaded_pdf":
        return row["status"]
    if row["sha256"] in known_hashes:
        return "excluded_duplicate_development_pdf"
    if row["sha256"] in seen_hashes:
        return "excluded_duplicate_holdout_pdf"
    seen_hashes.add(row["sha256"])
    return "downloaded_unique_pdf"


def build(plan_dir, access_dir, inspection_dir, output):
    if output.exists():
        raise ValueError("Preserve existing packet and use a new directory")
    manifest_path = plan_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = dict(manifest["input_sha256"])
    inputs.update({str(plan_dir / p): h for p, h in manifest["output_sha256"].items()})
    inputs[str(manifest_path)] = sha(manifest_path)
    inputs.update(protected_outputs())
    verify_hashes(inputs)
    selection = json.loads((plan_dir / "selection.json").read_text())
    queue = json.loads((plan_dir / "queue.json").read_text())
    expected = {r["opportunity_id"]: r for r in queue}
    accessed, inspections = {}, {}
    for directory in sorted(access_dir.glob("batch_*")):
        path = directory / "manifest.json"
        if not path.exists():
            raise ValueError("Incomplete access batch")
        access_manifest = json.loads(path.read_text())
        checked = {str(directory / p): h for p, h in access_manifest["output_sha256"].items()}
        verify_hashes(checked)
        inputs.update(checked)
        inputs[str(path)] = sha(path)
        rows = json.loads((directory / "access_results.json").read_text())
        for row in rows:
            ocid = row["opportunity_id"]
            if ocid not in expected or ocid in accessed or row["url"] != expected[ocid]["url"]:
                raise ValueError("Unexpected or repeated access source")
            accessed[ocid] = row
    if set(accessed) != set(expected):
        raise ValueError("All fixed eligible sources must have an access state")
    for path in sorted(inspection_dir.glob("batch_*/manifest.json")):
        m = json.loads(path.read_text())
        checked = {str(path.parent / p): h for p, h in m["output_sha256"].items()}
        verify_hashes(checked)
        inputs.update(checked)
        inputs[str(path)] = sha(path)
        for doc in json.loads((path.parent / "pdf_text.json").read_text())["documents"]:
            ocid = doc["opportunity_id"]
            if ocid in inspections or ocid not in accessed:
                raise ValueError("Unexpected inspection")
            if doc.get("source") and doc["source"]["sha256"] != accessed[ocid]["sha256"]:
                raise ValueError("PDF inspection differs from access")
            inspections[ocid] = doc
    known, seen = set(selection["known_development_pdf_sha256"]), set()
    packet = []
    for sample in selection["selected_opportunities"]:
        ocid = sample["opportunity_id"]
        access = accessed.get(ocid)
        status = access_status(access, known, seen) if access else sample["status"]
        doc = inspections.get(ocid) if status == "downloaded_unique_pdf" else None
        review_status = "pending_source_review"
        if doc is None:
            review_status = "source_not_ready"
        elif doc.get("truncated_by_page_limit"):
            review_status = "source_exceeds_page_limit"
        elif doc.get("status") != "text_extracted":
            review_status = "source_not_readable"
        elif all(not p["text"].strip() for p in doc.get("pages", [])):
            review_status = "requires_ocr_or_source_visual_review"
        packet.append({"sample_number": sample["sample_number"], "opportunity_id": ocid,
            "source_status": status, "source": access,
            "pdf_total_pages": doc.get("total_pages") if doc else None,
            "source_text_path": next((str(p) for p in inspection_dir.glob("batch_*/pdf_text.json")
                if doc and any(d["opportunity_id"] == ocid for d in json.loads(p.read_text())["documents"])), None),
            "review_status": review_status, "reference_conditions": None,
            "scope_verification": "pending_source_review", "first_source_only_review_complete": False,
            "second_source_only_review_complete": False, "predictions_generated": False,
            "human_annotation": False, "historical_content_proven": False, "ranking_eligible": False})
    summary = {"selected_opportunities": len(packet), "access_attempts": len(accessed),
        "source_status": dict(Counter(r["source_status"] for r in packet)),
        "review_status": dict(Counter(r["review_status"] for r in packet)),
        "reference_complete": 0, "predictions_generated": False,
        "reference_is_human_or_expert": False, "f1_estimable": False,
        "protected_outputs_verified": len(protected_outputs())}
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "source_review_packet.json", packet)
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
         "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "access", "inspection", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.plan.resolve(), args.access.resolve(), args.inspection.resolve(), args.output.resolve())))
