"""Prepare contamination exclusions and gates, without selecting a new test.

This is a readiness snapshot, NOT a precommitted final evaluation protocol.
No unseen release, PDF, reference or prediction is opened or generated here.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/document_experience_next_validation_plan_v1.md"
BASE_FILES = (
    "artifacts/extraction_commitment_v1/review_packet.json",
    "artifacts/experience_source_inventory_v1/document_inventory.json",
    "artifacts/experience_substitute_plan_v1/pending_plan.json",
    "artifacts/experience_holdout_plan_v1/selection.json",
)


def known_rows(data):
    if isinstance(data, list):
        return data
    if "selected_opportunities" in data:
        return data["selected_opportunities"] + [{"opportunity_id": ocid} for ocid in data["excluded_ocids"]]
    if "pending_sources" in data:
        return data["pending_sources"]
    raise ValueError("Unrecognized exclusion source schema")


def collect_exclusions(payloads):
    ocids, hashes = defaultdict(set), defaultdict(set)
    for source, data in payloads:
        for row in known_rows(data):
            ocid = row["opportunity_id"]
            if not isinstance(ocid, str) or not ocid.startswith("ocds-"):
                raise ValueError("Missing or malformed known opportunity ID")
            ocids[ocid].add(source)
            # Inaccessible/insufficient cases remain excluded by OCID too.
            if row.get("status") == "downloaded_pdf":
                checksum = row.get("sha256", "")
                if not re.fullmatch(r"[a-f0-9]{64}", checksum):
                    raise ValueError("Downloaded PDF lacks a complete checksum")
                hashes[checksum].add(source)
    return {"excluded_ocids": sorted(ocids), "known_pdf_sha256": sorted(hashes),
            "ocid_provenance": {k: sorted(v) for k, v in sorted(ocids.items())},
            "pdf_provenance": {k: sorted(v) for k, v in sorted(hashes.items())}}


def verify_listed_file(path, inputs):
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checksum = manifest["output_sha256"].get(path.name)
    if checksum is None:
        raise ValueError("Exclusion file not pinned by its own output manifest: " + str(path))
    verify_hashes({str(path): checksum})
    inputs[str(path)] = checksum
    inputs[str(manifest_path)] = sha(manifest_path)


def build(output):
    if output.exists():
        raise ValueError("Use a new readiness directory")
    inputs = dict(protected_outputs())
    development = ROOT / "artifacts/document_experience_development_v2"
    verify_artifact(development, inputs)
    inputs.update(json.loads((development / "manifest.json").read_text())["input_sha256"])
    paths = sorted({*(ROOT / rel for rel in BASE_FILES),
                    *(ROOT / "artifacts").glob("experience*/**/access_results.json")})
    payloads = []
    for path in paths:
        verify_listed_file(path, inputs)
        payloads.append((str(path.relative_to(ROOT)), json.loads(path.read_text())))
    exclusions = collect_exclusions(payloads)
    code = (Path(__file__).resolve(), ROOT / "tests/test_document_experience_validation_readiness_v1.py", PLAN)
    inputs.update({str(path): sha(path) for path in code})
    summary = {"version": "document_experience_validation_readiness_v1",
        "status": "preparation_only_not_final_protocol_or_sample_freeze",
        "exclusion_source_files": len(paths), "excluded_opportunities": len(exclusions["excluded_ocids"]),
        "known_pdf_hashes": len(exclusions["known_pdf_sha256"]),
        "all_old_holdout_50_including_inaccessible_excluded": True,
        "new_sample_selected": False, "new_source_content_inspected": False,
        "pdf_downloads_performed": False, "reference_created": False, "predictions_generated": False,
        "metrics_computed": False, "ready_for_new_evaluation": False,
        "pending_gates": ["Complete development limitations or explicitly freeze their abstention policy.",
            "Predefine logical-unit identity and duplicate/conflict handling before sample selection.",
            "Freeze exact code, sampling seed, source/OCR pipeline, scope, reference schema and comparator.",
            "Refresh exclusion registry, then select without looking at requirement content or predictions.",
            "Create and lock source-based reference before inference; disclose reviewer dependence."],
        "generalization_limit": "Unseen document content within the same annual 2025 inventory is not a new year or population.",
        "ranking_inputs_modified": False, "frozen_evaluation_preserved": True}
    old = json.loads((ROOT / "artifacts/experience_holdout_plan_v1/selection.json").read_text())
    if not {r["opportunity_id"] for r in old["selected_opportunities"]} <= set(exclusions["excluded_ocids"]):
        raise ValueError("Prior holdout selection not fully excluded")
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "exclusion_registry.json", exclusions)
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve())))
