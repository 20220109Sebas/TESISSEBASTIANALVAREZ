"""Record a source amendment without counting metadata or failures as documents."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_experience_documents import sha
from scripts.fetch_pending_experience_sources import check_code


def summarize(output):
    if output.exists():
        raise ValueError("Choose a new amendment summary directory")
    inputs = {}

    def read(folder, name):
        directory = ROOT / "artifacts" / folder
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for relative, expected in manifest["output_sha256"].items():
            if sha(directory / relative) != expected:
                raise ValueError("Artifact changed: " + str(directory / relative))
        path = directory / name
        inputs[str(path)] = sha(path)
        inputs[str(manifest_path)] = sha(manifest_path)
        return json.loads(path.read_text())

    original = read("experience_pending26_plan_v1", "pending_plan.json")
    check_code(original)
    records = read("experience_substitute_records_v1", "record_access.json")
    plan = read("experience_substitute_plan_v1", "pending_plan.json")
    decisions = read("experience_substitute_plan_v1", "decisions.json")
    rows = [r for i in range(1, len(plan["batches"]) + 1)
        for r in read(f"experience_substitute_access_v1/batch_{i:02d}", "access_results.json")]
    other = read("experience_substitute_other_access_v1", "access_results.json")
    coverage = read("experience_document_coverage_v3", "summary.json")
    if any(r["status"] == "downloaded_pdf" for r in rows):
        raise ValueError("New PDFs require a separate extraction audit before a failure summary")
    expected_ocids = [r["opportunity_id"] for r in original["pending_sources"]]
    if expected_ocids != [r["opportunity_id"] for r in records] or expected_ocids != [r["opportunity_id"] for r in decisions]:
        raise ValueError("Substitute inventory differs from the fixed original processes")
    if [r["url"] for r in rows] != [r["url"] for r in plan["pending_sources"]]:
        raise ValueError("Substitute download attempts differ from the fixed amendment")
    by_pdf = {r["opportunity_id"]: r for r in rows}
    by_other = {r["opportunity_id"]: r for r in other}
    if set(by_pdf) & set(by_other):
        raise ValueError("Duplicate process in separate access routes")
    amendments = []
    for original_doc, record, decision in zip(original["pending_sources"], records, decisions):
        ocid = original_doc["opportunity_id"]
        access = by_pdf.get(ocid) or by_other.get(ocid)
        amendments.append({"opportunity_id": ocid, "sample_number": original_doc["sample_number"],
            "original_document": original_doc, "current_record_url": record["record_url"],
            "current_record_sha256": record.get("sha256"),
            "current_record_retrieved_at_utc": record["finished_at_utc"],
            "selected_pdf_substitute": decision["selected"],
            "access_attempt": access,
            "amendment_state": "pdf_substitute_not_recovered" if ocid in by_pdf else
                "other_format_access_diagnostic_only" if ocid in by_other else
                "no_eligible_requirement_alternative_in_checked_record",
            "rules_applied_to_new_document": False, "original_document_replaced": False,
            "historical_content_proven": False, "ranking_eligible": False,
            "absence_of_requirements_proven": False, "negative_label": False})
    protected = {}
    annual_dir = ROOT / "artifacts/experiment_annual_v1"
    annual_manifest_path = annual_dir / "manifest.json"
    annual = json.loads(annual_manifest_path.read_text())
    for field in ("ranking_sha256", "component_sha256"):
        for relative, expected in annual[field].items():
            protected[str(annual_dir / relative)] = expected
    protected[str(annual_dir / "candidate_order.json")] = annual["candidate_order_sha256"]
    protected[str(annual_dir / "results.json")] = annual["results_sha256"]
    extraction_dir = ROOT / "artifacts/extraction_commitment_v1"
    extraction_manifest_path = extraction_dir / "manifest.json"
    extraction = json.loads(extraction_manifest_path.read_text())
    for relative, expected in extraction["output_sha256"].items():
        protected[str(extraction_dir / relative)] = expected
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError("Protected experiment changed: " + path)
    inputs.update(protected)
    inputs.update(original["frozen_rule_and_runner_sha256"])
    inputs[str(annual_manifest_path)] = sha(annual_manifest_path)
    inputs[str(extraction_manifest_path)] = sha(extraction_manifest_path)
    summary = {"original_processes_checked": len(amendments),
        "current_records_saved": sum(r["status"] == "record_saved" for r in records),
        "pdf_substitutes_identified_and_attempted": len(rows),
        "pdf_GET_statuses": dict(Counter(str(r.get("http_status")) for r in rows)),
        "pdf_temporal_metadata": dict(Counter(r["temporal_metadata_status"] for r in rows)),
        "other_format_alternatives_probed": len(other),
        "other_formats": dict(Counter(r["format"] for r in other)),
        "other_GET_prefix_statuses": dict(Counter(str(r.get("http_status")) for r in other)),
        "processes_without_eligible_requirement_alternative_in_checked_record": len(amendments) - len(rows) - len(other),
        "new_complete_documents_recovered": 0, "new_documents_with_rules_applied": 0,
        "existing_complete_pdf_opportunities": coverage["complete_pdf_opportunities"],
        "fixed_sample_size": coverage["sample_size"],
        "frozen_rule_files_verified": len(original["frozen_rule_and_runner_sha256"]),
        "protected_annual_outputs_verified": len(annual["ranking_sha256"]) + len(annual["component_sha256"]) + 2,
        "protected_extraction_outputs_verified": len(extraction["output_sha256"]),
        "source_amendment_authorized_by_user": True, "retrieval_complete": False,
        "rules_unchanged": True, "ranking_inputs_modified": False, "f1_estimable": False,
        "negative_labels_created": False, "new_ocr_performed": False,
        "new_embeddings_generated": False, "historical_content_proven": False}
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError("Input changed during summary")
    check_code(original)
    output.mkdir(parents=True)
    for name, value in [("source_amendment.json", amendments), ("summary.json", summary)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"input_sha256": inputs,
        "code_sha256": {str(Path(__file__)): sha(__file__)},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.output.resolve())))
