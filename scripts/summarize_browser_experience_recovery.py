"""Verify and summarize separate public-SEACE recovery, not scientific accuracy."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.register_browser_experience_sources import dump, protected_outputs, verify_hashes
from scripts.fetch_experience_documents import sha
from scripts.fetch_pending_experience_sources import check_code


def summarize(output):
    if output.exists():
        raise ValueError("Use a new summary directory")
    inputs = dict(protected_outputs())
    plan_path = ROOT / "artifacts/experience_substitute_plan_v1/pending_plan.json"
    plan = json.loads(plan_path.read_text())
    check_code(plan)
    inputs[str(plan_path)] = sha(plan_path)
    groups = [("v1", ""), ("v2", "batch_02"), ("v2", "batch_03")]
    accesses, summaries = [], []
    replays = []
    for version, batch in groups:
        access = ROOT / f"artifacts/experience_substitute_browser_access_{version}" / batch
        manifest = json.loads((access / "manifest.json").read_text())
        verify_hashes({str(access / p): h for p, h in manifest["output_sha256"].items()})
        verify_hashes(manifest["input_sha256"])
        verify_hashes(manifest["code_sha256"])
        inputs[str(access / "manifest.json")] = sha(access / "manifest.json")
        rows = json.loads((access / "access_results.json").read_text())
        accesses.extend(rows)
        for kind, names in (("inspection", ("pdf_text.json", "manifest.json")),
                            ("extraction", ("document_evidence.json", "conditional_variables.json",
                                            "formulation_candidates.json", "summary.json", "manifest.json"))):
            directory = ROOT / f"artifacts/experience_substitute_browser_{kind}_{version}" / batch
            replay = ROOT / f"artifacts/experience_substitute_browser_{kind}_replay_{version}" / batch
            manifest = json.loads((directory / "manifest.json").read_text())
            verify_hashes({str(directory / p): h for p, h in manifest["output_sha256"].items()})
            for name in names:
                if (directory / name).read_bytes() != (replay / name).read_bytes():
                    raise ValueError("Replay differs: " + str(directory / name))
                inputs[str(directory / name)] = sha(directory / name)
                inputs[str(replay / name)] = sha(replay / name)
            replays.append({"version": version, "batch": batch or "batch_01",
                            "kind": kind, "files_identical": len(names)})
        summaries.append(json.loads((ROOT / f"artifacts/experience_substitute_browser_extraction_{version}"
                                     / batch / "summary.json").read_text()))
    if [r["opportunity_id"] for r in accesses] != [r["opportunity_id"] for r in plan["pending_sources"]]:
        raise ValueError("Access inventory differs from the fixed thirteen-source queue")
    visual_path = ROOT / "artifacts/experience_substitute_browser_observations_v2/visual_review.json"
    visual = json.loads(visual_path.read_text())
    inputs[str(visual_path)] = sha(visual_path)
    recovered = {r["opportunity_id"]: r for r in accesses if r["status"] == "downloaded_pdf"}
    for review in visual["reviews"]:
        if review["pdf_sha256"] != recovered[review["opportunity_id"]]["sha256"]:
            raise ValueError("Visual review has different PDF identity")
    exception_path = visual_path.parent / "download_exceptions.json"
    inputs[str(exception_path)] = sha(exception_path)
    exceptions = json.loads(exception_path.read_text())["exceptions"]
    for exception in exceptions:
        if (exception["copied_to_corpus"] or exception["pdf_processed"] or exception["ranking_eligible"]
            or exception["bytes"] <= exception["max_bytes_per_document"]):
            raise ValueError("Unexpected exception disposition")
        if Path(exception["download_path"]).stat().st_size != exception["bytes"]:
            raise ValueError("Exception file size changed")
        inputs[exception["download_path"]] = sha(exception["download_path"])
        if inputs[exception["download_path"]] in {r["sha256"] for r in recovered.values()}:
            raise ValueError("Excluded download was included in the packet")
    coverage_path = ROOT / "artifacts/experience_document_coverage_v3/summary.json"
    coverage_manifest_path = coverage_path.parent / "manifest.json"
    coverage_manifest = json.loads(coverage_manifest_path.read_text())
    verify_hashes({str(coverage_path.parent / p): h for p, h in coverage_manifest["output_sha256"].items()})
    inputs[str(coverage_path)] = sha(coverage_path)
    inputs[str(coverage_manifest_path)] = sha(coverage_manifest_path)
    coverage = json.loads(coverage_path.read_text())
    counters = ("pages_processed", "qualification_clauses", "evaluation_factor_clauses",
                "conditional_variables", "conditional_monetary_branches", "additional_literal_candidates")
    cumulative = {key: sum(s[key] for s in summaries) for key in counters}
    current = {key: sum(s[key] for s in summaries[1:]) for key in counters}
    result = {
        "fichas_consulted_total": len(accesses), "fichas_consulted_this_block": 8,
        "accepted_substitute_pdfs_total": len(recovered), "accepted_substitute_pdfs_this_block": 4,
        "access_states_total": dict(Counter(r["status"] for r in accesses)),
        "cumulative_substitute_extraction": cumulative, "this_block_extraction": current,
        "original_coverage_unchanged": coverage,
        "mixed_native_and_image_documents_observed_this_block": sum(
            bool(r.get("mixed_native_and_image_content_observed")) for r in visual["reviews"]),
        "excluded_unexpected_downloads": len(exceptions), "replays": replays,
        "protected_annual_outputs_verified": len(protected_outputs()) - 4,
        "protected_extraction_outputs_verified": 4,
        "frozen_rule_files_verified": len(plan["frozen_rule_and_runner_sha256"]),
        "new_ocr_performed": False, "human_annotation": False, "f1_estimable": False,
        "ranking_inputs_modified": False, "historical_content_proven": False,
        "absence_of_requirements_proven": False,
    }
    verify_hashes(inputs)
    check_code(plan)
    output.mkdir(parents=True)
    dump(output / "summary.json", result)
    dump(output / "manifest.json", {"input_sha256": inputs,
        "code_sha256": {str(Path(__file__)): sha(__file__)},
        "output_sha256": {"summary.json": sha(output / "summary.json")}})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.output.resolve()), ensure_ascii=False))
