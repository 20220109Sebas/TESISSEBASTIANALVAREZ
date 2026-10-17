"""Check preservation and provenance of the roller/UI review block, not accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build(output):
    require(not output.exists(), "Use a new verification directory")
    before = ROOT / "artifacts/experience_holdout_review_progress_v10"
    partial = ROOT / "artifacts/experience_holdout_review_progress_v11"
    after = ROOT / "artifacts/experience_holdout_review_progress_v12"
    inputs = dict(protected_outputs())
    for directory in (before, partial, after):
        verify_artifact(directory, inputs)
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    lock_counts = {}
    for name, key in (
        ("experience_substitute_plan_v1/pending_plan.json", "frozen_rule_and_runner_sha256"),
        ("experience_holdout_models_lock_v1/model_lock.json", "original_code_sha256"),
        ("experience_holdout_evaluation_preflight_v1/evaluation_lock.json", "comparison_code_sha256"),
    ):
        path = ROOT / "artifacts" / name
        locks = json.loads(path.read_text())[key]
        inputs.update(locks)
        inputs[str(path)] = sha(path)
        lock_counts[key] = len(locks)
    old, middle, current = [json.loads((d / "review_progress.json").read_text()) for d in (before,partial,after)]
    require(len(old) == len(middle) == len(current) == 50, "Changed denominator")
    require([r["opportunity_id"] for r in old] == [r["opportunity_id"] for r in current], "Reordered or replaced cases")
    require([r["sample_number"] for r in old if r not in current] == [7,39], "Unexpected case changed")
    require(all(a == b for a,b in zip(old,middle) if a["sample_number"] != 39), "Partial pass changed another case")
    require(all(a == b for a,b in zip(middle,current) if a["sample_number"] != 7), "Second pass changed another case")
    for filename in ("compiled_source_reviews.json",):
        require(sha(before / filename) == sha(partial / filename) == sha(after / filename), "Full inventory changed")
    require(sha(partial / "partial_source_reviews.json") == sha(after / "partial_source_reviews.json"), "Partial inventory lost")
    old_evidence = json.loads((before / "second_pass_evidence.json").read_text())
    evidence = json.loads((after / "second_pass_evidence.json").read_text())
    require(evidence[:len(old_evidence)] == old_evidence and len(evidence) == len(old_evidence)+4,
        "Prior second evidence lost or wrong number added")
    source_path = ROOT / "artifacts/experience_holdout_source_view_v1/source_documents.json"
    sources = {d["opportunity_id"]:d for d in json.loads(source_path.read_text())}
    for e in evidence[len(old_evidence):]:
        source = sources[e["opportunity_id"]]
        page = next(p for p in source["pages"] if p["page"] == e["page"])
        require(page["text"][e["char_start"]:e["char_end"]] == e["quote"], "Second quote differs")
        require(hashlib.sha256(page["text"].encode()).hexdigest() == e["page_text_sha256"], "Second text hash differs")
        require(e["pdf_sha256"] == source["source"]["sha256"] and sha(e["image_path"]) == e["image_sha256"],
            "Second source/image lineage differs")
    row = next(r for r in current if r["sample_number"] == 7)
    require(row["second_pass_independent_reviewer"] is False, "Same reviewer falsely independent")
    coverage = next(r for r in current if r["sample_number"] == 39)["latest_partial_first_pass_coverage"]
    require(len(coverage["unreviewed_pages"]) == 28 and coverage["full_pdf_semantically_read"] is False,
        "Partial coverage overstated")
    replay_counts = {}
    for directory in (partial,after,ROOT / "artifacts/experience_holdout_roller_partial_verification_v1"):
        replay = Path(str(directory)+"_replay")
        manifest = json.loads((directory / "manifest.json").read_text())
        for filename in manifest["output_sha256"]:
            require(sha(directory / filename) == sha(replay / filename), "Replay differs: "+filename)
        replay_counts[directory.name] = len(manifest["output_sha256"])
    summary = json.loads((after / "summary.json").read_text())
    for key in ("reference_locked","predictions_generated","f1_estimable","ranking_inputs_modified"):
        require(summary[key] is False, "Unexpected evaluation/ranking finality")
    verify_hashes(inputs)
    result = {"selected_cases_preserved":50, "other_cases_unchanged":48,
        "first_pass_full":summary["first_pass_full_supplied_pdf"], "first_pass_partial":summary["first_pass_partial"],
        "downloaded_not_read":summary["downloaded_sources_not_read"], "second_pass_full":summary["second_pass_full_supplied_pdf"],
        "new_partial_pages_read":20,"new_partial_pages_remaining":28,"new_second_pages_read":6,
        "new_second_reviewer_independent":False,"protected_outputs_verified":len(protected_outputs()),
        "input_hashes_verified":len(inputs),"code_lock_counts":lock_counts,"reproduced_logical_outputs":replay_counts,
        "new_literal_quotes":11,"new_second_literal_quotes":4,"full_inventory_unchanged":True,
        "reference_locked":False,"model_accuracy_evaluated":False,
        "verification_scope":"preservation_and_literal_provenance_not_independent_semantic_gold"}
    output.mkdir(parents=True)
    dump(output / "checks.json",result)
    dump(output / "manifest.json",{"input_sha256":inputs,"script_sha256":sha(__file__),
        "output_sha256":{"checks.json":sha(output / "checks.json")}})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    print(json.dumps(build(parser.parse_args().output.resolve())))
