"""Attach literal and visual provenance to source-only scope observations."""
import argparse
import json
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.build_holdout_reference_draft import ref
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]


def build(source_dir, decisions_path, output):
    if output.exists():
        raise ValueError("Use a new observation directory")
    inputs = dict(protected_outputs())
    verify_artifact(source_dir, inputs)
    decisions = json.loads(decisions_path.read_text())
    if decisions.get("predictions_seen") is not False or decisions.get("whole_document_conditions_reviewed") is not False:
        raise ValueError("Unexpected scope review status")
    inputs[str(decisions_path)] = sha(decisions_path)
    by_id = {d["opportunity_id"]: d for d in json.loads((source_dir / "source_documents.json").read_text())}
    rows = []
    for decision in decisions["decisions"]:
        doc = by_id[decision["opportunity_id"]]
        page = next(p for p in doc["pages"] if p["page"] == decision["page"])
        if page["text"].count(decision["quote"]) != 1:
            raise ValueError("Scope evidence is missing or ambiguous")
        at = page["text"].index(decision["quote"])
        image = ROOT / decision["image"]
        pdf = Path(doc["source"]["file_path"]).resolve()
        inputs[str(image)] = sha(image)
        inputs[str(pdf)] = doc["source"]["sha256"]
        rows.append(dict(decision, evidence=ref(page["text"], at, at + len(decision["quote"]), page["page"]),
            source_image_path=str(image), source_image_sha256=sha(image),
            source_pdf_path=str(pdf), source_pdf_sha256=doc["source"]["sha256"],
            source_reading_method=page["extraction_method"] + "_and_full_page_visual_review",
            reviewer=decisions["reviewer"], predictions_seen=False, human_annotation=False,
            absence_of_experience_proven=False, replaces_selected_case=False,
            final_reference_scope_locked=False, ranking_eligible=False))
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "scope_observations.json", rows)
    summary = {"first_pass_scope_observations": len(rows), "final_scope_decisions": 0,
        "sample_replacements": 0, "predictions_generated": False, "ranking_inputs_modified": False}
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "decisions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source.resolve(), args.decisions.resolve(), args.output.resolve())))
