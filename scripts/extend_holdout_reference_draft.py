"""Preserve partial source decisions with native/OCR evidence, never predictions."""
import argparse
import json
import re
from pathlib import Path

from scripts.build_holdout_reference_draft import check_literal_amount, ref
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]


def literal_span(text, start_quote, end_quote, value_quote, page):
    if text.count(start_quote) != 1:
        raise ValueError("Context start is not unique")
    start = text.index(start_quote)
    end = text.index(end_quote, start + len(start_quote))
    context = text[start:end]
    if context.count(value_quote) != 1:
        raise ValueError("Field evidence is missing or ambiguous within its branch")
    at = start + context.index(value_quote)
    return ref(text, start, end, page), ref(text, at, at + len(value_quote), page)


def build(source_dir, base_dir, decisions_path, output):
    if output.exists():
        raise ValueError("Use a new reference draft directory")
    inputs = dict(protected_outputs())
    for directory in (source_dir, base_dir):
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        verify_hashes(manifest["input_sha256"])
        output_hashes = {str(directory / p): h for p, h in manifest["output_sha256"].items()}
        verify_hashes(output_hashes)
        inputs.update(output_hashes)
        inputs[str(manifest_path)] = sha(manifest_path)
    decision = json.loads(decisions_path.read_text())
    if decision.get("predictions_seen") is not False or decision.get("documents_complete") is not False:
        raise ValueError("Only explicitly source-only partial decisions are permitted")
    inputs[str(decisions_path)] = sha(decisions_path)
    documents = {d["opportunity_id"]: d for d in json.loads((source_dir / "source_documents.json").read_text())}
    rows = json.loads((base_dir / "reference_draft.json").read_text())
    for case in decision["conditions"]:
        doc = documents[case["opportunity_id"]]
        page = next(p for p in doc["pages"] if p["page"] == case["page"])
        context, amount = literal_span(page["text"], case["start_quote"], case["end_quote"], case["amount_quote"], case["page"])
        check_literal_amount(case["amount_quote"], case["billed_amount"])
        window = None
        if case["window_quote"] is not None:
            _, window = literal_span(page["text"], case["start_quote"], case["end_quote"], case["window_quote"], case["page"])
            number = re.search(r"\((\d+)\) años anteriores", window["literal"])
            if not number or int(number.group(1)) != case["lookback_years"]:
                raise ValueError("Window differs from literal")
        elif case["lookback_years"] is not None:
            raise ValueError("No period can be assigned without source evidence")
        image = ROOT / case["image"]
        pdf = Path(doc["source"]["file_path"]).resolve()
        inputs[str(image)] = sha(image)
        inputs[str(pdf)] = doc["source"]["sha256"]
        rows.append({"opportunity_id": case["opportunity_id"], "item": case["item"], "branch": case["branch"],
            "subject": "bidder", "section_role": "qualification", "kind": "required_billed_experience",
            "billed_amount": case["billed_amount"], "currency": case["currency"], "lookback_years": case["lookback_years"],
            "source_pdf_path": str(pdf), "source_pdf_sha256": doc["source"]["sha256"],
            "source_image_path": str(image), "source_image_sha256": sha(image),
            "context": context, "amount_evidence": amount, "window_evidence": window,
            "source_reading_method": page["extraction_method"] + "_and_full_page_visual_review",
            "native_text_sha256": page["native_text_sha256"], "warnings": case["warnings"],
            "review_status": "first_pass_partial_document_draft", "reviewer": "automatic_source_review",
            "predictions_seen": False, "second_source_only_pass_complete": False,
            "document_reference_complete": False, "company_applicability": "not_determined",
            "ranking_eligible": False, "human_annotation": False, "historical_content_proven": False})
    identity = [(r["opportunity_id"], r["subject"], r["section_role"], r["item"], r["branch"]) for r in rows]
    if len(set(identity)) != len(identity):
        raise ValueError("Duplicate draft identity")
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "reference_draft.json", rows)
    summary = {"documents_with_first_pass_conditions": len({r["opportunity_id"] for r in rows}),
        "first_pass_conditions": len(rows), "conditions_added": len(decision["conditions"]),
        "abstained_amounts": sum(r["billed_amount"] is None for r in rows),
        "abstained_periods": sum(r["lookback_years"] is None for r in rows),
        "complete_document_references": 0, "reference_locked_for_evaluation": False,
        "predictions_generated": False, "f1_estimable": False}
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "base", "decisions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source.resolve(), args.base.resolve(), args.decisions.resolve(), args.output.resolve())))
