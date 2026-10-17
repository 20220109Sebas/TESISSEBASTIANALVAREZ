"""Attach page evidence to source-only review decisions, without model predictions."""
import argparse
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from scripts.register_browser_experience_sources import dump, sha, verify_hashes, protected_outputs

ROOT = Path(__file__).resolve().parents[1]


def check_literal_amount(raw, value):
    if value is None:
        return
    token = re.fullmatch(r"S/\s+([\d .,]+)", raw)
    if not token:
        raise ValueError("Unsupported literal reference money")
    digits = token.group(1).replace(" ", "")
    if re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)\.\d{2}", digits):
        digits = digits.replace(",", "")
    elif re.fullmatch(r"\d+,\d{2}", digits):
        digits = digits.replace(",", ".")
    elif not digits.isdigit():
        raise ValueError("Ambiguous reference amount")
    if Decimal(digits) <= 0 or Decimal(digits) != Decimal(value):
        raise ValueError("Reference amount differs from literal")


def ref(text, start, end, page):
    raw = text[start:end]
    return {"page": page, "char_start": start, "char_end": end, "literal": raw,
            "literal_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "page_text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def build(decisions_path, output):
    if output.exists():
        raise ValueError("Use a new draft directory")
    decision = json.loads(decisions_path.read_text())
    if decision["predictions_seen"] or decision["documents_complete"]:
        raise ValueError("Unexpected reference status")
    inputs = dict(protected_outputs())
    inputs[str(decisions_path)] = sha(decisions_path)
    sources = {}
    for path in sorted((ROOT / "artifacts/experience_holdout_inspection_v1").glob("batch_*/pdf_text.json")):
        manifest = json.loads((path.parent / "manifest.json").read_text())
        verify_hashes({str(path): manifest["output_sha256"][path.name]})
        inputs[str(path)] = sha(path)
        for doc in json.loads(path.read_text())["documents"]:
            sources[doc["opportunity_id"]] = doc
    rows = []
    for case in decision["conditions"]:
        doc = sources[case["opportunity_id"]]
        page = next(p for p in doc["pages"] if p["page"] == case["page"])
        text = page["text"]
        start = text.index(case["start_quote"])
        end = text.index(case["end_quote"], start)
        if text.count(case["start_quote"]) != 1:
            raise ValueError("Ambiguous context start")
        context = text[start:end]
        if context.count(case["amount_quote"]) != 1 or context.count(case["window_quote"]) != 1:
            raise ValueError("Reference field is missing or repeated in its branch")
        amount_at = start + context.index(case["amount_quote"])
        window_at = start + context.index(case["window_quote"])
        check_literal_amount(case["amount_quote"], case["billed_amount"])
        token = re.search(r"\((\d+)\) años anteriores", case["window_quote"])
        if not token or int(token.group(1)) != case["lookback_years"]:
            raise ValueError("Window differs from reference literal")
        image_path = ROOT / case["image"]
        pdf_path = Path(doc["source"]["file_path"]).resolve()
        inputs[str(image_path)] = sha(image_path)
        inputs[str(pdf_path)] = doc["source"]["sha256"]
        rows.append({"opportunity_id": case["opportunity_id"], "item": case["item"],
            "branch": case["branch"], "subject": "bidder", "section_role": "qualification",
            "kind": "required_billed_experience", "billed_amount": case["billed_amount"],
            "currency": case["currency"], "lookback_years": case["lookback_years"],
            "source_pdf_path": str(pdf_path), "source_pdf_sha256": doc["source"]["sha256"],
            "source_image_path": str(image_path), "source_image_sha256": sha(image_path),
            "context": ref(text, start, end, case["page"]),
            "amount_evidence": ref(text, amount_at, amount_at + len(case["amount_quote"]), case["page"]),
            "window_evidence": ref(text, window_at, window_at + len(case["window_quote"]), case["page"]),
            "source_reading_method": "native_pdf_text_and_full_page_visual_review",
            "warnings": case["warnings"], "review_status": "first_pass_partial_document_draft",
            "reviewer": "automatic_source_review", "predictions_seen": False,
            "second_source_only_pass_complete": False, "document_reference_complete": False,
            "company_applicability": "not_determined", "ranking_eligible": False,
            "human_annotation": False, "historical_content_proven": False})
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "reference_draft.json", rows)
    summary = {"documents_with_first_pass_conditions": len({r["opportunity_id"] for r in rows}),
               "first_pass_conditions": len(rows), "abstained_amounts": sum(r["billed_amount"] is None for r in rows),
               "complete_document_references": 0, "reference_locked_for_evaluation": False,
               "predictions_generated": False, "f1_estimable": False}
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
         "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.decisions.resolve(), args.output.resolve())))
