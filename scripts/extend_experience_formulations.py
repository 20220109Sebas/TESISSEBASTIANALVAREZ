"""Run the formulation extension on saved evidence without altering earlier outputs."""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tesis_recommender.experience_formulations import extract_formulations
from scripts.compare_experience_segments import validate_result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_ref(ref, pages):
    """Verify exact assembly and page slices without invoking extraction rules."""
    if not ref["source_spans"]:
        raise ValueError("Evidence has no source spans")
    parts = []
    for s in ref["source_spans"]:
        p = pages[s["page"]]
        if not 0 <= s["char_start"] < s["char_end"] <= len(p["text"]):
            raise ValueError("Invalid source offsets")
        raw = p["text"][s["char_start"]:s["char_end"]]
        if raw != s["evidence"] or hashlib.sha256(raw.encode()).hexdigest() != s["evidence_sha256"]:
            raise ValueError("Source fragment changed")
        if hashlib.sha256(p["text"].encode()).hexdigest() != s["page_text_sha256"]:
            raise ValueError("Source page changed")
        if s["extraction_method"] != p.get("extraction_method", "unspecified"):
            raise ValueError("Extraction method changed")
        if p.get("ocr_used") and (s.get("ocr_json_sha256") != p["ocr_json_sha256"] or s.get("ocr_json_path") != p["ocr_json_path"]):
            raise ValueError("OCR lineage changed")
        parts.append(raw)
    assembled = "\n".join(parts)
    if assembled != ref["evidence"] or hashlib.sha256(assembled.encode()).hexdigest() != ref["evidence_sha256"]:
        raise ValueError("Evidence assembly changed")


def verify_record(row, pages):
    """Check values independently against literal tokens, not semantic accuracy."""
    for key in ["evidence", "context", "window_evidence", "delimiter_evidence"]:
        if row.get(key) is not None:
            verify_ref(row[key], pages)
    context = row["context"]
    for key in ["evidence", "window_evidence", "delimiter_evidence"]:
        if row.get(key) is None:
            continue
        ref = row[key]
        start, end = ref["segment_char_start"], ref["segment_char_end"]
        if not 0 <= start < end <= len(context["evidence"]) or context["evidence"][start:end] != ref["evidence"]:
            raise ValueError("Candidate not mapped to its context")
        expected_spans, cursor = [], 0
        for i, c in enumerate(context["source_spans"]):
            lo, hi = max(start, cursor), min(end, cursor + len(c["evidence"]))
            if lo < hi:
                expected_spans.append((c["page"], c["char_start"] + lo - cursor, c["char_start"] + hi - cursor))
            cursor += len(c["evidence"]) + (1 if i + 1 < len(context["source_spans"]) else 0)
        if [(s["page"], s["char_start"], s["char_end"]) for s in ref["source_spans"]] != expected_spans:
            raise ValueError("Candidate source offsets differ from logical context offsets")
    raw = row["evidence"]["evidence"]
    if row["kind"] == "minimum_prior_service_count":
        token = re.search(r"(?:\((\d{1,3})\)|(\d{1,3}))\s+servicios\b", raw, re.I)
        if not token or row["minimum_services"] != int(token.group(1) or token.group(2)) or row["minimum_services"] <= 0:
            raise ValueError("Service count differs from source")
    elif row["kind"] == "billed_experience_fraction_of_offered_value":
        token = re.search(r"(\d{1,3}(?:[.,]\d{1,2})?)\s*%\s+del\s+valor\s+ofertado\b", raw, re.I)
        if not token or Decimal(token.group(1).replace(",", ".")) != Decimal(row["percentage"]) or not 0 < Decimal(row["percentage"]) <= 100:
            raise ValueError("Percentage differs from source")
        if row["reference_basis"] != "offered_value_not_procurement_budget" or row["absolute_amount"] is not None or row["currency"] is not None:
            raise ValueError("Relative requirement incorrectly converted to money")
    elif row["kind"] == "explicit_billed_amount_in_brackets":
        token = re.fullmatch(r"(S\s*/\.?|US\s*\$|USD|PEN)\s*(\d[\d.,]*)", raw, re.I)
        if not token:
            raise ValueError("Invalid monetary token")
        number = token.group(2)
        if re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d{2})?", number):
            normalized = number.replace(",", "")
        elif re.fullmatch(r"(?:\d+|\d{1,3}(?:\.\d{3})+),\d{2}", number):
            normalized = number.replace(".", "").replace(",", ".")
        else:
            raise ValueError("Ambiguous money literal")
        currency = "USD" if token.group(1).upper().startswith("US") else "PEN"
        if Decimal(normalized) != Decimal(row["billed_amount"]) or Decimal(normalized) <= 0 or row["currency"] != currency:
            raise ValueError("Amount or currency differs from source")
        delimiter = row["delimiter_evidence"]["evidence"]
        if not delimiter.lstrip().startswith("[") or not delimiter.endswith("]") or raw not in delimiter:
            raise ValueError("Missing bracket evidence")
        window = row["window_evidence"]
        if window is None:
            if row["lookback_years"] is not None:
                raise ValueError("Window has no evidence")
        else:
            token = re.search(r"\(?(\d{1,2})\)?\s+a[ñn]os\s+anteriores", window["evidence"], re.I)
            if not token or int(token.group(1)) != row["lookback_years"]:
                raise ValueError("Window differs from source")
        if row["subject"] != "bidder" or row["section_role"] not in {"qualification", "evaluation_factor"}:
            raise ValueError("Wrong subject or role for billed amount")
    else:
        raise ValueError("Unsupported candidate kind")
    if row["kind"] != "explicit_billed_amount_in_brackets" and (row["subject"] != "supplier" or row["section_role"] != "supplier_profile" or row["parent_section_role"] == "annex_form"):
        raise ValueError("Wrong supplier context")
    for key in ["ranking_eligible", "independently_validated", "historical_content_proven", "company_capacity_determined", "legal_eligibility_determined", "whole_document_conditions_complete", "single_universal_threshold_supported"]:
        if row[key] is not False:
            raise ValueError("Unexpected validation, ranking or company inference")
    if row["company_applicability"] != "not_determined" or row["status"] != "literal_candidate_development_only":
        raise ValueError("Unexpected applicability or status")


def build(input_dir, output):
    if output.exists():
        raise ValueError("Choose a new output directory")
    path = input_dir / "document_evidence.json"
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha(path) != manifest["output_sha256"][path.name]:
        raise ValueError("Evidence packet changed")
    inputs = {str(path): sha(path), str(manifest_path): sha(manifest_path)}
    documents = []
    for d in json.loads(path.read_text()):
        if sha(d["source"]["file_path"]) != d["source"]["sha256"]:
            raise ValueError("PDF source changed")
        inputs[d["source"]["file_path"]] = d["source"]["sha256"]
        validate_result(d["result"], d["pages"])
        for p in d["pages"]:
            if hashlib.sha256(p["text"].encode()).hexdigest() != p["text_sha256"]:
                raise ValueError("Page text changed")
            if p.get("ocr_used"):
                if sha(p["ocr_json_path"]) != p["ocr_json_sha256"]:
                    raise ValueError("Saved OCR changed")
                inputs[p["ocr_json_path"]] = p["ocr_json_sha256"]
        result = extract_formulations(d["pages"], d["result"]["mentions"])
        pages = {p["page"]: p for p in d["pages"]}
        for row in result["records"]:
            verify_record(row, pages)
        for warning in result["warnings"]:
            verify_ref(warning["evidence"], pages)
        documents.append({"opportunity_id": d["opportunity_id"], "source": d["source"], "result": result})
    # Regression checks against previously inspected development cases, never blind labels.
    review_path = ROOT / "artifacts/experience_formulation_review_v1/review.json"
    review_manifest = json.loads((review_path.parent / "manifest.json").read_text())
    if sha(review_path) != review_manifest["output_sha256"][review_path.name]:
        raise ValueError("Post-inspection review changed")
    inputs[str(review_path)] = sha(review_path)
    by_id = {d["opportunity_id"]: d for d in documents}
    for case in json.loads(review_path.read_text())["cases"]:
        d = by_id[case["opportunity_id"]]
        if d["source"]["sha256"] != case["source"]["sha256"]:
            raise ValueError("Reviewed PDF changed")
        for f in case["findings"]:
            rows = [r for r in d["result"]["records"] if r["evidence"]["source_spans"][0]["page"] == case["page"]]
            if f["kind"] == "minimum_prior_service_count":
                assert any(r.get("minimum_services") == f["observed_value"] for r in rows)
            elif f["kind"] == "billed_experience_fraction_of_offered_value":
                assert any(r.get("percentage") == str(f["observed_percentage"]) and r.get("absolute_amount") is None for r in rows)
            elif f["kind"] == "explicit_amount_with_bracket_delimiter":
                assert any(r.get("billed_amount") == f["observed_numeric_literal"] for r in rows)
            elif f["kind"] == "literal_lookback_window":
                assert any(r.get("lookback_years") == f["observed_years"] for r in rows)
    for folder in ["extraction_commitment_v1", "experiment_annual_v1", "experience_condition_variables_v4"]:
        p = ROOT / "artifacts" / folder / "manifest.json"
        inputs[str(p)] = sha(p)
    counts = Counter(r["kind"] for d in documents for r in d["result"]["records"])
    summary = {"documents_processed": len(documents), "documents_with_additional_candidates": sum(bool(d["result"]["records"]) for d in documents),
               "additional_literal_candidates": sum(counts.values()), "counts_by_kind": dict(counts),
               "warnings": sum(len(d["result"]["warnings"]) for d in documents),
               "trace_and_literal_verification_passed": True, "post_inspection_regression_matches": True,
               "reference_is_blind_or_expert": False, "f1_estimable": False,
               "rankings_changed": False, "original_parser_changed": False, "original_audit_changed": False,
               "new_downloads": False, "new_ocr_performed": False, "historical_content_proven": False}
    for p, expected in inputs.items():
        if sha(p) != expected:
            raise ValueError("Input changed during execution")
    output.mkdir(parents=True)
    for name, obj in [("formulation_candidates.json", documents), ("summary.json", summary)]:
        (output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    code = [Path(__file__), ROOT / "src/tesis_recommender/experience_formulations.py", ROOT / "src/tesis_recommender/experience_conditions.py", ROOT / "src/tesis_recommender/experience.py", ROOT / "src/tesis_recommender/experience_segments.py", ROOT / "scripts/compare_experience_segments.py"]
    (output / "manifest.json").write_text(json.dumps({"input_sha256": inputs, "code_sha256": {str(p): sha(p) for p in code},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/experience_document_evidence_v5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output)))
