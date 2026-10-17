"""Save conditional experience variables as development evidence, not ranking inputs."""
import argparse
import hashlib
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tesis_recommender.experience_conditions import represent_experience_clause
from scripts.compare_experience_segments import validate_result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_variable(variable, clause, pages):
    """Check values and all nested evidence without invoking the branch parser."""
    by_page = {p["page"]: p for p in pages}
    def verify_range(ref):
        if ref is None:
            return
        expected = clause["evidence"][ref["segment_char_start"]:ref["segment_char_end"]]
        if expected != ref["evidence"] or hashlib.sha256(expected.encode()).hexdigest() != ref["evidence_sha256"]:
            raise ValueError("Logical variable evidence mismatch")
        for s in ref["source_spans"]:
            p = by_page[s["page"]]
            raw = p["text"][s["char_start"]:s["char_end"]]
            if raw != s["evidence"] or hashlib.sha256(raw.encode()).hexdigest() != s["evidence_sha256"]:
                raise ValueError("Raw variable evidence mismatch")
            if p["text_sha256"] != s["page_text_sha256"]:
                raise ValueError("Variable source page hash mismatch")
    for b in variable["branches"]:
        for key in ("amount_evidence", "window_evidence", "paragraph_evidence"):
            verify_range(b[key])
        for key in ("condition", "additional_condition"):
            verify_range(b.get(key, {}).get("evidence"))
        token = re.fullmatch(r"\s*(S\s*/\.?|US\s*\$|USD|PEN)\s*(\d[\d.,]*)\s*", b["amount_evidence"]["evidence"], re.I)
        if not token:
            raise ValueError("Amount evidence has no complete supported currency token")
        number = token.group(2)
        if re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d{2})?", number):
            normalized = number.replace(",", "")
        elif re.fullmatch(r"(?:\d+|\d{1,3}(?:\.\d{3})+),\d{2}", number):
            normalized = number.replace(".", "").replace(",", ".")
        else:
            raise ValueError("Amount evidence is numerically ambiguous")
        currency = "USD" if token.group(1).upper().startswith("US") else "PEN"
        if Decimal(normalized) <= 0 or Decimal(normalized) != Decimal(b["billed_amount"]) or b["currency"] != currency:
            raise ValueError("Normalized amount or currency differs from source token")
        if b["window_evidence"] is None:
            if b["lookback_years"] is not None:
                raise ValueError("Window has a value without source evidence")
        else:
            window = re.search(r"\(?(\d{1,2})\)?\s+a[ñn]os\s+anteriores", b["window_evidence"]["evidence"], re.I)
            if window is None or int(window.group(1)) != b["lookback_years"]:
                raise ValueError("Normalized window differs from source evidence")
        if b["applicability"] != "not_determined" or b["ranking_eligible"]:
            raise ValueError("Unexpected company applicability or ranking adoption")
    for ref in variable.get("unassigned_monetary_evidence", []):
        verify_range(ref)
    if variable["selected_branch"] is not None or variable["ranking_eligible"] or variable["company_capacity_determined"]:
        raise ValueError("Variables must not select a company requirement branch")
    return True


def build(input_dir, output):
    if output.exists():
        raise ValueError("Choose a new variable output directory")
    m = json.loads((input_dir / "manifest.json").read_text())
    path = input_dir / "document_evidence.json"
    if sha(path) != m["output_sha256"][path.name]:
        raise ValueError("Evidence packet changed")
    inputs = {str(path): sha(path), str(input_dir / "manifest.json"): sha(input_dir / "manifest.json")}
    docs = json.loads(path.read_text())
    variables = []
    for d in docs:
        if sha(d["source"]["file_path"]) != d["source"]["sha256"]:
            raise ValueError("PDF source changed")
        inputs[d["source"]["file_path"]] = d["source"]["sha256"]
        validate_result(d["result"], d["pages"])
        for page in d["pages"]:
            if hashlib.sha256(page["text"].encode()).hexdigest() != page["text_sha256"]:
                raise ValueError("Page text changed")
            if page.get("ocr_used"):
                if sha(page["ocr_json_path"]) != page["ocr_json_sha256"]:
                    raise ValueError("Saved OCR changed")
                inputs[page["ocr_json_path"]] = page["ocr_json_sha256"]
        for key in ("qualification_clauses", "evaluation_factor_clauses"):
            for c in d["result"][key]:
                v = represent_experience_clause(c)
                verify_variable(v, c, d["pages"])
                variables.append({"opportunity_id": d["opportunity_id"], "source": d["source"], "variable": v})
    # Check the previously inspected case without treating it as a blind reference.
    review_path = ROOT / "artifacts/experience_native_visual_review_v3/review.json"
    review = json.loads(review_path.read_text())
    inputs[str(review_path)] = sha(review_path)
    known = next(c for c in review["cases"] if "observed_amounts" in c)
    row = next(r for r in variables if r["opportunity_id"] == known["opportunity_id"] and r["variable"]["source_clause_page"] == known["page"])
    if row["source"]["sha256"] != known["source_pdf_sha256"]:
        raise ValueError("Known development case PDF mismatch")
    observed = [(b["billed_amount"], b["currency"], b["lookback_years"]) for b in row["variable"]["branches"]]
    expected = [(b["billed_amount"], b["currency"], b["lookback_years"]) for b in known["observed_amounts"]]
    if observed != expected:
        raise ValueError("Known conditional development case differs")
    for folder in ("extraction_commitment_v1", "experiment_annual_v1"):
        p = ROOT / "artifacts" / folder / "manifest.json"
        inputs[str(p)] = sha(p)
    summary = {"documents_read": len(docs), "variables_from_qualification_clauses": sum(r["variable"]["section_role"] == "qualification" for r in variables),
               "variables_from_evaluation_factors": sum(r["variable"]["section_role"] == "evaluation_factor" for r in variables),
               "amount_branches": sum(len(r["variable"]["branches"]) for r in variables),
               "conditional_variables": sum(r["variable"]["status"] == "conditional_branches_represented" for r in variables),
               "variables_with_unresolved_warnings": sum(bool(r["variable"]["warnings"]) for r in variables),
               "known_post_inspection_case_matches": True, "trace_verification_passed": True,
               "reference_is_blind_or_expert": False, "f1_estimable": False, "new_ocr_performed": False,
               "rankings_changed": False, "original_audit_changed": False, "company_applicability_determined": False,
               "historical_content_proven": False}
    for p, expected in inputs.items():
        if sha(p) != expected:
            raise ValueError("Protected input changed")
    output.mkdir(parents=True)
    for name, value in (("experience_variables.json", variables), ("summary.json", summary)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    code = [Path(__file__), ROOT / "src/tesis_recommender/experience_conditions.py", ROOT / "src/tesis_recommender/experience.py",
            ROOT / "scripts/compare_experience_segments.py"]
    (output / "manifest.json").write_text(json.dumps({"input_sha256": inputs, "code_sha256": {str(p): sha(p) for p in code},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "artifacts/experience_document_evidence_v3")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_condition_variables_v2")
    a = p.parse_args()
    print(json.dumps(build(a.input, a.output)))
