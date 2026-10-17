"""Run the new segment parser on saved pages without changing former results."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tesis_recommender.experience_segments import extract_experience_segments


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_result(result, pages):
    by_page = {p["page"]: p for p in pages}
    for row in result["mentions"]:
        pieces = []
        for span in row["evidence_spans"]:
            p = by_page[span["page"]]
            raw = p["text"][span["char_start"]:span["char_end"]]
            if raw != span["evidence"] or hashlib.sha256(raw.encode()).hexdigest() != span["evidence_sha256"]:
                raise ValueError("Evidence slice mismatch")
            if hashlib.sha256(p["text"].encode()).hexdigest() != span["page_text_sha256"]:
                raise ValueError("Evidence page mismatch")
            pieces.append(raw)
        if "\n".join(pieces) != row["evidence"]:
            raise ValueError("Evidence assembly mismatch")
        anchor = row["section_anchor"]
        if anchor:
            p = by_page[anchor["page"]]
            if p["text"][anchor["char_start"]:anchor["char_end"]] != anchor["evidence"]:
                raise ValueError("Section anchor mismatch")
    return True


def check_cases(reference, documents):
    docs = {d["opportunity_id"]: d for d in documents}
    checked = []
    for case in reference["cases"]:
        d = docs[case["opportunity_id"]]
        expected = case["expected"]
        if d["source"]["sha256"] != case["source"]["sha256"]:
            raise ValueError("Reference refers to a different PDF")
        if expected["subject"] is None:
            events = [e for e in d["result"]["heading_events"] if e["page"] <= case["page"]]
            actual = {"section_role": events[-1]["section_role"] if events else "unknown"}
            passed = actual["section_role"] == expected["section_role"]
        else:
            candidates = [r for r in d["result"]["mentions"] if r["page"] == case["page"] and r["subject"] == expected["subject"]]
            actual = [{k: r.get(k) for k in expected} for r in candidates]
            passed = any(all(r.get(k) == v for k, v in expected.items()) for r in candidates)
        checked.append({"case_id": case["case_id"], "passed": passed, "expected": expected, "actual": actual})
    return checked


def compare(input_dir, reference_dir, output):
    if output.exists():
        raise ValueError("Choose a new segment output")
    protected = [ROOT / "artifacts/experiment_annual_v1/manifest.json",
                 ROOT / "artifacts/extraction_commitment_v1/manifest.json",
                 ROOT / "artifacts/experience_document_evidence_v2/manifest.json"]
    paths = [input_dir / "document_evidence.json", reference_dir / "reference_cases.json", *protected]
    snapshots = {str(p): sha(p) for p in paths}
    for directory, name in ((input_dir, "document_evidence.json"), (reference_dir, "reference_cases.json")):
        manifest = json.loads((directory / "manifest.json").read_text())
        if sha(directory / name) != manifest["output_sha256"][name]:
            raise ValueError("Input manifest mismatch")
    reference = json.loads((reference_dir / "reference_cases.json").read_text())
    original = json.loads((input_dir / "document_evidence.json").read_text())
    docs, changes = [], []
    for d in original:
        pdf = Path(d["source"]["file_path"])
        if sha(pdf) != d["source"]["sha256"]:
            raise ValueError("PDF changed")
        snapshots[str(pdf)] = sha(pdf)
        result = extract_experience_segments(d["pages"])
        validate_result(result, d["pages"])
        docs.append({"opportunity_id": d["opportunity_id"], "source": d["source"], "result": result})
        changes.append({"opportunity_id": d["opportunity_id"],
            "v1_qualification_clauses": len(d["result"]["qualification_clauses"]),
            "v2_qualification_clauses": len(result["qualification_clauses"]),
            "v1_factor_clauses": len(d["result"]["evaluation_factor_clauses"]),
            "v2_factor_clauses": len(result["evaluation_factor_clauses"]),
            "v1_mentions": len(d["result"]["mentions"]), "v2_heading_candidates": len(result["mentions"])})
    checks = check_cases(reference, docs)
    if not all(c["passed"] for c in checks):
        raise ValueError(json.dumps(checks, ensure_ascii=False))
    summary = {"documents": len(docs), "pages": sum(len(d["pages"]) for d in original),
               "qualification_clauses": sum(len(d["result"]["qualification_clauses"]) for d in docs),
               "evaluation_factor_clauses": sum(len(d["result"]["evaluation_factor_clauses"]) for d in docs),
               "multi_page_extracted_clauses": sum(len(c["evidence_spans"]) > 1 for d in docs for key in ("qualification_clauses", "evaluation_factor_clauses") for c in d["result"][key]),
               "development_cases": len(checks), "development_cases_passed": sum(c["passed"] for c in checks),
               "trace_validation_passed": True, "f1_estimable": False,
               "reference_is_blind_or_expert": False, "rankings_changed": False,
               "historical_content_proven": False, "original_audit_changed": False,
               "comparison": changes}
    for p, expected in snapshots.items():
        if sha(p) != expected:
            raise ValueError("Protected input changed")
    output.mkdir(parents=True)
    for name, value in (("document_segments.json", docs), ("development_checks.json", checks), ("summary.json", summary)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    modules = [Path(__file__), ROOT / "src/tesis_recommender/experience_segments.py", ROOT / "src/tesis_recommender/experience.py"]
    (output / "manifest.json").write_text(json.dumps({"input_sha256": snapshots,
        "code_sha256": {str(p): sha(p) for p in modules},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "artifacts/experience_document_evidence_v2")
    p.add_argument("--reference", type=Path, default=ROOT / "artifacts/experience_segment_reference_v1")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_segments_v2")
    a = p.parse_args()
    print(json.dumps(compare(a.input, a.reference, a.output), ensure_ascii=False))
