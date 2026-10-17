"""Append native PDF evidence to the isolated development packet, without new OCR."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tesis_recommender.experience_segments import extract_experience_segments
from tesis_recommender.ocr_evidence import merge_page
from scripts.compare_experience_segments import validate_result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(docs, added_ids, sample_size):
    return {"sample_size": sample_size, "complete_pdf_opportunities": len(docs),
            "opportunities_without_complete_pdf": sample_size - len(docs),
            "new_documents": len(added_ids), "pages_processed": sum(len(d["pages"]) for d in docs),
            "new_pages_processed": sum(len(d["pages"]) for d in docs if d["opportunity_id"] in added_ids),
            "native_pages_with_usable_text": sum(p["extraction_method"] == "native_pdf_text" for d in docs for p in d["pages"]),
            "previous_ocr_pages_retained": sum(p.get("ocr_used", False) for d in docs for p in d["pages"]),
            "pages_without_usable_text": sum(p["extraction_method"] == "no_usable_text" for d in docs for p in d["pages"]),
            "documents_with_no_usable_text": sum(all(not p["text"].strip() for p in d["pages"]) for d in docs),
            "page_limit_truncated_documents": sum(d["truncated_by_page_limit"] for d in docs),
            "qualification_clauses": sum(len(d["result"]["qualification_clauses"]) for d in docs),
            "new_qualification_clauses": sum(len(d["result"]["qualification_clauses"]) for d in docs if d["opportunity_id"] in added_ids),
            "evaluation_factor_clauses": sum(len(d["result"]["evaluation_factor_clauses"]) for d in docs),
            "new_ocr_performed": False, "f1_estimable": False, "labels_read": False,
            "historical_content_proven": False, "rankings_changed": False, "original_audit_changed": False}


def combine(base_dir, inspection_dirs, output):
    if output.exists():
        raise ValueError("Use a new cumulative evidence directory")
    inputs = {}
    def read_checked(directory, name):
        m = json.loads((directory / "manifest.json").read_text())
        path = directory / name
        if sha(path) != m["output_sha256"][name]:
            raise ValueError("Evidence input changed")
        inputs[str(path)] = sha(path)
        inputs[str(directory / "manifest.json")] = sha(directory / "manifest.json")
        return json.loads(path.read_text())
    docs = read_checked(base_dir, "document_evidence.json")
    seen = {d["opportunity_id"] for d in docs}
    if len(seen) != len(docs):
        raise ValueError("Duplicate base opportunity")
    added = set()
    for directory in inspection_dirs:
        for d in read_checked(directory, "pdf_text.json")["documents"]:
            if d["status"] != "text_extracted":
                continue
            if d["opportunity_id"] in seen:
                raise ValueError("New document overlaps existing opportunity")
            pages = [merge_page(p) for p in d["pages"]]
            docs.append({"opportunity_id": d["opportunity_id"], "source": d["source"],
                         "total_pages": d["total_pages"], "processed_pages": len(pages),
                         "truncated_by_page_limit": d["truncated_by_page_limit"], "pages": pages,
                         "historical_content_proven": False, "ranking_eligible": False})
            seen.add(d["opportunity_id"])
            added.add(d["opportunity_id"])
    queue_dir = ROOT / "artifacts/experience_annex_plan_v2"
    queue = read_checked(queue_dir, "queue.json")
    queue_by_id = {d["opportunity_id"]: d for d in queue}
    for d in docs:
        if d["opportunity_id"] not in queue_by_id or queue_by_id[d["opportunity_id"]]["url"] != d["source"]["url"]:
            raise ValueError("PDF not in the fixed source queue")
        if sha(d["source"]["file_path"]) != d["source"]["sha256"]:
            raise ValueError("PDF bytes changed")
        inputs[d["source"]["file_path"]] = d["source"]["sha256"]
        for page in d["pages"]:
            if page.get("ocr_used"):
                if sha(page["ocr_json_path"]) != page["ocr_json_sha256"]:
                    raise ValueError("Saved OCR changed")
                inputs[page["ocr_json_path"]] = page["ocr_json_sha256"]
        d["result"] = extract_experience_segments(d["pages"])
        validate_result(d["result"], d["pages"])
    for folder in ("extraction_commitment_v1", "experiment_annual_v1"):
        p = ROOT / "artifacts" / folder / "manifest.json"
        inputs[str(p)] = sha(p)
    summary = summarize(docs, added, len(queue))
    for p, expected in inputs.items():
        if sha(p) != expected:
            raise ValueError("Input changed during composition")
    output.mkdir(parents=True)
    for name, value in (("document_evidence.json", docs), ("summary.json", summary)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    code = [Path(__file__), ROOT / "src/tesis_recommender/experience_segments.py",
            ROOT / "src/tesis_recommender/experience.py", ROOT / "src/tesis_recommender/ocr_evidence.py",
            ROOT / "scripts/compare_experience_segments.py"]
    (output / "manifest.json").write_text(json.dumps({"input_sha256": inputs,
        "code_sha256": {str(p): sha(p) for p in code},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=ROOT / "artifacts/experience_document_evidence_v2")
    p.add_argument("--inspection", type=Path, action="append")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_document_evidence_v3")
    a = p.parse_args()
    print(json.dumps(combine(a.base, a.inspection or [ROOT / "artifacts/experience_pdf_inspection_v3"], a.output)))
