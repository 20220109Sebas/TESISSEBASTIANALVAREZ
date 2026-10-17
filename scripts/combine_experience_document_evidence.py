"""Combine two bounded annex batches with saved OCR, keeping all former runs intact."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tesis_recommender.experience import extract_experience
from tesis_recommender.ocr_evidence import merge_page, bind_clause_method


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def combine(output, inspections, ocr_dir):
    if output.exists():
        raise ValueError("Use a new evidence directory")
    evidence = {}
    all_docs = []
    for folder in inspections:
        manifest = json.loads((folder / "manifest.json").read_text())
        path = folder / "pdf_text.json"
        if sha(path) != manifest["output_sha256"][path.name]:
            raise ValueError("PDF inspection changed")
        evidence[str(path)] = sha(path)
        evidence[str(folder / "manifest.json")] = sha(folder / "manifest.json")
        all_docs.extend(json.loads(path.read_text())["documents"])
    ocr_manifest = json.loads((ocr_dir / "manifest.json").read_text())
    for name, expected in ocr_manifest["output_sha256"].items():
        path = ocr_dir / name
        if sha(path) != expected:
            raise ValueError("OCR artifact changed")
        evidence[str(path)] = expected
    ocr = json.loads((ocr_dir / "ocr_results.json").read_text())
    ocr_by_page = {}
    for row in ocr:
        key = row["opportunity_id"], row["page"]
        if key in ocr_by_page:
            raise ValueError("Duplicate OCR page")
        ocr_by_page[key] = row
    docs, seen = [], set()
    for doc in all_docs:
        if doc["status"] != "text_extracted":
            continue
        oid = doc["opportunity_id"]
        if oid in seen:
            raise ValueError("Duplicate opportunity in evidence batches")
        seen.add(oid)
        if sha(doc["source"]["file_path"]) != doc["source"]["sha256"]:
            raise ValueError("PDF bytes changed")
        evidence[doc["source"]["file_path"]] = doc["source"]["sha256"]
        pages = []
        for native in doc["pages"]:
            row = ocr_by_page.get((oid, native["page"]))
            if row and row["source_pdf_sha256"] != doc["source"]["sha256"]:
                raise ValueError("OCR linked to another PDF")
            pages.append(merge_page(native, row))
        result = extract_experience(pages)
        for kind in ("qualification_clauses", "evaluation_factor_clauses", "mentions"):
            result[kind] = [bind_clause_method(c, pages) for c in result[kind]]
        docs.append({"opportunity_id": oid, "source": doc["source"], "total_pages": doc["total_pages"],
                     "processed_pages": len(pages), "truncated_by_page_limit": doc["truncated_by_page_limit"],
                     "pages": pages, "result": result, "historical_content_proven": False,
                     "ranking_eligible": False})
    summary = {"documents": len(docs), "sample_size": 50, "pages": sum(d["processed_pages"] for d in docs),
               "ocr_pages_used": sum(p["ocr_used"] for d in docs for p in d["pages"]),
               "documents_with_qualification_clause": sum(bool(d["result"]["qualification_clauses"]) for d in docs),
               "qualification_clauses": sum(len(d["result"]["qualification_clauses"]) for d in docs),
               "evaluation_factor_clauses": sum(len(d["result"]["evaluation_factor_clauses"]) for d in docs),
               "documents_truncated_by_page_limit": sum(d["truncated_by_page_limit"] for d in docs),
               "documents_without_supported_clause": sum(not d["result"]["qualification_clauses"] for d in docs),
               "f1_estimable": False, "development_sample_only": True, "labels_read": False,
               "original_audit_changed": False, "rankings_changed": False, "historical_content_proven": False}
    for path, expected in evidence.items():
        if sha(path) != expected:
            raise ValueError("Input changed during merge")
    output.mkdir(parents=True)
    for name, value in (("document_evidence.json", docs), ("summary.json", summary)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    modules = [ROOT / "src/tesis_recommender/experience.py", ROOT / "src/tesis_recommender/ocr_evidence.py"]
    (output / "manifest.json").write_text(json.dumps({"input_sha256": evidence,
        "code_sha256": {str(p): sha(p) for p in [Path(__file__), *modules]},
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_document_evidence_v2")
    p.add_argument("--ocr", type=Path, default=ROOT / "artifacts/experience_targeted_ocr_v1")
    a = p.parse_args()
    folders = [ROOT / "artifacts/experience_pdf_inspection_v1", ROOT / "artifacts/experience_pdf_inspection_v2"]
    print(json.dumps(combine(a.output, folders, a.ocr)))
