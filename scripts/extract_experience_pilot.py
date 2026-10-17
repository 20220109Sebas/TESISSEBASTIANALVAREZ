"""Apply the new experience extension to the bounded annex inspection."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tesis_recommender.experience import extract_experience


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(inspection, output):
    if output.exists():
        raise ValueError("Use a new extraction directory")
    manifest = json.loads((inspection / "manifest.json").read_text())
    text_path = inspection / "pdf_text.json"
    if sha(text_path) != manifest["output_sha256"][text_path.name]:
        raise ValueError("Page text changed")
    data = json.loads(text_path.read_text())
    docs = []
    for doc in data["documents"]:
        if doc["status"] != "text_extracted":
            continue
        result = extract_experience(doc["pages"])
        docs.append({"opportunity_id": doc["opportunity_id"], "source": doc["source"], "total_pages": doc["total_pages"],
                     "processed_pages": doc["processed_pages"], "historical_content_proven": False, "result": result})
    summary = {"documents_processed": len(docs),
               "documents_with_qualification_clause": sum(bool(d["result"]["qualification_clauses"]) for d in docs),
               "qualification_clauses": sum(len(d["result"]["qualification_clauses"]) for d in docs),
               "evaluation_factor_clauses": sum(len(d["result"]["evaluation_factor_clauses"]) for d in docs),
               "documents_without_text": sum(d["result"]["status"] == "requires_ocr_or_other_text_source" for d in docs),
               "f1_estimable": False, "independent_reference_available": False,
               "historical_pdf_content_proven": False, "original_five_field_audit_modified": False,
               "ranking_modified": False, "labels_read": False, "development_pilot": True}
    output.mkdir(parents=True)
    for name, value in (("extraction.json", docs), ("summary.json", summary)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    code = ROOT / "src/tesis_recommender/experience.py"
    (output / "manifest.json").write_text(json.dumps({"inspection_sha256": sha(text_path),
        "script_sha256": sha(Path(__file__)), "module_sha256": sha(code),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inspection", type=Path, default=ROOT / "artifacts/experience_pdf_inspection_v1")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_extraction_pilot_v1")
    a = p.parse_args()
    print(json.dumps(run(a.inspection, a.output)))
