"""Extract bounded pages from the downloaded annex pilot; never feed frozen rankings."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pypdf

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(access, output):
    if output.exists():
        raise ValueError("Use a new PDF inspection directory")
    manifest = json.loads((access / "manifest.json").read_text())
    if sha(access / "access_results.json") != manifest["output_sha256"]["access_results.json"]:
        raise ValueError("Access record changed")
    plan = json.loads(Path(manifest["access_plan_path"]).read_text())
    if sha(manifest["access_plan_path"]) != manifest["access_plan_sha256"]:
        raise ValueError("Access plan changed")
    rows = json.loads((access / "access_results.json").read_text())
    docs = []
    for row in rows:
        if row["status"] != "downloaded_pdf":
            continue
        # BiddingDocuments is insufficient: source metadata may also label proposals this way.
        title = str(row.get("title", "")).casefold()
        if "propuesta" in title or not any(word in title for word in ("bases", "referencia", "requisito")):
            docs.append({"opportunity_id": row["opportunity_id"], "status": "excluded_document_role", "title": row.get("title")})
            continue
        path = Path(row["file_path"])
        if sha(path) != row["sha256"]:
            raise ValueError("Downloaded PDF changed")
        record = {"opportunity_id": row["opportunity_id"], "source": row,
                  "historical_content_proven": False, "ranking_eligible": False}
        try:
            reader = pypdf.PdfReader(path)
            if reader.is_encrypted:
                record.update(status="encrypted_pdf", pages=[])
            else:
                pages = []
                for index, page in enumerate(reader.pages[:plan["limits"]["max_pages_per_document"]]):
                    text = page.extract_text() or ""
                    pages.append({"page": index + 1, "text": text,
                                  "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                  "word_count": len(re.findall(r"\w+", text)),
                                  "contains_experience_word": "experiencia" in text.casefold()})
                record.update(status="text_extracted", total_pages=len(reader.pages), processed_pages=len(pages),
                              truncated_by_page_limit=len(pages) < len(reader.pages), pages=pages,
                              pages_with_little_text=sum(p["word_count"] < 10 for p in pages),
                              experience_pages=[p["page"] for p in pages if p["contains_experience_word"]])
        except (pypdf.errors.PdfReadError, ValueError, KeyError) as exc:
            record.update(status="pdf_parse_error", error=str(exc), pages=[])
        docs.append(record)
        print(json.dumps({"opportunity_id": record["opportunity_id"], "status": record["status"],
                          "pages": record.get("processed_pages"), "experience_pages": record.get("experience_pages")}), flush=True)
    output.mkdir(parents=True)
    data = {"pypdf_version": pypdf.__version__, "documents": docs, "ocr_performed": False,
            "human_annotation": False, "labels_read": False, "frozen_rankings_changed": False,
            "historical_pdf_content_proven": False}
    (output / "pdf_text.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"access_manifest_sha256": sha(access / "manifest.json"), "input_sha256": {r["file_path"]: r["sha256"] for r in rows if r["status"] == "downloaded_pdf"},
              "script_sha256": sha(Path(__file__)), "pypdf_version": pypdf.__version__,
              "output_sha256": {"pdf_text.json": sha(output / "pdf_text.json")}}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"documents": len(docs), "text_extracted": sum(d["status"] == "text_extracted" for d in docs),
            "pages": sum(d.get("processed_pages", 0) for d in docs), "historical_content_proven": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--access", type=Path, default=ROOT / "artifacts/experience_document_access_v2")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_pdf_inspection_v1")
    a = p.parse_args()
    print(json.dumps(inspect(a.access, a.output)))
