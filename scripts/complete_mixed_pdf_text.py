"""Complete saved mixed PDF text locally, preserving native pages and frozen rules.

The raster-size screen is fixed before OCR and does not inspect relevance labels
or experience detections. OCR confidence is not scientific extraction accuracy.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.register_browser_experience_sources import dump, protected_outputs, verify_hashes
from scripts.fetch_experience_documents import sha
from scripts.fetch_pending_experience_sources import check_code

ALLOWED = {"ocds-dgv273-seacev3-1116792", "ocds-dgv273-seacev3-2025-200379-2",
           "ocds-dgv273-seacev3-1137572", "ocds-dgv273-seacev3-1115544"}


def raster_sizes(resources, seen=None, depth=0):
    """Inspect image metadata, including forms, without decoding or selecting by text."""
    if depth > 8:
        raise ValueError("PDF form nesting exceeds the inspection bound")
    seen = set() if seen is None else seen
    resources = resources.get_object() if resources else {}
    objects = resources.get("/XObject")
    objects = objects.get_object() if objects else {}
    found = []
    for value in objects.values():
        obj = value.get_object()
        marker = id(obj)
        if marker in seen:
            continue
        seen.add(marker)
        if obj.get("/Subtype") == "/Image":
            found.append([int(obj.get("/Width", 0)), int(obj.get("/Height", 0))])
        elif obj.get("/Subtype") == "/Form":
            found.extend(raster_sizes(obj.get("/Resources"), seen, depth + 1))
    return found


def needs_ocr(sizes):
    return any(width >= 600 and height >= 600 for width, height in sizes)


def complete_page(native, ocr, selected):
    """Use a full-page OCR view where planned, while retaining the exact native text."""
    result = dict(native, native_text=native["text"], native_text_sha256=native["text_sha256"],
                  ocr_selected=selected, ocr_used=False, absence_of_requirements_proven=False)
    if hashlib.sha256(native["text"].encode()).hexdigest() != native["text_sha256"]:
        raise ValueError("Native text changed")
    if not selected:
        result.update(extraction_method="native_pdf_text", selected_ocr_unresolved=False)
        return result
    if ocr is None or ocr.get("status") != "ok" or not ocr.get("text", "").strip():
        result.update(extraction_method="native_pdf_text_incomplete_ocr_unresolved",
                      selected_ocr_unresolved=True)
        return result
    if ocr["page"] != native["page"] or hashlib.sha256(ocr["text"].encode()).hexdigest() != ocr["text_sha256"]:
        raise ValueError("OCR page or text hash differs")
    result.update(text=ocr["text"], text_sha256=ocr["text_sha256"],
                  word_count=len(re.findall(r"\w+", ocr["text"])),
                  extraction_method="apple_vision_local_ocr", ocr_used=True,
                  ocr_text=ocr["text"], ocr_text_sha256=ocr["text_sha256"],
                  ocr_json_path=ocr["ocr_json_path"], ocr_json_sha256=ocr["ocr_json_sha256"],
                  low_confidence_lines=ocr["low_confidence_lines"],
                  confidence_is_not_semantic_accuracy=True, selected_ocr_unresolved=False)
    return result


def prepare(output, binary):
    import pypdf
    if output.exists():
        raise ValueError("Use a new OCR directory")
    frozen_path = ROOT / "artifacts/experience_substitute_plan_v1/pending_plan.json"
    frozen = json.loads(frozen_path.read_text())
    check_code(frozen)
    inputs = dict(protected_outputs())
    inputs.update(frozen["frozen_rule_and_runner_sha256"])
    inputs.update({str(frozen_path): sha(frozen_path), str(binary): sha(binary),
                   str(ROOT / "scripts/vision_pdf_ocr.swift"): sha(ROOT / "scripts/vision_pdf_ocr.swift"),
                   str(Path(__file__)): sha(__file__)})
    inspections = [ROOT / "artifacts/experience_pasco_retry_inspection_v1",
                   ROOT / "artifacts/experience_substitute_browser_inspection_v2/batch_02",
                   ROOT / "artifacts/experience_substitute_browser_inspection_v2/batch_03"]
    documents = []
    for directory in inspections:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        verify_hashes({str(directory / p): h for p, h in manifest["output_sha256"].items()})
        text_path = directory / "pdf_text.json"
        inputs[str(manifest_path)] = sha(manifest_path)
        inputs[str(text_path)] = sha(text_path)
        for document in json.loads(text_path.read_text())["documents"]:
            if document["opportunity_id"] not in ALLOWED:
                continue
            source = document["source"]
            inputs[source["file_path"]] = source["sha256"]
            verify_hashes({source["file_path"]: source["sha256"]})
            if source["ranking_eligible"] or not source["source_substitution"] or document.get("truncated_by_page_limit"):
                raise ValueError("Unexpected source scope or truncated native inspection")
            reader = pypdf.PdfReader(source["file_path"])
            if len(reader.pages) != len(document["pages"]) or len(reader.pages) > 120:
                raise ValueError("Physical pages differ or exceed the existing limit")
            page_metadata = []
            for number, page in enumerate(reader.pages, 1):
                sizes = raster_sizes(page.get("/Resources"))
                page_metadata.append({"page": number, "image_pixel_sizes": sizes,
                                      "selected_for_ocr": needs_ocr(sizes)})
            documents.append({"opportunity_id": document["opportunity_id"], "source": source,
                "inspection_path": str(text_path), "total_pages": len(reader.pages),
                "page_screen": page_metadata,
                "selected_pages": [r["page"] for r in page_metadata if r["selected_for_ocr"]]})
    if {d["opportunity_id"] for d in documents} != ALLOWED or len(documents) != 4:
        raise ValueError("The complete fixed four-document scope is required")
    verify_hashes(inputs)
    plan = {"purpose": "authorized_local_text_completion_separate_development",
            "input_sha256": inputs, "documents": documents,
            "binary_path": str(binary), "pypdf_version": pypdf.__version__,
            "page_selection": "Every page containing an image XObject of at least 600 by 600 pixels, including nested forms. No text or model detections used.",
            "limits": {"documents": 4, "max_pages_per_document": 120, "workers": 1,
                       "timeout_per_page_seconds": 45, "stop_after_consecutive_errors": 2},
            "settings": {"dpi": 180, "vision_revision": 3, "recognition_level": "accurate",
                         "languages": ["es-ES", "en-US"], "language_correction": False},
            "external_ocr_service": False, "ranking_eligible": False,
            "historical_content_proven": False, "human_annotation": False}
    output.mkdir(parents=True)
    dump(output / "plan.json", plan)
    dump(output / "plan_lock.json", {"plan_sha256": sha(output / "plan.json")})
    return {"documents": len(documents), "ocr_pages_planned": sum(len(d["selected_pages"]) for d in documents),
            "pages_by_document": {d["opportunity_id"]: d["selected_pages"] for d in documents}}


def load_plan(output):
    plan_path = output / "plan.json"
    lock = json.loads((output / "plan_lock.json").read_text())
    verify_hashes({str(plan_path): lock["plan_sha256"]})
    plan = json.loads(plan_path.read_text())
    verify_hashes(plan["input_sha256"])
    return plan


def run(output):
    plan = load_plan(output)
    pages_dir = output / "pages"
    if pages_dir.exists():
        raise ValueError("OCR already started. Preserve the run and use a new plan to repeat")
    pages_dir.mkdir()
    rows, errors, stop = [], 0, False
    for document in plan["documents"]:
        for number in document["selected_pages"]:
            path = pages_dir / f'{document["source"]["sha256"][:16]}_{number:03d}.json'
            row = {"opportunity_id": document["opportunity_id"], "page": number,
                   "source_pdf_sha256": document["source"]["sha256"], "method": "apple_vision_local_ocr"}
            try:
                subprocess.run([plan["binary_path"], document["source"]["file_path"], str(number), str(path)],
                    check=True, capture_output=True, text=True, timeout=45)
                result = json.loads(path.read_text())
                if (result["page"] != number or result["revision"] != 3 or result["languageCorrection"]
                    or result["languages"] != plan["settings"]["languages"]):
                    raise ValueError("OCR settings differ from the locked plan")
                row.update(status="ok", text=result["text"],
                    text_sha256=hashlib.sha256(result["text"].encode()).hexdigest(),
                    ocr_json_path=str(path), ocr_json_sha256=sha(path), line_count=len(result["lines"]),
                    low_confidence_lines=sum(line["confidence"] < .8 for line in result["lines"]),
                    engine=result["engine"], languages=result["languages"], os_version=result["osVersion"],
                    raster_pixels=[result["pixelWidth"], result["pixelHeight"]])
                errors = 0
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
                row.update(status="ocr_error", error=str(exc), stderr=getattr(exc, "stderr", None))
                errors += 1
            rows.append(row)
            dump(output / "ocr_results.json", rows)
            if len(rows) == 1 or len(rows) % 5 == 0:
                print(json.dumps({"pages_attempted": len(rows), "pages_ok": sum(r["status"] == "ok" for r in rows)}), flush=True)
            if errors >= 2:
                stop = True
                break
        if stop:
            break
    planned = sum(len(d["selected_pages"]) for d in plan["documents"])
    summary = {"planned_pages": planned, "pages_attempted": len(rows),
               "pages_ok": sum(r["status"] == "ok" for r in rows),
               "complete": len(rows) == planned and all(r["status"] == "ok" for r in rows),
               "low_confidence_lines": sum(r.get("low_confidence_lines", 0) for r in rows),
               "confidence_is_not_semantic_accuracy": True, "new_ocr_performed": bool(rows),
               "historical_content_proven": False, "ranking_eligible": False, "human_annotation": False}
    load_plan(output)
    dump(output / "summary.json", summary)
    dump(output / "manifest.json", {"input_sha256": plan["input_sha256"],
        "plan_sha256": sha(output / "plan.json"),
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return summary


def compose(ocr_dir, output):
    if output.exists():
        raise ValueError("Use a new completed-text directory")
    plan = load_plan(ocr_dir)
    manifest_path = ocr_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_hashes({str(ocr_dir / p): h for p, h in manifest["output_sha256"].items()})
    rows = json.loads((ocr_dir / "ocr_results.json").read_text())
    by_page = {(r["opportunity_id"], r["page"]): r for r in rows}
    if len(by_page) != len(rows):
        raise ValueError("Duplicate OCR result")
    expected = {(d["opportunity_id"], n) for d in plan["documents"] for n in d["selected_pages"]}
    if set(by_page) != expected or not all(r["status"] == "ok" for r in rows):
        raise ValueError("Selected OCR pages incomplete or failed")
    inputs = dict(plan["input_sha256"])
    inputs.update({str(ocr_dir / p): h for p, h in manifest["output_sha256"].items()})
    inputs[str(manifest_path)] = sha(manifest_path)
    documents = []
    for selected in plan["documents"]:
        native = next(d for d in json.loads(Path(selected["inspection_path"]).read_text())["documents"]
                      if d["opportunity_id"] == selected["opportunity_id"])
        pages = [complete_page(p, by_page.get((native["opportunity_id"], p["page"])),
                               p["page"] in selected["selected_pages"]) for p in native["pages"]]
        documents.append(dict(native, pages=pages, selected_ocr_pages=selected["selected_pages"],
                              ranking_eligible=False, historical_content_proven=False))
    result = {"documents": len(documents), "physical_pages": sum(len(d["pages"]) for d in documents),
              "ocr_pages_used": sum(p["ocr_used"] for d in documents for p in d["pages"]),
              "selected_ocr_unresolved": sum(p["selected_ocr_unresolved"] for d in documents for p in d["pages"]),
              "native_text_preserved": True, "text_accuracy_independently_validated": False,
              "all_document_content_proven_recovered": False, "ranking_eligible": False}
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / "pdf_text.json", {"documents": documents, "ocr_performed": True,
         "human_annotation": False, "historical_content_proven": False, "ranking_inputs_modified": False})
    dump(output / "summary.json", result)
    dump(output / "manifest.json", {"input_sha256": inputs,
         "code_sha256": {str(Path(__file__)): sha(__file__)},
         "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "compose"))
    parser.add_argument("--ocr-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "tmp/vision_ocr_build/vision-pdf-ocr")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    folder = args.ocr_dir.resolve()
    if args.mode == "prepare":
        result = prepare(folder, args.binary.resolve())
    elif args.mode == "run":
        result = run(folder)
    else:
        if args.output is None:
            parser.error("compose requires --output")
        result = compose(folder, args.output.resolve())
    print(json.dumps(result, ensure_ascii=False))
