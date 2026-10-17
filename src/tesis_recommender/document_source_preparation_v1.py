"""Source-only native/OCR preparation: no requirement detector or truth values."""
import hashlib
import re
from collections import defaultdict

MAX_PAGES, MAX_BYTES = 120, 10 * 1024 * 1024
MAX_PAGE_CHARS, MAX_DOC_CHARS = 200000, 2000000
OCR_WORD_THRESHOLD, IMAGE_PIXELS, FORM_DEPTH = 50, 600, 8


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def raster_sizes(resources, seen=None, depth=0):
    if depth > FORM_DEPTH:
        raise ValueError("Image form nesting exceeds fixed bound")
    seen = set() if seen is None else seen
    resources = resources.get_object() if resources else {}
    objects = resources.get("/XObject")
    objects = objects.get_object() if objects else {}
    found = []
    for value in objects.values():
        obj = value.get_object()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if obj.get("/Subtype") == "/Image":
            found.append([int(obj.get("/Width", 0)), int(obj.get("/Height", 0))])
        elif obj.get("/Subtype") == "/Form":
            found.extend(raster_sizes(obj.get("/Resources"), seen, depth + 1))
    return found


def prepare_native(reader, source):
    base = {"opportunity_id": source["opportunity_id"], "source": dict(source), "pages": [],
            "historical_content_proven": False, "ranking_eligible": False}
    if reader.is_encrypted:
        return dict(base, status="encrypted_pdf_not_evaluable")
    total = len(reader.pages)
    if not 0 < total <= MAX_PAGES:
        return dict(base, status="physical_page_count_outside_limit", total_pages=total)
    pages, chars = [], 0
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        chars += len(text)
        if len(text) > MAX_PAGE_CHARS or chars > MAX_DOC_CHARS:
            return dict(base, status="native_text_size_outside_limit", total_pages=total)
        sizes = raster_sizes(page.get("/Resources"))
        words = len(re.findall(r"\w+", text))
        selected = words < OCR_WORD_THRESHOLD or any(w >= IMAGE_PIXELS and h >= IMAGE_PIXELS for w, h in sizes)
        pages.append({"page": number, "text": text, "text_sha256": text_sha(text),
                      "word_count": words, "image_pixel_sizes": sizes, "selected_for_ocr": selected})
    return dict(base, status="complete_native_view", total_pages=total, processed_pages=total,
                truncated_by_page_limit=False, pages=pages)


def check_ocr(raw, page, os_version_string):
    if (type(raw.get("page")) is not int or raw.get("page") != page
            or raw.get("engine") != "Apple Vision VNRecognizeTextRequest"
            or type(raw.get("revision")) is not int or raw.get("revision") != 3 or raw.get("languages") != ["es-ES", "en-US"]
            or raw.get("languageCorrection") is not False or raw.get("osVersion") != os_version_string):
        raise ValueError("OCR engine/settings/runtime differ from frozen plan")
    if not isinstance(raw.get("text"), str) or not isinstance(raw.get("lines"), list):
        raise ValueError("Malformed OCR output")
    if any(not isinstance(line, dict) or not isinstance(line.get("text"), str) for line in raw["lines"]):
        raise ValueError("Invalid OCR line text")
    if raw["text"] != "\n".join(line["text"] for line in raw["lines"]):
        raise ValueError("OCR text lost line assembly")
    if len(raw["text"]) > MAX_PAGE_CHARS:
        raise ValueError("OCR text exceeds fixed page size limit")
    return True


def merge_page(native, ocr=None):
    if text_sha(native["text"]) != native["text_sha256"]:
        raise ValueError("Native text changed")
    selected = native["selected_for_ocr"]
    result = dict(native, native_text=native["text"], native_text_sha256=native["text_sha256"],
                  ocr_selected=selected, ocr_used=False, absence_of_requirements_proven=False)
    if not selected:
        if ocr is not None:
            raise ValueError("Unplanned OCR overlay")
        return dict(result, extraction_method="native_pdf_text", selected_ocr_unresolved=False)
    if not ocr or ocr.get("status") != "ok" or not ocr.get("text", "").strip():
        return dict(result, extraction_method="native_pdf_text_incomplete_ocr_unresolved", selected_ocr_unresolved=True)
    if ocr["page"] != native["page"] or text_sha(ocr["text"]) != ocr["text_sha256"]:
        raise ValueError("OCR overlay page or hash differs")
    if (not isinstance(ocr.get("ocr_json_path"), str) or not ocr["ocr_json_path"]
            or not re.fullmatch(r"[a-f0-9]{64}", str(ocr.get("ocr_json_sha256", "")))):
        raise ValueError("Missing OCR JSON lineage")
    return dict(result, text=ocr["text"], text_sha256=ocr["text_sha256"], ocr_used=True,
                extraction_method="apple_vision_local_ocr", selected_ocr_unresolved=False,
                ocr_text=ocr["text"], ocr_text_sha256=ocr["text_sha256"],
                ocr_json_path=ocr["ocr_json_path"], ocr_json_sha256=ocr["ocr_json_sha256"],
                confidence_is_not_semantic_accuracy=True)


def compose_documents(native_documents, ocr_rows):
    planned = {(d["opportunity_id"], p["page"]): d["source"]["sha256"] for d in native_documents
               if d["status"] == "complete_native_view" for p in d["pages"] if p["selected_for_ocr"]}
    by_page = {}
    for row in ocr_rows:
        key = row["opportunity_id"], row["page"]
        if key in by_page or key not in planned or row["source_pdf_sha256"] != planned[key]:
            raise ValueError("OCR ledger has duplicate, unplanned or different-source page")
        by_page[key] = row
    if set(by_page) != set(planned):
        raise ValueError("Every planned OCR page needs an explicit outcome, including unattempted pages")
    documents = []
    for native in native_documents:
        if native["status"] != "complete_native_view":
            continue
        pages = [merge_page(p, by_page.get((native["opportunity_id"], p["page"]))) for p in native["pages"]]
        if sum(len(p["text"]) for p in pages) > MAX_DOC_CHARS:
            raise ValueError("Composed text exceeds document bound; do not truncate")
        documents.append(dict(native, pages=pages, predictions_generated=False,
                              all_document_content_proven_recovered=False))
    return documents


def combine_batches(selected_cases, batches):
    """Retain the full selection; detect equal PDF bytes even across batches."""
    selected = {r["opportunity_id"]: r for r in selected_cases}
    if len(selected) != len(selected_cases):
        raise ValueError("Repeated selected opportunity")
    documents, ledger, pdf_ids = {}, {}, defaultdict(set)
    for batch in batches:
        for row in batch["coverage_ledger"]:
            ocid = row["opportunity_id"]
            if ocid in ledger or ocid not in selected:
                raise ValueError("Repeated or unselected source ledger entry")
            ledger[ocid] = dict(row)
        for doc in batch["source_documents"]:
            ocid = doc["opportunity_id"]
            if ocid in documents or ocid not in selected or doc.get("predictions_generated") is not False:
                raise ValueError("Repeated/unselected document or unexpected inference")
            if doc.get("ranking_eligible") is not False or doc.get("historical_content_proven") is not False:
                raise ValueError("Source preparation cannot become a historical ranking input")
            documents[ocid] = doc
            pdf_ids[doc["source"]["sha256"]].add(ocid)
    if any(ocid not in ledger for ocid in documents):
        raise ValueError("Source view lost its coverage ledger")
    if any(ledger[ocid]["source_status"] != "complete_native_view" for ocid in documents):
        raise ValueError("Complete source view conflicts with its coverage status")
    duplicate_ids = {ocid for ids in pdf_ids.values() if len(ids) > 1 for ocid in ids}
    full_ledger = []
    for row in selected_cases:
        ocid = row["opportunity_id"]
        status = ("duplicate_pdf_across_batches_no_replacement" if ocid in duplicate_ids
                  else ledger[ocid]["source_status"] if ocid in ledger
                  else "source_access_not_attempted" if row["status"] == "eligible_initial_pdf_selected"
                  else row["status"])
        full_ledger.append({"opportunity_id": ocid, "sample_number": row.get("sample_number"),
                            "source_status": status, "absence_of_requirements_proven": False,
                            "ranking_eligible": False})
    return {"source_documents": [documents[r["opportunity_id"]] for r in selected_cases
                                 if r["opportunity_id"] in documents and r["opportunity_id"] not in duplicate_ids],
            "duplicate_documents_preserved": [documents[ocid] for ocid in sorted(duplicate_ids)],
            "coverage_ledger": full_ledger, "silent_replacements_performed": False,
            "reference_created": False, "model_inference_performed": False}
