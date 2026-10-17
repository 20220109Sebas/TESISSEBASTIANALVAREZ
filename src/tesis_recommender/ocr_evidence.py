"""Merge saved OCR evidence without overwriting native text or assuming accuracy."""
import hashlib


def merge_page(native, ocr=None):
    result = dict(native)
    raw = native["text"]
    if native.get("text_sha256") and hashlib.sha256(raw.encode()).hexdigest() != native["text_sha256"]:
        raise ValueError("Native text hash mismatch")
    if raw.strip():
        result.update(extraction_method="native_pdf_text", ocr_used=False)
        return result
    if ocr is None or ocr.get("status") != "ok":
        result.update(extraction_method="no_usable_text", ocr_used=False, absence_of_requirement_proven=False)
        return result
    if ocr["page"] != native["page"]:
        raise ValueError("OCR page mismatch")
    text = ocr["text"]
    if hashlib.sha256(text.encode()).hexdigest() != ocr["text_sha256"]:
        raise ValueError("OCR text hash mismatch")
    result.update(text=text, text_sha256=ocr["text_sha256"], extraction_method="apple_vision_local_ocr",
                  ocr_used=True, ocr_json_path=ocr["ocr_json_path"], ocr_json_sha256=ocr["ocr_json_sha256"],
                  low_confidence_lines=ocr["low_confidence_lines"],
                  confidence_is_not_semantic_accuracy=True, native_text_sha256=hashlib.sha256(raw.encode()).hexdigest())
    return result


def bind_clause_method(clause, pages):
    page = next((p for p in pages if p["page"] == clause["page"]), None)
    if page is None or page["text"][clause["char_start"]:clause["char_end"]] != clause["evidence"]:
        raise ValueError("Clause does not match its page evidence")
    result = dict(clause, extraction_method=page["extraction_method"], page_text_sha256=page["text_sha256"],
                  historical_content_proven=False, ranking_eligible=False)
    if page.get("ocr_used"):
        result.update(ocr_json_path=page["ocr_json_path"], ocr_json_sha256=page["ocr_json_sha256"],
                      confidence_is_not_semantic_accuracy=True, value_visually_verified=False)
    return result
