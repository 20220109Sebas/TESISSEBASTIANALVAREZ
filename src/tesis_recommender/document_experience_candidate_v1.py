"""Fixed text-input candidate combining extraction, provenance and unit gates.

No PDF access, OCR, reference values or metrics are accepted by this interface.
Missing content and unresolved unit identity stay visible. The input preparation
pipeline and evaluation protocol must still be fixed before a new sample.
"""
from __future__ import annotations

import re

from .document_experience_dev_v1 import digest
from .document_experience_dev_v2 import extract_development, verify_provenance
from .document_experience_units_dev_v1 import project_units

VERSION = "document_experience_text_candidate_v1"
MAX_DOCUMENTS, MAX_PAGES, MAX_BYTES = 50, 120, 10 * 1024 * 1024
METHODS = {"native_pdf_text", "apple_vision_local_ocr", "native_pdf_text_incomplete_ocr_unresolved"}


def validate_document(document):
    if document.get("ranking_eligible") is not False or document.get("historical_content_proven") is not False:
        raise ValueError("Candidate sources cannot enter the historical ranking")
    if not isinstance(document.get("opportunity_id"), str) or not document["opportunity_id"].startswith("ocds-"):
        raise ValueError("Source opportunity identity required")
    source = document["source"]
    if source.get("opportunity_id") != document["opportunity_id"] or source.get("status") != "downloaded_pdf":
        raise ValueError("Source identity/status differs")
    if not isinstance(source.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"]):
        raise ValueError("Complete source PDF checksum required")
    if type(source.get("bytes")) is not int or not 0 < source["bytes"] <= MAX_BYTES:
        raise ValueError("PDF size outside fixed input limit")
    pages = document["pages"]
    if not isinstance(pages, list) or not 0 < len(pages) <= MAX_PAGES:
        raise ValueError("Nonempty bounded page input required")
    if (type(document.get("total_pages")) is not int or document["total_pages"] != len(pages)
            or type(document.get("processed_pages")) is not int or document["processed_pages"] != len(pages)
            or document.get("truncated_by_page_limit") is not False):
        raise ValueError("Do not silently process a partial document")
    if [p["page"] for p in pages] != list(range(1, len(pages) + 1)) or any(type(p["page"]) is not int for p in pages):
        raise ValueError("Missing, reordered or repeated physical pages")
    for page in pages:
        for name in ("text", "native_text"):
            if not isinstance(page.get(name), str) or digest(page[name]) != page.get(name + "_sha256"):
                raise ValueError("Source text differs from its hash")
        flags = (page.get("ocr_selected"), page.get("ocr_used"), page.get("selected_ocr_unresolved"))
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("OCR status flags must be explicit booleans")
        method = page.get("extraction_method")
        if method not in METHODS:
            raise ValueError("Unsupported page text method")
        if method == "apple_vision_local_ocr":
            if flags != (True, True, False):
                raise ValueError("OCR method and status disagree")
            if (not isinstance(page.get("ocr_json_path"), str) or not page["ocr_json_path"]
                    or not re.fullmatch(r"[a-f0-9]{64}", str(page.get("ocr_json_sha256", "")))
                    or page.get("ocr_text") != page["text"] or page.get("ocr_text_sha256") != page["text_sha256"]):
                raise ValueError("OCR lineage is incomplete")
        else:
            expected = (False, False, False) if method == "native_pdf_text" else (True, False, True)
            if flags != expected or page["text"] != page["native_text"] or page["text_sha256"] != page["native_text_sha256"]:
                raise ValueError("Native text or unresolved OCR lost its original lineage")
    return True


def predict_documents(documents):
    if not isinstance(documents, list) or not 0 < len(documents) <= MAX_DOCUMENTS:
        raise ValueError("Provide one to fifty complete source documents")
    if len({doc["opportunity_id"] for doc in documents}) != len(documents):
        raise ValueError("Exactly one fixed source document per opportunity is required")
    extracted, coverage = [], []
    for document in documents:
        validate_document(document)
        result = extract_development(document["pages"], max_pages=2, max_chars=6000)
        verify_provenance(result, document["pages"])
        extracted.append({"opportunity_id": document["opportunity_id"], "source_pdf_sha256": document["source"]["sha256"],
                          "result": result})
        coverage.append({"opportunity_id": document["opportunity_id"], "physical_pages": len(document["pages"]),
            "empty_text_pages": [p["page"] for p in document["pages"] if not p["text"].strip()],
            "unresolved_ocr_pages": [p["page"] for p in document["pages"] if p["selected_ocr_unresolved"]],
            "all_document_content_proven_recovered": False, "absence_of_requirements_proven": False})
    units = project_units(extracted)
    return {"version": VERSION, "stage": "fixed_candidate_source_only_no_metrics",
            "document_outputs": extracted, "occurrence_projection": units, "coverage": coverage,
            "internal_development_metadata_preserved": True,
            "number_words_semantic_consistency_proven": False,
            "reference_or_expected_values_used": False, "metrics_computed": False,
            "pdf_downloads_performed": False, "new_ocr_performed": False, "ranking_eligible": False}
