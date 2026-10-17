"""Coverage states for a fixed source queue, without inferring negative labels."""
from collections import Counter


def coverage_rows(queue, accesses, documents):
    if len({d["opportunity_id"] for d in queue}) != len(queue) or len({d["url"] for d in queue}) != len(queue):
        raise ValueError("Duplicate fixed queue identity")
    if len({d["opportunity_id"] for d in documents}) != len(documents):
        raise ValueError("Duplicate inspected opportunity")
    by_id = {d["opportunity_id"]: d for d in documents}
    queued = {d["opportunity_id"]: d for d in queue}
    for d in documents:
        if d["opportunity_id"] not in queued or d["source"]["url"] != queued[d["opportunity_id"]]["url"]:
            raise ValueError("Document does not match the fixed queue")
    by_url = {}
    priority = {"downloaded_pdf": 0, "too_large_partial_file": 1, "not_pdf": 2,
                "http_error": 3, "access_error": 4, "pending": 5}
    for r in accesses:
        by_url.setdefault(r["url"], []).append(r)
    rows = []
    for q in queue:
        attempts = by_url.get(q["url"], [])
        best = min(attempts, key=lambda r: priority.get(r["status"], 9)) if attempts else None
        state = best["status"] if best else "unattempted"
        d = by_id.get(q["opportunity_id"])
        if d and state != "downloaded_pdf":
            raise ValueError("Inspected document lacks a complete download")
        row = {"opportunity_id": q["opportunity_id"], "sample_number": q["sample_number"],
               "source_document_title": q["title"], "source_declared_format": q.get("format"),
               "source_url": q["url"], "download_state": state, "access_attempts_recorded": len(attempts),
               "complete_pdf_available": state == "downloaded_pdf", "qualification_candidates": 0,
               "evaluation_factor_candidates": 0, "historical_content_proven": False,
               "absence_of_requirement_proven": False, "semantic_extraction_validated": False,
               "ranking_eligible": False}
        if d is None:
            row.update(text_state="not_inspected", experience_state="not_evaluable_without_inspected_text")
        else:
            native = sum(p.get("extraction_method") == "native_pdf_text" and bool(p["text"].strip()) for p in d["pages"])
            ocr = sum(p.get("ocr_used", False) and bool(p["text"].strip()) for p in d["pages"])
            empty = sum(not p["text"].strip() for p in d["pages"])
            if not native and not ocr:
                text_state = "no_usable_text"
            elif ocr and native:
                text_state = "mixed_native_and_saved_ocr"
            elif ocr:
                text_state = "saved_ocr_only"
            else:
                text_state = "native_text_available"
            qual = len(d["result"]["qualification_clauses"])
            mentions = d["result"]["mentions"]
            row.update(text_state=text_state, pages_processed=len(d["pages"]), total_pages=d["total_pages"],
                       native_text_pages=native, saved_ocr_pages=ocr, empty_pages=empty,
                       page_limit_truncated=d["truncated_by_page_limit"],
                       pdf_source_path=d["source"]["file_path"], pdf_sha256=d["source"]["sha256"],
                       qualification_candidates=qual,
                       evaluation_factor_candidates=len(d["result"]["evaluation_factor_clauses"]),
                       mention_status_counts=dict(Counter(m["status"] for m in mentions)),
                       experience_state="supported_qualification_candidate_found" if qual else
                           "not_evaluable_no_usable_text" if not native and not ocr else
                           "mentions_without_supported_qualification" if mentions else
                           "no_supported_heading_detected_unknown_requirement_presence")
        rows.append(row)
    return rows


def coverage_summary(rows):
    return {"sample_size": len(rows), "download_states": dict(Counter(r["download_state"] for r in rows)),
            "text_states": dict(Counter(r["text_state"] for r in rows)),
            "experience_states": dict(Counter(r["experience_state"] for r in rows)),
            "complete_pdf_opportunities": sum(r["complete_pdf_available"] for r in rows),
            "opportunities_without_complete_pdf": sum(not r["complete_pdf_available"] for r in rows),
            "qualification_candidates": sum(r["qualification_candidates"] for r in rows),
            "evaluation_factor_candidates": sum(r["evaluation_factor_candidates"] for r in rows),
            "absence_of_requirement_proven": False, "f1_estimable": False,
            "negative_labels_created": False, "rankings_changed": False}
