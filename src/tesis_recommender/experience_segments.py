"""Bounded, traceable experience segments, separate from the frozen v1 extractor.

This development parser does not determine eligibility or supply ranking inputs.
It preserves raw page slices and abstains instead of repairing uncertain OCR.
"""
from __future__ import annotations

import hashlib
import re

from .experience import LOOKBACK, MONEY, PLACEHOLDER, PREDICATE, parse_money

VERSION = "explicit_bidder_segments_v2"
PREFIX = r"(?:(?:[A-Z](?:\.\d+)*|\d+(?:\.\d+)*|[IVX]+)[.)]?\s+)?"
QUAL = re.compile(PREFIX + r"REQUISITOS DE CALIFICACI[ÓO]N(?: OBLIGATORIOS| FACULTATIVOS)?\s*:?[ \t]*", re.I)
EVAL = re.compile(PREFIX + r"FACTORES DE EVALUACI[ÓO]N(?: OBLIGATORIOS| FACULTATIVOS| PUNTAJE)?\s*:?[ \t]*", re.I)
ANNEX = re.compile(r"ANEXO\s+N[º°o]?\s*\d+[ \t]*", re.I)
SUBJECT = re.compile(PREFIX + r"EXPERIENCIA\s+(?:DEL|DE)\s+(POSTOR|PERSONAL\s+CLAVE)(?:\s+EN\s+LA\s+ESPECIALIDAD)?\s*:?(?:\s+\d+\s*puntos)?[ \t]*", re.I)
CHAPTER = re.compile(r"CAP[IÍ]TULO\s+[IVX\d]+[ \t]*", re.I)
# Another numbered, uppercase heading ends a candidate, but not its parent section.
OTHER = re.compile(r"(?:[A-Z](?:\.\d+)*|\d+(?:\.\d+)*)[.)]?\s+[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ /()\d.,:-]{3,}")
ACCREDITATION = re.compile(r"^[ \t]*acreditaci[oó]n\s*:", re.I | re.M)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def heading_events(pages):
    """Full-line headings only. Inline references never change section state."""
    events = []
    section, anchor = "unknown", None
    for page in pages:
        position = 0
        lines = page["text"].splitlines(keepends=True)
        for line in lines:
            clean = line.strip()
            role, subject = None, None
            if QUAL.fullmatch(clean):
                role = "qualification"
            elif EVAL.fullmatch(clean):
                role = "evaluation_factor"
            elif ANNEX.fullmatch(clean):
                role = "annex_form"
            elif CHAPTER.fullmatch(clean):
                role = "unknown"
            match = SUBJECT.fullmatch(clean)
            if role:
                section = role
                anchor = {"page": page["page"], "char_start": position,
                          "char_end": position + len(line), "evidence": line,
                          "page_text_sha256": digest(page["text"])}
            if match:
                subject = "bidder" if match.group(1).casefold() == "postor" else "key_personnel"
            if role or subject or OTHER.fullmatch(clean):
                events.append({"page": page["page"], "char_start": position,
                               "char_end": position + len(line), "subject": subject,
                               "section_role": section, "section_anchor": anchor})
            position += len(line)
    return events


def _spans(pages, event, following, max_pages, max_chars):
    start_index = next(i for i, p in enumerate(pages) if p["page"] == event["page"])
    spans, remaining, stop_reason = [], max_chars, "document_end"
    previous = None
    for page in pages[start_index:start_index + max_pages]:
        if previous is not None and page["page"] != previous + 1:
            stop_reason = "non_adjacent_page"
            break
        start = event["char_start"] if previous is None else 0
        end = len(page["text"])
        event_stop = following and following["page"] == page["page"]
        if event_stop:
            end = min(end, following["char_start"])
            stop_reason = "next_heading"
        if end - start > remaining:
            end = start + remaining
            stop_reason = "character_limit"
        raw = page["text"][start:end]
        accreditation = ACCREDITATION.search(raw)
        if accreditation:
            end = start + accreditation.start()
            raw = page["text"][start:end]
            stop_reason = "accreditation"
        if raw:
            span = {"page": page["page"], "char_start": start, "char_end": end,
                    "evidence": raw, "evidence_sha256": digest(raw),
                    "page_text_sha256": digest(page["text"]),
                    "extraction_method": page.get("extraction_method", "unspecified")}
            if page.get("ocr_used"):
                span.update(ocr_json_path=page.get("ocr_json_path"),
                            ocr_json_sha256=page.get("ocr_json_sha256"),
                            confidence_is_not_semantic_accuracy=True)
            spans.append(span)
            remaining -= len(raw)
        previous = page["page"]
        if accreditation or event_stop or remaining == 0:
            break
    else:
        if start_index + max_pages < len(pages):
            stop_reason = "page_limit"
    return spans, stop_reason


def extract_experience_segments(pages, *, max_pages=2, max_chars=5000):
    if not pages or len({p["page"] for p in pages}) != len(pages):
        raise ValueError("Empty pages or duplicate page identifiers")
    if max_pages not in (1, 2) or max_chars < 100:
        raise ValueError("Segments require one or two pages and at least 100 characters")
    pages = sorted(pages, key=lambda p: p["page"])
    for page in pages:
        if page.get("text_sha256") and digest(page["text"]) != page["text_sha256"]:
            raise ValueError("Page text hash mismatch")
    events = heading_events(pages)
    mentions = []
    for i, event in enumerate(events):
        if event["subject"] is None:
            continue
        following = events[i + 1] if i + 1 < len(events) else None
        spans, stop = _spans(pages, event, following, max_pages, max_chars)
        block = "\n".join(s["evidence"] for s in spans)
        row = {"page": event["page"], "subject": event["subject"],
               "section_role": event["section_role"], "section_anchor": event["section_anchor"],
               "evidence_spans": spans, "evidence": block, "evidence_sha256": digest(block),
               "assembly_separator": "\n", "segment_stop": stop, "rule_version": VERSION,
               "legal_eligibility_determined": False, "ranking_eligible": False,
               "historical_content_proven": False}
        predicate = PREDICATE.search(block)
        if row["subject"] != "bidder":
            row["status"] = "personnel_mention_not_bidder_requirement"
        elif row["section_role"] == "annex_form":
            row["status"] = "annex_form_not_requirement"
        elif not predicate:
            row["status"] = "mention_without_supported_explicit_threshold"
        else:
            tail = block[predicate.end():]
            money = MONEY.match(tail)
            amount = parse_money(money.group("amount")) if money else None
            if PLACEHOLDER.search(tail[:220]) or amount is None:
                row["status"] = "unresolved_or_template_threshold"
            elif row["section_role"] not in {"qualification", "evaluation_factor"}:
                row["status"] = "explicit_clause_with_unknown_section"
            else:
                lookback = LOOKBACK.search(block)
                currency = money.group("currency").upper()
                row.update(status="explicit_clause_extracted", billed_amount=amount,
                           amount_text=money.group("amount"), currency="USD" if currency.startswith(("US", "USD")) else "PEN",
                           lookback_years=int(lookback.group(1)) if lookback else None)
        mentions.append(row)
    extracted = [m for m in mentions if m["status"] == "explicit_clause_extracted"]
    qualification = [m for m in extracted if m["section_role"] == "qualification"]
    return {"rule_version": VERSION,
            "status": "requires_ocr_or_other_text_source" if all(not p["text"].strip() for p in pages) else "explicit_bidder_clause_found" if qualification else "no_supported_qualification_clause_detected",
            "qualification_clauses": qualification,
            "evaluation_factor_clauses": [m for m in extracted if m["section_role"] == "evaluation_factor"],
            "mentions": mentions, "heading_events": events,
            "limits": {"max_pages": max_pages, "max_chars": max_chars},
            "absence_of_experience_proven": False, "semantic_extraction_validated": False,
            "ranking_eligible": False}
