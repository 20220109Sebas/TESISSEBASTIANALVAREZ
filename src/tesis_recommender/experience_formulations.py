"""Additional literal formulations, isolated from frozen experience and ranking code.

Rules were developed after inspecting two source cases. Outputs are candidates,
not independently validated requirements or evidence of company eligibility.
"""
from __future__ import annotations

import re
from decimal import Decimal

from .experience import LOOKBACK, MONEY, PLACEHOLDER, PREDICATE, parse_money
from .experience_conditions import MICRO_SMALL, digest, source_range
from .experience_segments import OTHER, heading_events

VERSION = "experience_literal_formulations_v1"
SUPPLIER = re.compile(r"(?:\d+(?:\.\d+)*[.)]?\s+)?PERFIL\s+M[IÍ]NIMO\s+REQUERIDO\s+DEL\s+PROVEEDOR\s*:?[ \t]*", re.I)
COUNT = re.compile(r"Experiencia\s+no\s+menor\s+de\s+(?:(?P<word>[a-záéíóúñ]+)\s+)?\(?(?P<count>\d{1,3})\)?\s+servicios\b", re.I)
RELATIVE = re.compile(r"El\s+proveedor\s+deber[áa]\s+acreditar\s+como\s+m[ií]nimo\s+un\s+monto\s+facturado\s+acumulado\s+equivalente\s+al\s+(?P<percent>\d{1,3}(?:[.,]\d{1,2})?)\s*%\s+del\s+valor\s+ofertado\b", re.I)
STAFF = re.compile(r"(?:\bpersonal\b|\bcoordinador\b|\bequipo\s+de\s+trabajo\b|Dicha\s+experiencia\s+deber[áa]\s+ser\s+acreditada|Acreditaci[oó]n\s*:)", re.I)
WORDS = {"un": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
         "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}


def record(clause, kind, match, **values):
    result = {"rule_version": VERSION, "kind": kind, "subject": clause["subject"],
            "section_role": clause["section_role"], "status": "literal_candidate_development_only",
            "evidence": source_range(clause, match.start(), match.end()),
            "context": source_range(clause, 0, len(clause["evidence"])),
            "segment_stop": clause["segment_stop"], "company_applicability": "not_determined",
            "company_capacity_determined": False, "legal_eligibility_determined": False,
            "whole_document_conditions_complete": False, "ranking_eligible": False,
            "historical_content_proven": False, "independently_validated": False,
            "single_universal_threshold_supported": False, **values}
    if "parent_section_role" in clause:
        result["parent_section_role"] = clause["parent_section_role"]
    return result


def supplier_segments(pages, *, max_chars=5000):
    """Same-page provider sections only, ending before personnel or accreditation."""
    events = heading_events(pages)
    for page in pages:
        text = page["text"]
        cursor = 0
        starts, boundaries = [], []
        for line in text.splitlines(keepends=True):
            if SUPPLIER.fullmatch(line.strip()):
                starts.append(cursor)
            elif OTHER.fullmatch(line.strip()):
                boundaries.append(cursor)
            cursor += len(line)
        for start in starts:
            preceding = [e for e in events if (e["page"], e["char_start"]) <= (page["page"], start)]
            parent_role = preceding[-1]["section_role"] if preceding else "unknown"
            if parent_role == "annex_form":
                continue
            end = min(len(text), start + max_chars,
                      next((b for b in boundaries if b > start), len(text)))
            stop = "same_page_boundary_or_character_limit"
            staff = STAFF.search(text, start, end)
            if staff:
                end, stop = staff.start(), "personnel_or_accreditation_boundary"
            raw = text[start:end]
            span = {"page": page["page"], "char_start": start, "char_end": end,
                    "evidence": raw, "evidence_sha256": digest(raw),
                    "page_text_sha256": digest(text),
                    "extraction_method": page.get("extraction_method", "unspecified")}
            if page.get("ocr_used"):
                span.update(ocr_json_path=page.get("ocr_json_path"),
                            ocr_json_sha256=page.get("ocr_json_sha256"),
                            confidence_is_not_semantic_accuracy=True)
            yield {"subject": "supplier", "section_role": "supplier_profile",
                   "parent_section_role": parent_role,
                   "evidence": raw, "evidence_sha256": digest(raw),
                   "evidence_spans": [span], "segment_stop": stop}


def extract_formulations(pages, mentions):
    if not pages or len({p["page"] for p in pages}) != len(pages):
        raise ValueError("Empty pages or duplicate page identifiers")
    pages = sorted(pages, key=lambda p: p["page"])
    for page in pages:
        if page.get("text_sha256") and digest(page["text"]) != page["text_sha256"]:
            raise ValueError("Page text hash mismatch")
    records, warnings = [], []
    for clause in supplier_segments(pages):
        for match in COUNT.finditer(clause["evidence"]):
            count, word = int(match.group("count")), match.group("word")
            if count <= 0 or (word and WORDS.get(word.casefold()) != count):
                warnings.append({"reason": "unresolved_service_count", "evidence": source_range(clause, match.start(), match.end())})
                continue
            records.append(record(clause, "minimum_prior_service_count", match, minimum_services=count))
        for match in RELATIVE.finditer(clause["evidence"]):
            percentage = Decimal(match.group("percent").replace(",", "."))
            if not 0 < percentage <= 100:
                warnings.append({"reason": "unsupported_percentage_range", "evidence": source_range(clause, match.start(), match.end())})
                continue
            records.append(record(clause, "billed_experience_fraction_of_offered_value", match,
                                  percentage=format(percentage, "f"),
                                  reference_basis="offered_value_not_procurement_budget",
                                  absolute_amount=None, currency=None))
    for clause in mentions:
        text = clause["evidence"]
        if digest(text) != clause["evidence_sha256"] or "\n".join(s["evidence"] for s in clause["evidence_spans"]) != text:
            raise ValueError("Logical clause evidence mismatch")
        if clause["subject"] != "bidder" or clause["section_role"] not in {"qualification", "evaluation_factor"}:
            continue
        predicate = PREDICATE.search(text)
        if not predicate:
            continue
        end = next((m.start() for m in MICRO_SMALL.finditer(text)), len(text))
        opening = re.match(r"\[\s*", text[predicate.end():end])
        if not opening:
            continue
        money = MONEY.match(text, predicate.end() + opening.end())
        close = text.find("]", money.end(), min(end, money.end() + 240)) if money else -1
        amount = parse_money(money.group("amount")) if money else None
        if money is None or money.end() > end or close < 0 or amount is None or PLACEHOLDER.search(text[predicate.end():close + 1]):
            warnings.append({"reason": "unresolved_bracketed_amount", "evidence": source_range(clause, predicate.start(), end)})
            continue
        window = LOOKBACK.search(text, close + 1, end)
        records.append(record(clause, "explicit_billed_amount_in_brackets", money,
                              billed_amount=amount,
                              currency="USD" if money.group("currency").upper().startswith("US") else "PEN",
                              lookback_years=int(window.group(1)) if window else None,
                              window_evidence=source_range(clause, window.start(), window.end()) if window else None,
                              delimiter_evidence=source_range(clause, predicate.end(), close + 1),
                              warnings=["filled_delimiter_is_not_full_legal_validation",
                                        "conditional_and_accreditation_rules_not_resolved"]))
    return {"rule_version": VERSION, "records": records, "warnings": warnings,
            "f1_estimable": False, "reference_is_blind_or_expert": False,
            "rankings_changed": False, "absence_of_experience_proven": False}
