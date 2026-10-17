"""Observed-document development only; never replaces frozen experience rules.

Parse complete monetary tokens and explicit local ITEM headings. All evidence is
mapped to unchanged archived page text. Unknowns remain unknown; no eligibility,
ranking adoption, independent accuracy or unseen-test claim is made here.
"""
from __future__ import annotations

import hashlib
import re
from decimal import Decimal

from .experience import LOOKBACK, PREDICATE
from .experience_conditions import MICRO_SMALL
from .experience_segments import heading_events

VERSION = "document_experience_development_v1"
SPACE = r"[ \t\r\n\u00a0\u202f]"
CURRENCY = re.compile(r"(?:S[ \t]*/\.?|US[ \t]*\$|USD|PEN)(?![A-Za-z])", re.I)
NUMBER_RUN = re.compile(r"[0-9](?:[0-9.,'´’ \t\r\n\u00a0\u202f]*[0-9])?")
ITEM = re.compile(r"^[ \t]*[ÍI]TEM\s*(?:N[°ºo]\.?\s*)?([0-9]{1,3})\s*(?:[:.\-–]\s*[^\n]*)?[ \t]*$", re.I | re.M)
MYPE_VARIANT = re.compile(
    r"En\s+(?:el\s+)?caso\s+(?:(?:de\s+)?postores\s+que\s+declaren|(?:el\s+)?postor\s+que\s+declare)"
    r"[\s\S]{0,240}?condici[oó]n\s+(?:de\s+)?micro\s+(?:y|o)\s+peque[ñn]a\s+empresa[.,]?\s*"
    r"se\s+acredita(?:r[áa])?\s+una\s+experiencia\s+de\s*", re.I)
BOUNDARY = re.compile(r"^[ \t]*(?:Acreditaci[oó]n\s*:|Se\s+consideran\s+servicios\b)", re.I | re.M)
BARRIER = re.compile(r"^[ \t]*(?:[A-Z][.)]?\s+)?(?:CAPACIDAD\s+T[ÉE]CNICA|FORMACI[ÓO]N\s+ACAD[ÉE]MICA|EQUIPAMIENTO\s+ESTRAT[ÉE]GICO)\b[^\n]*", re.I | re.M)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def parse_number(raw):
    """Exact two-decimal canonicalization; no prefix acceptance or rounding.

    Spaces between regular thousand groups are supported. A broken space within
    a punctuation-delimited number is collapsed only if the resulting complete
    punctuation grouping is valid. Malformed unpunctuated groups abstain.
    """
    if not raw or not re.fullmatch(r"[0-9][0-9.,'´’ \t\r\n\u00a0\u202f]*[0-9]|[0-9]", raw):
        return None
    decimal, fraction = None, "00"
    punctuation = [i for i, c in enumerate(raw) if c in ".,"]
    integer = raw
    if punctuation:
        last = punctuation[-1]
        suffix = raw[last + 1:]
        if len(suffix) == 2 and suffix.isascii() and suffix.isdigit():
            decimal, integer, fraction = raw[last], raw[:last], suffix
        elif raw[last] == ".":
            # A bare dotted three-decimal form is not silently reinterpreted.
            return None
    if not integer:
        return None
    punctuation_group = "," if decimal != "," else "."
    if decimal and decimal in integer:
        return None
    if re.fullmatch(r"[0-9]+", integer):
        digits = integer
    else:
        allowed = re.escape(punctuation_group) + "'´’"
        if re.search("[" + allowed + "]", integer):
            compact = re.sub(SPACE + "+", "", integer)
            if not re.fullmatch(r"[0-9]{1,3}(?:[" + allowed + r"][0-9]{3})+", compact):
                return None
            digits = re.sub("[" + allowed + "]", "", compact)
        elif re.fullmatch(r"[0-9]{1,3}(?:" + SPACE + r"+[0-9]{3})+", integer):
            digits = re.sub(SPACE + "+", "", integer)
        else:
            return None
    amount = Decimal(digits + "." + fraction)
    return format(amount, ".2f") if amount > 0 else None


def money_at(text, start):
    cursor = start
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if text[cursor:cursor + 1] == "[":
        cursor += 1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
    currency = CURRENCY.match(text, cursor)
    if not currency:
        return {"billed_amount": None, "currency": None, "status": "no_literal_absolute_money",
                "start": cursor, "end": cursor, "currency_end": cursor, "warnings": []}
    code = "USD" if currency.group().upper().startswith("US") else "PEN"
    numeric_start = currency.end()
    while numeric_start < len(text) and text[numeric_start].isspace():
        numeric_start += 1
    match = NUMBER_RUN.match(text, numeric_start)
    if not match:
        return {"billed_amount": None, "currency": code, "status": "missing_or_unsupported_number",
                "start": cursor, "end": numeric_start, "currency_end": currency.end(), "warnings": []}
    raw = match.group()
    amount = parse_number(raw)
    tail = text[match.end():]
    if re.match(r"[A-Za-z'´’]", tail) or re.match(r"\s*[/+−-]\s*[0-9]", tail):
        amount = None
    warnings = []
    # A limited explicit centavos guard, not a Spanish words/eligibility parser.
    words = re.match(r"\s*\(([^)]{0,900})\)", tail)
    if words and amount is not None:
        cents = re.search(r"\b(?:con|y)\s+([0-9]{2})\s*/\s*100\b", words.group(1), re.I)
        if cents and amount.rsplit(".", 1)[1] != cents.group(1):
            amount = None
            warnings.append("explicit_centavos_conflict_digits_vs_parenthetical")
    return {"billed_amount": amount, "currency": code,
            "status": "complete_token_normalized" if amount is not None else "ambiguous_number_abstained",
            "start": cursor, "end": match.end(), "currency_end": currency.end(), "warnings": warnings}


def page_index(pages):
    if not pages or len({p["page"] for p in pages}) != len(pages):
        raise ValueError("Nonempty unique pages required")
    if [p["page"] for p in pages] != sorted(p["page"] for p in pages):
        raise ValueError("Pages must be in physical reading order")
    if [p["page"] for p in pages] != list(range(pages[0]["page"], pages[0]["page"] + len(pages))):
        raise ValueError("Do not inherit context across missing physical pages")
    cursor, layout = 0, []
    for page in pages:
        if digest(page["text"]) != page["text_sha256"]:
            raise ValueError("Archived page text changed")
        layout.append((cursor, cursor + len(page["text"]), page))
        cursor += len(page["text"]) + 1
    return "\n".join(p["text"] for p in pages), layout


def evidence(text, layout, start, end):
    if not 0 <= start < end <= len(text):
        raise ValueError("Invalid source range")
    # Do not include a synthetic inter-page separator without its next page.
    while start < end and not any(a <= start < b for a, b, _ in layout):
        start += 1
    while start < end and not any(a < end <= b for a, b, _ in layout):
        end -= 1
    spans = []
    for a, b, page in layout:
        lo, hi = max(a, start), min(b, end)
        if lo < hi:
            raw = page["text"][lo - a:hi - a]
            span = {"page": page["page"], "char_start": lo - a, "char_end": hi - a,
                    "evidence": raw, "evidence_sha256": digest(raw), "page_text_sha256": page["text_sha256"],
                    "extraction_method": page.get("extraction_method", "unspecified")}
            for key in ("ocr_json_path", "ocr_json_sha256"):
                if key in page:
                    span[key] = page[key]
            spans.append(span)
    if not spans:
        raise ValueError("No archived source fragment")
    return {"evidence": text[start:end], "evidence_sha256": digest(text[start:end]),
            "source_spans": spans, "assembly_separator": "\n"}


def extract_development(pages, *, max_pages=2, max_chars=6000):
    if max_pages != 2 or max_chars < 100:
        raise ValueError("This development version uses two-page bounded clauses")
    text, layout = page_index(pages)
    starts = {p["page"]: a for a, _, p in layout}
    timeline = []
    for event in heading_events(pages):
        anchor = event["section_anchor"]
        semantic = anchor and (anchor["page"], anchor["char_start"]) == (event["page"], event["char_start"])
        if event["subject"] is not None or semantic:
            timeline.append({"position": starts[event["page"]] + event["char_start"],
                "kind": "subject" if event["subject"] is not None else "role",
                "subject": event["subject"], "role": event["section_role"],
                "end": starts[event["page"]] + event["char_end"]})
    for pattern, kind in ((ITEM, "item"), (BARRIER, "barrier")):
        for match in pattern.finditer(text):
            timeline.append({"position": match.start(), "end": match.end(), "kind": kind,
                             "item": int(match.group(1)) if kind == "item" else None})
    timeline.sort(key=lambda e: (e["position"], e["kind"]))
    clauses = [(m, "general") for m in PREDICATE.finditer(text)]
    mype = {m.start(): m for regex in (MICRO_SMALL, MYPE_VARIANT) for m in regex.finditer(text)}
    clauses.extend((m, "mype") for m in mype.values())
    clauses.sort(key=lambda pair: pair[0].start())
    records, skipped = [], []
    role, subject, item, item_ref = "unknown", None, None, None
    t = 0
    for i, (clause, branch) in enumerate(clauses):
        while t < len(timeline) and timeline[t]["position"] <= clause.start():
            event = timeline[t]
            if event["kind"] in ("role", "subject"):
                role = event["role"]
                subject = event["subject"] if event["kind"] == "subject" else None
                item, item_ref = None, None
            elif event["kind"] == "barrier":
                subject, item, item_ref = None, None, None
            elif subject == "bidder" and role in ("qualification", "evaluation_factor"):
                if event["item"] > 0:
                    item = event["item"]
                    item_ref = evidence(text, layout, event["position"], event["end"])
            t += 1
        if subject != "bidder" or role not in ("qualification", "evaluation_factor"):
            skipped.append({"reason": "unsupported_subject_or_role_context", "role": role,
                            "subject": subject, "predicate": evidence(text, layout, clause.start(), clause.end())})
            continue
        p_index = next(j for j, (a, b, _) in enumerate(layout) if a <= clause.start() < b)
        page_end = layout[min(p_index + max_pages - 1, len(layout) - 1)][1]
        if p_index + 1 < len(layout) and layout[p_index + 1][2]["page"] != layout[p_index][2]["page"] + 1:
            page_end = layout[p_index][1]
        end = min(page_end, clause.start() + max_chars,
                  clauses[i + 1][0].start() if i + 1 < len(clauses) else len(text))
        next_boundary = next((e["position"] for e in timeline[t:] if e["position"] > clause.start()), end)
        end = min(end, next_boundary)
        boundary = BOUNDARY.search(text, clause.end(), end)
        if boundary:
            end = boundary.start()
        token = money_at(text[:end], clause.end())
        window = LOOKBACK.search(text, max(clause.end(), token["end"]), end)
        record = {"version": VERSION, "prediction_id": f"dev:{layout[p_index][2]['page']}:{clause.start()}:{branch}",
            "page": layout[p_index][2]["page"], "subject": "bidder", "section_role": role,
            "branch": branch, "item": item,
            "fields": {"billed_amount": token["billed_amount"], "currency": token["currency"],
                       "lookback_years": int(window.group(1)) if window else None},
            "amount_status": token["status"], "warnings": token["warnings"],
            "amount_evidence": evidence(text, layout, token["start"], token["end"]) if token["end"] > token["start"] else None,
            "currency_evidence": evidence(text, layout, token["start"], token["currency_end"]) if token["currency_end"] > token["start"] else None,
            "window_evidence": evidence(text, layout, window.start(), window.end()) if window else None,
            "item_evidence": copy_ref(item_ref), "clause_evidence": evidence(text, layout, clause.start(), end),
            "stage": "observed_source_development_not_unseen_validation", "company_applicability": "not_determined",
            "ranking_eligible": False, "historical_content_proven": False, "legal_eligibility_determined": False}
        records.append(record)
    return {"version": VERSION, "records": records, "skipped_contexts": skipped,
            "parameters": {"max_pages": max_pages, "max_chars": max_chars},
            "stage": "development_on_previously_inspected_sources", "f1_computed": False,
            "independent_semantic_accuracy_proven": False, "ranking_eligible": False,
            "automatic_deduplication_performed": False, "archived_text_modified": False}


def copy_ref(ref):
    if ref is None:
        return None
    return {**ref, "source_spans": [dict(s) for s in ref["source_spans"]]}


def verify_provenance(result, pages):
    indexed = {p["page"]: p for p in pages}
    if any(digest(p["text"]) != p["text_sha256"] for p in pages):
        raise ValueError("Archived page hash differs during provenance verification")
    def check(ref):
        if ref is None:
            return
        for span in ref["source_spans"]:
            page = indexed[span["page"]]
            if not 0 <= span["char_start"] < span["char_end"] <= len(page["text"]):
                raise ValueError("Invalid archived source offsets")
            raw = page["text"][span["char_start"]:span["char_end"]]
            if (raw != span["evidence"] or digest(raw) != span["evidence_sha256"]
                    or page["text_sha256"] != span["page_text_sha256"]
                    or span["extraction_method"] != page.get("extraction_method", "unspecified")
                    or any(span.get(k) != page.get(k) for k in ("ocr_json_path", "ocr_json_sha256"))):
                raise ValueError("Development evidence lost its archived source")
        if digest(ref["evidence"]) != ref["evidence_sha256"]:
            raise ValueError("Development evidence hash differs")
        if "\n".join(s["evidence"] for s in ref["source_spans"]) != ref["evidence"]:
            raise ValueError("Development evidence assembly differs")
    for row in result["records"]:
        for name in ("amount_evidence", "currency_evidence", "window_evidence", "item_evidence", "clause_evidence"):
            check(row[name])
    for row in result["skipped_contexts"]:
        check(row["predicate"])
    return True
