"""Post-inspection development candidates. Never repair OCR or set ranking inputs.

Rules were motivated by inspected sources. Their recovery count is not accuracy.
"""
import re

from .experience import MONEY, LOOKBACK, PLACEHOLDER, parse_money
from .experience_conditions import MICRO_SMALL, digest, source_range
from .experience_segments import heading_events, SUBJECT, QUAL, EVAL, ANNEX, CHAPTER, ACCREDITATION

VERSION = "experience_recovery_candidates_v1"
LITERAL = re.compile(r"monto\s+facturado\s+acumulado\s+equivalente\s+a\s*", re.I)
RELATIVE = re.compile(r"monto\s+facturado\s+acumulado\s+equivalente\s+a\s+"
                      r"(?P<factor>UNA|DOS|TRES)\s+(?:VEZ|VECES)\s+LA\s+CUANT[IÍ]A\s+"
                      r"DEL\s+PROCEDIMIENTO\s+DE\s+SELECCI[ÓO]N\b", re.I)
FACTORS = {"una": 1, "dos": 2, "tres": 3}


def bounded_context(page, event, max_chars=4000):
    """Same-page evidence ends at accreditation or explicit semantic headings.

    Does not treat stamp-like uppercase strings as headings. This is a new,
    conservative candidate view, not a replacement for frozen segmentation.
    """
    start = event["char_start"]
    end = min(len(page["text"]), start + max_chars)
    cursor = event["char_end"]
    for line in page["text"][cursor:end].splitlines(keepends=True):
        clean = line.strip()
        if any(rule.fullmatch(clean) for rule in (SUBJECT, QUAL, EVAL, ANNEX, CHAPTER)):
            end = cursor
            break
        cursor += len(line)
    raw = page["text"][start:end]
    boundary = ACCREDITATION.search(raw)
    if boundary:
        raw = raw[:boundary.start()]
    span = {"page": page["page"], "char_start": start, "char_end": start + len(raw),
            "evidence": raw, "evidence_sha256": digest(raw),
            "page_text_sha256": digest(page["text"]),
            "extraction_method": page.get("extraction_method", "unspecified")}
    if page.get("ocr_used"):
        span.update(ocr_json_path=page["ocr_json_path"], ocr_json_sha256=page["ocr_json_sha256"],
                    confidence_is_not_semantic_accuracy=True)
    return {"subject": "bidder", "section_role": event["section_role"],
            "evidence": raw, "evidence_sha256": digest(raw), "evidence_spans": [span]}


def recover_candidates(pages, frozen_result):
    for page in pages:
        if digest(page["text"]) != page["text_sha256"]:
            raise ValueError("Page text changed")
    accepted = {(c["page"], c["section_role"]) for key in
                ("qualification_clauses", "evaluation_factor_clauses") for c in frozen_result[key]}
    by_page = {p["page"]: p for p in pages}
    if len(by_page) != len(pages):
        raise ValueError("Duplicate pages")
    records = []
    for event in heading_events(pages):
        if event["subject"] != "bidder" or event["section_role"] not in {"qualification", "evaluation_factor"}:
            continue
        if (event["page"], event["section_role"]) in accepted:
            continue
        clause = bounded_context(by_page[event["page"]], event)
        text = clause["evidence"]
        # A following MYPE paragraph is not merged into the primary condition.
        condition = MICRO_SMALL.search(text)
        end = condition.start() if condition else len(text)
        relative = RELATIVE.search(text, 0, end)
        literal = LITERAL.search(text, 0, end)
        match, values = None, {}
        if relative:
            match = relative
            values = {"kind": "billed_experience_multiple_of_procedure_amount",
                      "multiplier": FACTORS[relative.group("factor").casefold()],
                      "reference_basis": "procedure_amount_not_offered_value",
                      "absolute_amount": None, "currency": None}
        elif literal:
            money = MONEY.match(text, literal.end(), end)
            amount = parse_money(money.group("amount")) if money else None
            if money and amount is not None and not PLACEHOLDER.search(text[literal.end():money.end()]):
                match = money
                values = {"kind": "literal_money_in_unresolved_bidder_context",
                          "billed_amount": amount,
                          "currency": "USD" if money.group("currency").upper().startswith("US") else "PEN"}
        if match is None:
            continue
        window = LOOKBACK.search(text, match.end(), end)
        records.append({"rule_version": VERSION, "page": event["page"],
            "subject": "bidder", "section_role": event["section_role"],
            "kind": values.pop("kind"), **values,
            "evidence": source_range(clause, match.start(), match.end()),
            "context": source_range(clause, 0, len(text)),
            "lookback_years": int(window.group(1)) if window else None,
            "window_evidence": source_range(clause, window.start(), window.end()) if window else None,
            "warnings": ["post_inspection_development_candidate", "predicate_or_formulation_not_accepted_by_frozen_rules",
                         "primary_paragraph_not_universal_threshold", "ocr_and_legal_review_required"],
            "status": "requires_separate_review", "ranking_eligible": False,
            "independently_validated": False, "historical_content_proven": False,
            "company_applicability": "not_determined", "absolute_amount_inferred": False,
            "whole_document_conditions_complete": False})
    return {"rule_version": VERSION, "records": records,
            "frozen_rules_modified": False, "f1_estimable": False,
            "reference_is_independent": False, "absence_of_experience_proven": False}
