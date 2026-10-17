"""Conservative, evidence-first extraction of explicit bidder experience clauses.

Development extension only. No inference of legal eligibility or company capacity,
no connection to the frozen ranking, and no replacement for a labelled evaluation.
"""
from __future__ import annotations

import hashlib
import re
from decimal import Decimal

HEADING = re.compile(r"experiencia\s+(?:del|de)\s+(postor|personal\s+clave)(?:\s+en\s+la\s+especialidad)?", re.I)
QUALIFICATION = re.compile(r"requisitos\s+de\s+calificaci[oó]n", re.I)
EVALUATION = re.compile(r"factores\s+de\s+evaluaci[oó]n", re.I)
ANNEX = re.compile(r"(?:^|\n)\s*anexo\s+n[º°o]?\s*\d+", re.I)
PREDICATE = re.compile(r"el\s+postor\s+debe\s+acreditar\s+un\s+monto\s+facturado\s+acumulado\s+equivalente\s+a\s*", re.I)
MONEY = re.compile(r"(?P<currency>S\s*/\.?|US\s*\$|USD|PEN)\s*(?P<amount>\d[\d.,]*)", re.I)
LOOKBACK = re.compile(r"durante\s+los\s+(?:[a-záéíóúñ]+\s+)?\(?(\d{1,2})\)?\s+a[ñn]os\s+anteriores", re.I)
PLACEHOLDER = re.compile(r"\[(?:\s*CONSIGNAR|\s*INDICAR)|\[\s*[.…]{2,}", re.I)


def parse_money(value: str) -> str | None:
    """Accept only unambiguous digits or grouped amounts with two decimal places."""
    if re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d{2})?", value):
        normalized = value.replace(",", "")
    elif re.fullmatch(r"(?:\d+|\d{1,3}(?:\.\d{3})+),\d{2}", value):
        normalized = value.replace(".", "").replace(",", ".")
    else:
        return None
    amount = Decimal(normalized)
    return format(amount, "f") if amount > 0 else None


def extract_experience(pages: list[dict]) -> dict:
    if not pages or len({p["page"] for p in pages}) != len(pages):
        raise ValueError("Empty pages or duplicate page identifiers")
    section = "unknown"
    requirements, mentions = [], []
    for page in sorted(pages, key=lambda p: p["page"]):
        text = page["text"]
        # Identify section transitions in reading order, not from future pages.
        transitions = [(m.start(), "qualification") for m in QUALIFICATION.finditer(text)]
        transitions += [(m.start(), "evaluation_factor") for m in EVALUATION.finditer(text)]
        transitions += [(m.start(), "annex_form") for m in ANNEX.finditer(text)]
        transitions.sort()
        headings = list(HEADING.finditer(text))
        cursor = 0
        for index, heading in enumerate(headings):
            while cursor < len(transitions) and transitions[cursor][0] < heading.start():
                section = transitions[cursor][1]; cursor += 1
            end = min(len(text), heading.end() + 4000,
                      headings[index + 1].start() if index + 1 < len(headings) else len(text))
            # An accreditation paragraph is not the threshold clause.
            stop = re.search(r"acreditaci[oó]n\s*:", text[heading.end():end], re.I)
            if stop:
                end = heading.end() + stop.start()
            block = text[heading.start():end]
            subject = "bidder" if heading.group(1).casefold() == "postor" else "key_personnel"
            row = {"page": page["page"], "subject": subject, "section_role": section,
                   "char_start": heading.start(), "char_end": end, "evidence": block,
                   "evidence_sha256": hashlib.sha256(block.encode()).hexdigest()}
            predicate = PREDICATE.search(block)
            if subject != "bidder":
                row["status"] = "personnel_mention_not_bidder_requirement"
            elif section == "annex_form":
                row["status"] = "annex_form_not_requirement"
            elif not predicate:
                row["status"] = "mention_without_supported_explicit_threshold"
            else:
                tail = block[predicate.end():]
                money = MONEY.match(tail)
                amount = parse_money(money.group("amount")) if money else None
                if PLACEHOLDER.search(tail[:220]) or money is None or amount is None:
                    row["status"] = "unresolved_or_template_threshold"
                elif section not in {"qualification", "evaluation_factor"}:
                    row["status"] = "explicit_clause_with_unknown_section"
                else:
                    lookback = LOOKBACK.search(block)
                    row.update(status="explicit_clause_extracted", minimum_billed_amount=amount,
                               amount_text=money.group("amount"), currency="PEN" if money.group("currency").upper().startswith(("S", "PEN")) and not money.group("currency").upper().startswith("US") else "USD",
                               lookback_years=int(lookback.group(1)) if lookback else None,
                               legal_eligibility_determined=False, ranking_eligible=False,
                               rule_version="explicit_bidder_billed_amount_v1")
                    requirements.append(row)
            mentions.append(row)
        for _, new_section in transitions[cursor:]:
            section = new_section
    qualification = [r for r in requirements if r["section_role"] == "qualification"]
    sparse = [p["page"] for p in pages if len(re.findall(r"\w+", p["text"])) < 50]
    all_empty = all(not p["text"].strip() for p in pages)
    return {"status": "requires_ocr_or_other_text_source" if all_empty else "explicit_bidder_clause_found" if qualification else "no_supported_qualification_clause_detected",
            "qualification_clauses": qualification,
            "evaluation_factor_clauses": [r for r in requirements if r["section_role"] == "evaluation_factor"],
            "mentions": mentions, "pages_with_sparse_text": sparse,
            "absence_of_experience_proven": False, "semantic_extraction_validated": False,
            "ranking_eligible": False}
