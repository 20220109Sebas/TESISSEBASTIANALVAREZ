"""Represent supported monetary experience paragraphs without selecting a legal branch.

An isolated development layer over saved segments. It does not compare company
capacity, overwrite tender amounts, or provide inputs to the frozen recommender.
"""
from __future__ import annotations

import hashlib
import re

from .experience import LOOKBACK, MONEY, PLACEHOLDER, PREDICATE, parse_money

VERSION = "bidder_experience_branch_representation_v1"
MICRO_SMALL = re.compile(
    r"En\s+el\s+caso\s+de\s+postores\s+que\s+declaren\s+en\s+el\s+Anexo\s+N[º°o]?\s*\d+\s+"
    r"tener\s+la\s+condici[oó]n\s+de\s+micro\s+y\s+peque[ñn]a\s+empresa\s*,\s*"
    r"se\s+acredita\s+una\s+experiencia\s+de\s*", re.I)
CONSORTIUM = re.compile(
    r"En\s+el\s+caso\s+de\s+consorcios,\s+todos\s+los\s+integrantes\s+deben\s+contar\s+con\s+"
    r"la\s+condici[oó]n\s+de\s+micro\s+y\s+peque[ñn]a\s+empresa\.", re.I)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_range(clause, start, end):
    """Map a logical segment range back to unchanged page slices."""
    text = clause["evidence"]
    if not 0 <= start < end <= len(text):
        raise ValueError("Invalid logical evidence range")
    spans, cursor = [], 0
    for i, span in enumerate(clause["evidence_spans"]):
        raw = span["evidence"]
        if digest(raw) != span["evidence_sha256"]:
            raise ValueError("Source span hash mismatch")
        lo, hi = max(start, cursor), min(end, cursor + len(raw))
        if lo < hi:
            fragment = raw[lo - cursor:hi - cursor]
            mapped = {"page": span["page"], "char_start": span["char_start"] + lo - cursor,
                      "char_end": span["char_start"] + hi - cursor,
                      "evidence": fragment, "evidence_sha256": digest(fragment),
                      "page_text_sha256": span["page_text_sha256"],
                      "extraction_method": span["extraction_method"]}
            for key in ("ocr_json_path", "ocr_json_sha256", "confidence_is_not_semantic_accuracy"):
                if key in span:
                    mapped[key] = span[key]
            spans.append(mapped)
        cursor += len(raw) + (1 if i + 1 < len(clause["evidence_spans"]) else 0)
    return {"segment_char_start": start, "segment_char_end": end,
            "evidence": text[start:end], "evidence_sha256": digest(text[start:end]),
            "source_spans": spans, "assembly_separator": "\n"}


def represent_experience_clause(clause):
    text = clause["evidence"]
    if digest(text) != clause["evidence_sha256"] or "\n".join(s["evidence"] for s in clause["evidence_spans"]) != text:
        raise ValueError("Logical clause evidence mismatch")
    result = {"representation_version": VERSION, "source_rule_version": clause["rule_version"],
              "source_clause_page": clause["page"], "source_clause_sha256": clause["evidence_sha256"],
              "subject": clause["subject"], "section_role": clause["section_role"],
              "field_name": "required_bidder_billed_experience",
              "not_procurement_budget": True, "branches": [], "warnings": [],
              "selected_branch": None, "company_applicability": "not_determined",
              "company_capacity_determined": False, "legal_eligibility_determined": False,
              "ranking_eligible": False, "historical_content_proven": False,
              "whole_document_conditions_complete": False}
    if clause["status"] != "explicit_clause_extracted" or clause["subject"] != "bidder" or clause["section_role"] not in {"qualification", "evaluation_factor"}:
        result["status"] = "unsupported_subject_section_or_clause"
        return result
    predicate = PREDICATE.search(text)
    if not predicate:
        raise ValueError("Extracted clause has no supported predicate")
    conditionals = list(MICRO_SMALL.finditer(text))
    primary_end = conditionals[0].start() if conditionals else len(text)
    starts = [("primary_paragraph", predicate.start(), predicate.end(), primary_end,
               {"type": "base_paragraph_not_resolved_for_company"})]
    for i, match in enumerate(conditionals):
        end = conditionals[i + 1].start() if i + 1 < len(conditionals) else len(text)
        starts.append((f"micro_small_declaration_{i + 1}", match.start(), match.end(), end,
                       {"type": "explicit_declaration_micro_small_enterprise",
                        "evidence": source_range(clause, match.start(), match.end()),
                        "declaration_verified_for_company": False}))
    supported_money_ranges = set()
    for branch_id, begin, amount_start, end, condition in starts:
        money = MONEY.match(text, amount_start)
        amount = parse_money(money.group("amount")) if money and money.end() <= end else None
        if amount is None or PLACEHOLDER.search(text[amount_start:min(end, amount_start + 220)]):
            result["warnings"].append("unresolved_or_template_amount_in_supported_branch")
            continue
        supported_money_ranges.add((money.start(), money.end()))
        currency = "USD" if money.group("currency").upper().startswith("US") else "PEN"
        window = LOOKBACK.search(text, money.end(), end)
        branch = {"branch_id": branch_id, "condition": condition,
                  "billed_amount": amount, "currency": currency,
                  "lookback_years": int(window.group(1)) if window else None,
                  "amount_evidence": source_range(clause, money.start(), money.end()),
                  "window_evidence": source_range(clause, window.start(), window.end()) if window else None,
                  "paragraph_evidence": source_range(clause, begin, end),
                  "applicability": "not_determined", "ranking_eligible": False}
        if window is None:
            result["warnings"].append("lookback_not_identified_in_this_branch")
        if branch_id == "primary_paragraph" and (amount, currency, branch["lookback_years"]) != (clause["billed_amount"], clause["currency"], clause["lookback_years"]):
            # The old parser may have picked a window from another paragraph.
            result["warnings"].append("branch_scoped_values_differ_from_v2_summary")
        if branch_id.startswith("micro_small"):
            consortium = CONSORTIUM.search(text, money.end(), end)
            if consortium:
                branch["additional_condition"] = {
                    "type": "all_consortium_members_micro_small_as_stated",
                    "evidence": source_range(clause, consortium.start(), consortium.end()),
                    "verified_for_company": False}
        result["branches"].append(branch)
    extra = [m for m in MONEY.finditer(text) if (m.start(), m.end()) not in supported_money_ranges]
    if extra:
        result["warnings"].append("additional_monetary_mentions_not_assigned_to_a_supported_branch")
        result["unassigned_monetary_evidence"] = [source_range(clause, m.start(), m.end()) for m in extra]
    if clause.get("segment_stop") in {"page_limit", "character_limit", "non_adjacent_page"}:
        result["warnings"].append("segment_boundary_may_omit_conditions")
    result["single_universal_threshold_supported"] = False
    result["status"] = "requires_review_of_unresolved_segment" if result["warnings"] else "conditional_branches_represented" if len(result["branches"]) > 1 else "single_supported_paragraph_represented"
    return result
