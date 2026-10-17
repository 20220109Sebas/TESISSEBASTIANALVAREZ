"""Heading-context development on observed documents, not unseen evaluation.

Only heading comparison keys are accent-folded. Archived text and monetary
parsing remain unchanged. Supplier-profile/TDR and unknown contexts are not
promoted to qualification; every recognized context retains its raw evidence.
"""
from __future__ import annotations

import re
import unicodedata

from .document_experience_dev_v1 import (
    BARRIER, BOUNDARY, ITEM, LOOKBACK, MICRO_SMALL, MYPE_VARIANT, PREDICATE,
    copy_ref, digest, evidence, money_at, page_index,
    verify_provenance as verify_v1_provenance,
)
from .experience_segments import ANNEX, CHAPTER, EVAL, QUAL, SUBJECT

VERSION = "document_experience_development_v2"
QUAL_WITH_ARTICLE = re.compile(
    r"(?:(?:[A-Z](?:\.\d+)*|\d+(?:\.\d+)*|[IVX]+)[.)]?\s+)?"
    r"REQUISITOS DE LA CALIFICACION(?: OBLIGATORIOS| FACULTATIVOS)?\s*:?[ \t]*", re.I)
SUPPLIER_PROFILE = re.compile(
    r"(?:(?:[A-Z](?:\.\d+)*|\d+(?:\.\d+)*|[IVX]+)[.)]?\s+)?"
    r"(?:REQUISITOS|PERFIL) DEL PROVEEDOR(?: Y/O PERSONAL)?(?: \(OBLIGATORIO\))?\s*:?[ \t]*", re.I)


def heading_key(raw):
    """Diacritic normalization only; no OCR word correction or body rewriting."""
    return "".join(c for c in unicodedata.normalize("NFD", raw.strip())
                   if not unicodedata.combining(c))


def heading_events(pages):
    events = []
    section, section_ref = "unknown", None
    for page in pages:
        position = 0
        for line in page["text"].splitlines(keepends=True):
            key = heading_key(line)
            role, subject = None, None
            if QUAL.fullmatch(key) or QUAL_WITH_ARTICLE.fullmatch(key):
                role = "qualification"
            elif EVAL.fullmatch(key):
                role = "evaluation_factor"
            elif ANNEX.fullmatch(key):
                role = "annex_form"
            elif CHAPTER.fullmatch(key):
                role = "unknown"
            elif SUPPLIER_PROFILE.fullmatch(key):
                role = "tdr_supplier_profile"
            match = SUBJECT.fullmatch(key)
            if role is not None:
                section = role
                section_ref = {"page": page["page"], "char_start": position,
                               "char_end": position + len(line)}
            if match:
                subject = "bidder" if match.group(1).casefold() == "postor" else "key_personnel"
            if role is not None or subject is not None:
                events.append({"page": page["page"], "char_start": position,
                    "char_end": position + len(line), "kind": "subject" if subject else "role",
                    "subject": subject, "section_role": section,
                    "section_anchor": dict(section_ref) if section_ref else None,
                    "heading_key": key, "diacritic_fold_applied": key != line.strip()})
            position += len(line)
    return events


def extract_development(pages, *, max_pages=2, max_chars=6000):
    if max_pages != 2 or max_chars < 100:
        raise ValueError("This development version uses two-page bounded clauses")
    text, layout = page_index(pages)
    starts = {p["page"]: a for a, _, p in layout}
    timeline = []
    for event in heading_events(pages):
        anchor = event["section_anchor"]
        section_ref = evidence(text, layout, starts[anchor["page"]] + anchor["char_start"],
                               starts[anchor["page"]] + anchor["char_end"]) if anchor else None
        timeline.append({"position": starts[event["page"]] + event["char_start"],
            "end": starts[event["page"]] + event["char_end"], "kind": event["kind"],
            "subject": event["subject"], "role": event["section_role"], "section_ref": section_ref,
            "subject_ref": evidence(text, layout, starts[event["page"]] + event["char_start"],
                                    starts[event["page"]] + event["char_end"]) if event["subject"] else None})
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
    section_ref, subject_ref = None, None
    t = 0
    for i, (clause, branch) in enumerate(clauses):
        while t < len(timeline) and timeline[t]["position"] <= clause.start():
            event = timeline[t]
            if event["kind"] in ("role", "subject"):
                role, subject = event["role"], event["subject"]
                section_ref, subject_ref = event["section_ref"], event["subject_ref"]
                item, item_ref = None, None
            elif event["kind"] == "barrier":
                subject, subject_ref, item, item_ref = None, None, None, None
            elif subject == "bidder" and role in ("qualification", "evaluation_factor"):
                if event["item"] > 0:
                    item = event["item"]
                    item_ref = evidence(text, layout, event["position"], event["end"])
            t += 1
        context = {"section_evidence": copy_ref(section_ref), "subject_evidence": copy_ref(subject_ref)}
        predicate_ref = evidence(text, layout, clause.start(), clause.end())
        if subject != "bidder" or role not in ("qualification", "evaluation_factor"):
            skipped.append({"reason": "unsupported_subject_or_role_context", "role": role,
                            "subject": subject, "predicate": predicate_ref, **context})
            continue
        p_index = next(j for j, (a, b, _) in enumerate(layout) if a <= clause.start() < b)
        end = min(layout[min(p_index + max_pages - 1, len(layout) - 1)][1],
                  clause.start() + max_chars,
                  clauses[i + 1][0].start() if i + 1 < len(clauses) else len(text))
        next_boundary = next((e["position"] for e in timeline[t:] if e["position"] > clause.start()), end)
        end = min(end, next_boundary)
        boundary = BOUNDARY.search(text, clause.end(), end)
        if boundary:
            end = boundary.start()
        token = money_at(text[:end], clause.end())
        window = LOOKBACK.search(text, max(clause.end(), token["end"]), end)
        records.append({"version": VERSION,
            "prediction_id": f"dev2:{layout[p_index][2]['page']}:{clause.start()}:{branch}",
            "page": layout[p_index][2]["page"], "subject": "bidder", "section_role": role,
            "branch": branch, "item": item, **context, "predicate_evidence": predicate_ref,
            "fields": {"billed_amount": token["billed_amount"], "currency": token["currency"],
                       "lookback_years": int(window.group(1)) if window else None},
            "amount_status": token["status"], "warnings": token["warnings"],
            "amount_evidence": evidence(text, layout, token["start"], token["end"]) if token["end"] > token["start"] else None,
            "currency_evidence": evidence(text, layout, token["start"], token["currency_end"]) if token["currency_end"] > token["start"] else None,
            "window_evidence": evidence(text, layout, window.start(), window.end()) if window else None,
            "item_evidence": copy_ref(item_ref), "clause_evidence": evidence(text, layout, clause.start(), end),
            "stage": "observed_source_development_not_unseen_validation", "company_applicability": "not_determined",
            "ranking_eligible": False, "historical_content_proven": False, "legal_eligibility_determined": False})
    return {"version": VERSION, "records": records, "skipped_contexts": skipped,
            "parameters": {"max_pages": max_pages, "max_chars": max_chars},
            "stage": "development_on_previously_inspected_sources", "f1_computed": False,
            "independent_semantic_accuracy_proven": False, "ranking_eligible": False,
            "automatic_deduplication_performed": False, "archived_text_modified": False,
            "normalization_scope": "heading_comparison_keys_only_no_fuzzy_spelling_correction"}


def verify_provenance(result, pages):
    verify_v1_provenance(result, pages)
    # Reuse the identical evidence validator for the new context references too.
    refs = [row.get(key) for row in result["records"] + result["skipped_contexts"]
            for key in ("section_evidence", "subject_evidence", "predicate_evidence") if row.get(key)]
    verify_v1_provenance({"records": [], "skipped_contexts": [{"predicate": ref} for ref in refs]}, pages)
    return True
