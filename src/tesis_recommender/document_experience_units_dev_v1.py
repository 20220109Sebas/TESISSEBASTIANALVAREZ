"""Conservative occurrence identity for observed-document development only.

Never selects a field value across disagreeing occurrences. Exact repetition
requires matching source/scope, text, fields and parser diagnostics. An unresolved
group blocks comparison rather than disappearing from its denominator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from decimal import Decimal

from .document_experience_dev_v1 import ITEM

VERSION = "document_experience_units_development_v1"
FIELDS = ("billed_amount", "currency", "lookback_years")
IDENTITY = ("opportunity_id", "source_pdf_sha256", "subject", "section_role", "item", "branch")
AMOUNT_STATES = {"complete_token_normalized", "no_literal_absolute_money",
                 "missing_or_unsupported_number", "ambiguous_number_abstained"}


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def checksum(value):
    return hashlib.sha256(value.encode()).hexdigest()


def clause_key(text):
    """Whitespace-only equivalence: no case, spelling or punctuation changes."""
    return " ".join(text.split())


def anchor_key(ref):
    if not ref or not ref.get("source_spans"):
        raise ValueError("Explicit context evidence is required")
    return tuple((s["page"], s["char_start"], s["char_end"], s["evidence_sha256"])
                 for s in ref["source_spans"])


def validate_row(row):
    if row["subject"] != "bidder" or row["section_role"] not in ("qualification", "evaluation_factor"):
        raise ValueError("Unsupported subject or role must remain outside this stream")
    if row["branch"] not in ("general", "mype"):
        raise ValueError("Unknown branch")
    item = row["item"]
    if item is not None and (type(item) is not int or item < 1):
        raise ValueError("Item must be an explicit positive integer or null")
    if item is not None:
        ref = row.get("item_evidence")
        match = ITEM.fullmatch(ref["evidence"].strip()) if ref else None
        if not match or int(match.group(1)) != item:
            raise ValueError("Item identity lacks matching explicit source evidence")
    elif row.get("item_evidence") is not None:
        raise ValueError("Null item cannot carry an assigned item heading")
    fields = row["fields"]
    if set(fields) != set(FIELDS):
        raise ValueError("Expected exactly the three document fields")
    amount = fields["billed_amount"]
    if amount is not None and (not isinstance(amount, str)
            or not re.fullmatch(r"(?:0|[1-9][0-9]*)\.[0-9]{2}", amount) or Decimal(amount) <= 0):
        raise ValueError("Amount must be a positive canonical decimal string or null")
    if fields["currency"] not in (None, "PEN", "USD"):
        raise ValueError("Unsupported literal currency")
    years = fields["lookback_years"]
    if years is not None and (type(years) is not int or not 1 <= years <= 99):
        raise ValueError("Window must be explicit positive years or null")
    if row["amount_status"] not in AMOUNT_STATES:
        raise ValueError("Unknown monetary parsing state")
    if (row["amount_status"] == "complete_token_normalized") != (amount is not None):
        raise ValueError("Amount and parser state disagree")
    if not isinstance(row["warnings"], list) or any(not isinstance(w, str) for w in row["warnings"]):
        raise ValueError("Invalid parser diagnostics")
    if row.get("ranking_eligible") is not False:
        raise ValueError("Development records cannot supply ranking inputs")
    anchor_key(row.get("section_evidence"))
    anchor_key(row.get("subject_evidence"))
    raw = row["clause_evidence"]["evidence"]
    if not clause_key(raw) or checksum(raw) != row["clause_evidence"]["evidence_sha256"]:
        raise ValueError("Clause is empty or differs from its evidence hash")
    anchor_key(row["clause_evidence"])
    if not isinstance(row["prediction_id"], str) or not row["prediction_id"]:
        raise ValueError("Missing occurrence identifier")


def project_units(documents):
    occurrences, groups = [], defaultdict(list)
    seen_docs, seen_ids, seen_locations = set(), set(), set()
    pdfs_by_ocid, ocids_by_pdf = defaultdict(set), defaultdict(set)
    for document in documents:
        ocid, pdf = document["opportunity_id"], document["source_pdf_sha256"]
        if not isinstance(ocid, str) or not ocid.startswith("ocds-") or not re.fullmatch(r"[a-f0-9]{64}", pdf):
            raise ValueError("Invalid source identity")
        if (ocid, pdf) in seen_docs:
            raise ValueError("Repeated document input")
        seen_docs.add((ocid, pdf))
        pdfs_by_ocid[ocid].add(pdf)
        ocids_by_pdf[pdf].add(ocid)
        for row in document["result"]["records"]:
            validate_row(row)
            identity = dict(opportunity_id=ocid, source_pdf_sha256=pdf,
                            **{k: row[k] for k in IDENTITY[2:]})
            source_id = ocid, pdf, row["prediction_id"]
            location = ocid, pdf, anchor_key(row["clause_evidence"])
            if source_id in seen_ids or location in seen_locations:
                raise ValueError("Repeated occurrence input, not a distinct source appearance")
            seen_ids.add(source_id)
            seen_locations.add(location)
            occurrence = {"occurrence_id": "occurrence-dev1:" + checksum(stable_json(source_id)),
                          "identity": identity, "record": copy.deepcopy(row)}
            occurrences.append(occurrence)
            groups[stable_json(identity)].append(occurrence)
    units = []
    for encoded, members in sorted(groups.items()):
        identity = json.loads(encoded)
        members.sort(key=lambda m: (anchor_key(m["record"]["clause_evidence"]), m["occurrence_id"]))
        rows = [m["record"] for m in members]
        reasons, conflicts = [], {}
        if len(pdfs_by_ocid[identity["opportunity_id"]]) > 1:
            reasons.append("multiple_pdf_sources_for_opportunity_unresolved")
        if len(ocids_by_pdf[identity["source_pdf_sha256"]]) > 1:
            reasons.append("same_pdf_under_multiple_opportunities_unresolved")
        for field in FIELDS:
            known = {stable_json(r["fields"][field]) for r in rows if r["fields"][field] is not None}
            if len(known) > 1:
                conflicts[field] = [json.loads(v) for v in sorted(known)]
        if conflicts:
            reasons.append("conflicting_known_field_values")
        if len({stable_json(r["fields"]) for r in rows}) > 1 and not conflicts:
            reasons.append("known_vs_null_field_disagreement")
        text_hashes = sorted({checksum(clause_key(r["clause_evidence"]["evidence"])) for r in rows})
        if len(text_hashes) > 1:
            reasons.append("different_clause_text_same_candidate_identity")
        if len({stable_json([r["amount_status"], r["warnings"]]) for r in rows}) > 1:
            reasons.append("inconsistent_parser_diagnostics")
        if len(rows) > 1 and identity["item"] is None:
            contexts = {(anchor_key(r["section_evidence"]), anchor_key(r["subject_evidence"])) for r in rows}
            if len(contexts) > 1:
                reasons.append("unknown_item_across_distinct_context_anchors")
        status = "unresolved_blocks_comparison" if reasons else "exact_repeat_consolidated" if len(rows) > 1 else "single_occurrence"
        units.append({"unit_id": "unit-dev1:" + checksum(encoded), "identity": identity,
            "status": status, "reasons": reasons, "known_value_conflicts": conflicts,
            "fields": copy.deepcopy(rows[0]["fields"]) if not reasons else None,
            "member_occurrence_ids": [m["occurrence_id"] for m in members],
            "whitespace_only_clause_sha256": text_hashes,
            "all_original_occurrences_preserved": True, "semantic_or_legal_identity_proven": False})
    partition = [member for unit in units for member in unit["member_occurrence_ids"]]
    if len(partition) != len(occurrences) or len(set(partition)) != len(occurrences):
        raise ValueError("Occurrence partition is not lossless")
    blocked = [u["unit_id"] for u in units if u["status"] == "unresolved_blocks_comparison"]
    return {"version": VERSION, "stage": "observed_document_development_only",
        "occurrences": sorted(occurrences, key=lambda o: o["occurrence_id"]), "candidate_units": units,
        "identity_gate": {"passed": not blocked, "blocked_unit_ids": blocked,
                          "action": "stop_before_field_comparison" if blocked else "identity_projection_only_not_final_evaluation"},
        "all_occurrences_preserved": True, "input_records_modified": False,
        "reference_or_expected_values_used": False, "metrics_computed": False,
        "new_evaluation_ready": False, "ranking_eligible": False}
