"""Selective extraction: uncertain logical identity emits no assertion.

Raw extraction and strict repetition detection remain unchanged. Every context
and occurrence is retained, including conflicting groups. Abstention is a model
decision, not adjudication, source exclusion or permission to remove gold rows.
"""
from __future__ import annotations

import copy

from .document_experience_candidate_v1 import predict_documents as predict_raw
from .document_experience_units_dev_v1 import anchor_key, checksum, project_units, stable_json
from .reference_evaluation import key

VERSION = "document_experience_selective_candidate_v2"
POLICY = "whole_group_abstention_without_value_selection_v1"
SOURCE_ERRORS = {"multiple_pdf_sources_for_opportunity_unresolved", "same_pdf_under_multiple_opportunities_unresolved"}


def decisions(raw):
    projection = project_units(raw["document_outputs"])
    if projection != raw["occurrence_projection"]:
        raise ValueError("Raw occurrence audit changed")
    occurrences = {o["occurrence_id"]: o for o in projection["occurrences"]}
    predictions, emitted, abstentions, contexts = [], [], [], []
    for unit in projection["candidate_units"]:
        if SOURCE_ERRORS & set(unit["reasons"]):
            raise ValueError("Ambiguous PDF identity is an input error, not a model abstention")
        context_groups = {}
        for oid in unit["member_occurrence_ids"]:
            row = occurrences[oid]["record"]
            anchors = {name: anchor_key(row[name]) for name in ("section_evidence", "subject_evidence")}
            context_id = "context-v2:" + checksum(stable_json([unit["identity"], anchors]))
            context_groups.setdefault(context_id, {"context_id": context_id, "unit_id": unit["unit_id"],
                "section_evidence": copy.deepcopy(row["section_evidence"]),
                "subject_evidence": copy.deepcopy(row["subject_evidence"]), "occurrence_ids": []})["occurrence_ids"].append(oid)
        contexts.extend(context_groups[c] for c in sorted(context_groups))
        if unit["status"] == "unresolved_blocks_comparison":
            if not unit["reasons"] or unit["fields"] is not None:
                raise ValueError("Unresolved group must retain reasons and no selected fields")
            abstentions.append({"unit_id": unit["unit_id"], "identity": copy.deepcopy(unit["identity"]),
                "decision": "abstain_entire_candidate_identity_group", "reasons": list(unit["reasons"]),
                "context_ids": sorted(context_groups), "occurrence_ids": list(unit["member_occurrence_ids"]),
                "emitted_values": None, "identity_resolved": False})
        elif unit["status"] in {"single_occurrence", "exact_repeat_consolidated"}:
            predictions.append({**{k: unit["identity"][k] for k in
                ("opportunity_id", "subject", "section_role", "item", "branch")}, "fields": copy.deepcopy(unit["fields"])})
            emitted.append(unit["unit_id"])
        else:
            raise ValueError("Unknown raw unit status")
    all_ids = {u["unit_id"] for u in projection["candidate_units"]}
    abstained = {a["unit_id"] for a in abstentions}
    if set(emitted) & abstained or set(emitted) | abstained != all_ids:
        raise ValueError("Every candidate group needs exactly one model decision")
    if len({key(p) for p in predictions}) != len(predictions):
        raise ValueError("Duplicate emitted prediction identity")
    return {"policy": POLICY, "predictions": predictions, "emitted_unit_ids": emitted,
            "abstentions": abstentions, "source_contexts": contexts,
            "original_occurrences": len(occurrences), "candidate_groups": len(all_ids),
            "all_raw_occurrences_preserved": True, "uncertain_groups_resolved": False,
            "reference_values_used": False, "reference_denominator_may_be_filtered": False}


def predict_documents(documents):
    if isinstance(documents, list) and len({d["source"]["sha256"] for d in documents}) != len(documents):
        raise ValueError("Ambiguous PDF identity is an input error, not a model abstention")
    raw = predict_raw(documents)
    selective = decisions(raw)
    return {"version": VERSION, "stage": "selective_source_only_candidate_no_metrics",
            "raw_candidate_output": raw, "selective_decisions": selective,
            "reference_or_expected_values_used": False, "metrics_computed": False,
            "pdf_downloads_performed": False, "new_ocr_performed": False, "ranking_eligible": False,
            "raw_text_parser_changed": False, "semantic_identity_accuracy_proven": False}
