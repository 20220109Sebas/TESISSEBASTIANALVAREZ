"""Versioned reference-only visual evidence; candidate inputs remain untouched."""
from __future__ import annotations

import copy
import re
from decimal import Decimal

from .document_experience_evaluation_v1 import (
    FIELDS, ROLES, SCOPES, fingerprint, require_hash, source_population,
    validate_evidence as validate_text_evidence,
)
from .document_experience_units_dev_v1 import checksum
from .reference_evaluation import key

VERSION = "document_experience_evaluation_gates_v2"
VISUAL_KIND = "visual_transcription_v1"


def validate_catalog(catalog, sources, notes, visual_registry, evidence_files):
    if (catalog.get("version") != VISUAL_KIND or catalog.get("source_only") is not True
            or catalog.get("predictions_consulted") is not False
            or catalog.get("candidate_input_modified") is not False):
        raise ValueError("Visual transcripts are disclosed source-only reference evidence")
    archived = {d["opportunity_id"]: d for d in notes["documents"]}
    index = {}
    for entry in catalog["entries"]:
        ident, ocid, number = entry["id"], entry["opportunity_id"], entry["page"]
        if not isinstance(ident, str) or not ident or ident in index or ocid not in sources:
            raise ValueError("Unique transcript identity and fixed source required")
        source = sources[ocid]
        if type(number) is not int or not 1 <= number <= len(source["pages"]):
            raise ValueError("Transcript page outside physical source")
        if (entry["pdf_sha256"] != source["source"]["sha256"]
                or entry["page_text_sha256"] != source["pages"][number - 1]["text_sha256"]):
            raise ValueError("Visual transcript belongs to a different PDF/page view")
        text = entry["text"]
        if not isinstance(text, str) or not text.strip() or checksum(text) != entry["text_sha256"]:
            raise ValueError("Transcript text hash differs")
        if entry.get("transcription_method") != "source_only_visual_reconciliation_not_candidate_ocr":
            raise ValueError("Declare visual transcription separately from candidate OCR")
        confirmations = entry["archived_pass_confirmations"]
        if [c["pass_number"] for c in confirmations] != [1, 2] or any(type(c["pass_number"]) is not int for c in confirmations):
            raise ValueError("Both archived source passes must support visual reconciliation")
        for confirmation in confirmations:
            number_pass = confirmation["pass_number"]
            old_reading = next(r for r in archived[ocid]["readings"] if r["pass_number"] == number_pass)
            if confirmation["reading_sha256"] != fingerprint(old_reading):
                raise ValueError("Visual reconciliation rewrites an archived reading")
            if entry.get("positive_enterprise_requirement") is True:
                known = entry.get("known_fields", {})
                identity = entry.get("condition_identity", {})
                if (set(known) != set(FIELDS) or not any(v is not None for v in known.values())
                        or identity.get("subject") != "bidder" or identity.get("section_role") not in ROLES
                        or identity.get("branch") not in {"general", "mype"}):
                    raise ValueError("Visual positive requires explicit identity and known fields")
                if not any(all(observation.get(field + "_observed") == value
                               for field, value in known.items() if value is not None)
                           for observation in old_reading.get("source_only_observations", [])):
                    raise ValueError("Visual known fields lack support in an archived source pass")
            matches = [v for v in visual_registry if (v["opportunity_id"], v["page"], v["pass_number"])
                       == (ocid, number, number_pass)]
            if len(matches) != 1:
                raise ValueError("Visual transcript lacks an archived full-page binding for both passes")
            view = matches[0]
            if (view["source_pdf_sha256"] != entry["pdf_sha256"]
                    or entry["image_evidence"] != {"path": view["path"], "sha256": view["sha256"]}):
                raise ValueError("Transcript PNG differs from the archived visual pass")
        image = entry["image_evidence"]
        require_hash(image["sha256"])
        evidence_files[image["path"]] = image["sha256"]
        index[ident] = entry
    visual_positives = {d["opportunity_id"] for d in notes["documents"]
                       if any(r.get("source_only_confirmation", {}).get(
                           "visual_requirement_not_recovered_in_frozen_prepared_text") is True
                              and r.get("source_only_confirmation", {}).get(
                                  "concrete_experience_condition_visually_identified") is True
                              for r in d["readings"])}
    if not visual_positives <= {e["opportunity_id"] for e in index.values()
                               if e.get("positive_enterprise_requirement") is True}:
        raise ValueError("Preserve every archived visual positive, not just text-recovered conditions")
    return index


def validate_visual_page(review, source, pass_number, visual_registry, evidence_files):
    matches = [v for v in visual_registry if (v["opportunity_id"], v["page"], v["pass_number"])
               == (source["opportunity_id"], review["page"], pass_number)]
    if len(matches) != 1 or review.get("image_evidence") != {
            "path": matches[0]["path"], "sha256": matches[0]["sha256"]}:
        raise ValueError("Visual readable status requires its archived full-page pass")
    view = matches[0]
    if view["source_pdf_sha256"] != source["source"]["sha256"]:
        raise ValueError("Visual readable page has another PDF identity")
    evidence_files[view["path"]] = view["sha256"]


def validate_visual_positive_retention(decisions, catalog):
    used = {ref["transcript_id"] for d in decisions if d["scope"] == "enterprise_services"
            for c in d["conditions"] for refs in [c["identity_evidence"], *c["field_evidence"].values()]
            for ref in refs if ref.get("kind") == VISUAL_KIND}
    if any(e.get("positive_enterprise_requirement") is True and ident not in used for ident, e in catalog.items()):
        raise ValueError("Do not silently discard a visual positive from the evaluable reference")
    all_conditions = {key(c): c for d in decisions if d["scope"] == "enterprise_services" for c in d["conditions"]}
    for ident, entry in catalog.items():
        if entry.get("positive_enterprise_requirement") is not True:
            continue
        identity = dict(entry["condition_identity"], opportunity_id=entry["opportunity_id"])
        condition = all_conditions.get(key(identity))
        if condition is None:
            raise ValueError("Visual positive identity must remain in the evaluable reference")
        for field, value in entry["known_fields"].items():
            if value is not None and (condition["fields"][field] != value or not any(
                    ref.get("transcript_id") == ident for ref in condition["field_evidence"][field])):
                raise ValueError("Do not ungrade an agreed visual field or drop its source evidence")


def validate_evidence(ref, source, catalog):
    if ref.get("kind") == VISUAL_KIND:
        entry = catalog.get(ref.get("transcript_id"))
        if entry is None or entry["opportunity_id"] != source["opportunity_id"]:
            raise ValueError("Unknown or cross-document visual transcript")
        if {"char_start", "char_end"} & ref.keys():
            raise ValueError("Visual offsets must not impersonate native text offsets")
        start, end = ref["transcript_char_start"], ref["transcript_char_end"]
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(entry["text"])
                or ref["quote"] != entry["text"][start:end]
                or type(ref["page"]) is not int or ref["page"] != entry["page"] or ref["pdf_sha256"] != entry["pdf_sha256"]
                or ref["page_text_sha256"] != entry["page_text_sha256"]
                or ref["transcript_text_sha256"] != entry["text_sha256"]):
            raise ValueError("Visual evidence differs from its fixed transcript")
        return
    if ref.get("kind") not in (None, "frozen_prepared_text_literal"):
        raise ValueError("Unsupported reference evidence kind")
    validate_text_evidence(ref, source)


def validate_condition(condition, source, catalog):
    if (condition["opportunity_id"] != source["opportunity_id"] or condition["subject"] != "bidder"
            or condition["section_role"] not in ROLES or condition["branch"] not in ("general", "mype")):
        raise ValueError("Condition has unsupported source, subject, role or branch")
    item = condition.get("item")
    if item is not None and (type(item) is not int or item < 1):
        raise ValueError("Item must be explicit positive integer or null")
    if set(condition["fields"]) != set(FIELDS) or set(condition["unknown_reasons"]) != set(FIELDS):
        raise ValueError("All three fields and unknown reasons must be explicit")
    evidence = condition["identity_evidence"]
    if not evidence or not condition.get("identity_reason"):
        raise ValueError("Source-only scope/identity evidence and rationale required")
    for ref in evidence:
        validate_evidence(ref, source, catalog)
    if set(condition["field_evidence"]) != set(FIELDS):
        raise ValueError("Each field must retain its own literal evidence list")
    for field in FIELDS:
        value, reason = condition["fields"][field], condition["unknown_reasons"][field]
        if (value is None and (not isinstance(reason, str) or not reason.strip())) or (value is not None and reason is not None):
            raise ValueError("Unknowns need reasons; known fields cannot be marked unknown")
        refs = condition["field_evidence"][field]
        if not isinstance(refs, list) or (value is not None and not refs):
            raise ValueError("Known values require literal source support")
        for ref in refs:
            validate_evidence(ref, source, catalog)
        if value is None:
            continue
        if field == "billed_amount" and (not isinstance(value, str)
                or not re.fullmatch(r"(?:0|[1-9][0-9]*)\.[0-9]{2}", value) or Decimal(value) <= 0):
            raise ValueError("No inferred, rounded or zero billed amount")
        if field == "currency" and value not in ("PEN", "USD"):
            raise ValueError("Unsupported literal currency")
        if field == "lookback_years" and (type(value) is not int or not 1 <= value <= 99):
            raise ValueError("No inferred lookback window")


def validate_reading(reading, source, evidence_files, catalog, visual_registry):
    if (reading.get("full_supplied_document_read") is not True
            or reading.get("predictions_consulted") is not False
            or reading.get("model_rules_consulted") is not False or not reading.get("reviewer")):
        raise ValueError("Two disclosed source-only full-document readings required")
    if reading.get("scope") not in SCOPES - {"inaccessible_source"}:
        raise ValueError("Each source reading must record its own scope decision")
    if reading["scope"] in {"outside_consultoria_de_obra", "outside_individual_professional"}:
        if not reading.get("scope_evidence"):
            raise ValueError("Out-of-domain exclusion requires literal scope evidence in both readings")
        for ref in reading["scope_evidence"]:
            validate_evidence(ref, source, catalog)
    reviews = reading["page_reviews"]
    if [r["page"] for r in reviews] != list(range(1, len(source["pages"]) + 1)) or any(type(r["page"]) is not int for r in reviews):
        raise ValueError("Every physical page needs an ordered review outcome")
    inadequate = False
    for review, page in zip(reviews, source["pages"]):
        if review["page_text_sha256"] != page["text_sha256"]:
            raise ValueError("Reading belongs to another page view")
        status = review["content_status"]
        if status == "readable":
            if not page["text"].strip() or page["selected_ocr_unresolved"]:
                raise ValueError("Incomplete text cannot be relabelled as readable")
        elif status == "visually_verified_readable":
            validate_visual_page(review, source, reading["pass_number"], visual_registry, evidence_files)
        elif status == "visually_verified_blank":
            if page["text"].strip():
                raise ValueError("Nonempty text cannot become a blank page")
            image = review["image_evidence"]
            require_hash(image["sha256"])
            if not isinstance(image["path"], str) or not image["path"]:
                raise ValueError("Blank pages require archived visual evidence")
            if image["path"] in evidence_files and evidence_files[image["path"]] != image["sha256"]:
                raise ValueError("Conflicting image evidence hashes")
            evidence_files[image["path"]] = image["sha256"]
        elif status == "unreadable":
            inadequate = True
        else:
            raise ValueError("Unknown page readability status")
    conditions = reading["conditions"]
    if len({key(c) for c in conditions}) != len(conditions):
        raise ValueError("Reference identity unresolved; do not select the best match")
    for condition in conditions:
        validate_condition(condition, source, catalog)
    return inadequate


def build_reference(draft, documents, ledger, catalog, notes, visual_registry, baseline):
    """Reconcile two archived readings, never predictions. Unknowns stay unknown."""
    sources = source_population(documents, ledger)
    evidence_files = {}
    catalog = validate_catalog(catalog, sources, notes, visual_registry, evidence_files)
    if draft.get("reconciliation_complete") is not True:
        raise ValueError("Complete source-only reconciliation required before reference lock")
    prior = {d["opportunity_id"]: d for d in baseline["documents"]}
    if (draft.get("reference_type") != "automatic_source_review" or draft.get("locked") is not False
            or draft.get("predictions_seen_before_lock") is not False
            or draft.get("is_human_or_expert") is not False or draft.get("independent_reviewer") is not False
            or not draft.get("independence_note")):
        raise ValueError("Disclose automatic same-agent review before inference")
    decisions = draft["documents"]
    if len(decisions) != 50 or len({d["opportunity_id"] for d in decisions}) != 50 or {d["opportunity_id"] for d in decisions} != {r["opportunity_id"] for r in ledger}:
        raise ValueError("Reference must retain the complete fifty-case denominator")
    reference_docs = []
    for decision in decisions:
        ocid, scope = decision["opportunity_id"], decision["scope"]
        reconciliation = decision.get("reconciliation", {})
        expected_status = "abstained" if scope in {"insufficient_reference", "inaccessible_source"} else "resolved"
        if (reconciliation.get("status") != expected_status or not reconciliation.get("reason")
                or reconciliation.get("prior_decision_sha256") != fingerprint(prior[ocid])):
            raise ValueError("Every case needs a bound source-only reconciliation or explicit abstention")
        if scope not in SCOPES or not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
            raise ValueError("Every case requires a scope and rationale")
        base = {"opportunity_id": ocid, "scope": scope, "reason": decision["reason"],
                "complete": False, "first_pass_complete": False, "second_pass_complete": False,
                "conditions": copy.deepcopy(decision["conditions"])}
        source = sources.get(ocid)
        if source is None:
            if scope != "inaccessible_source" or decision["conditions"] or decision["readings"] or decision.get("source_pdf_sha256") is not None:
                raise ValueError("Unavailable/duplicate/excluded sources are not negative references")
            reference_docs.append(base)
            continue
        if scope == "inaccessible_source" or decision["source_pdf_sha256"] != source["source"]["sha256"]:
            raise ValueError("Reference PDF/source identity differs")
        readings = decision["readings"]
        if len(readings) != 2 or [r["pass_number"] for r in readings] != [1, 2] or any(type(r["pass_number"]) is not int for r in readings):
            raise ValueError("Exactly two ordered source readings required")
        inadequate = [validate_reading(r, source, evidence_files, catalog, visual_registry) for r in readings]
        if any(r["scope"] != scope for r in readings):
            raise ValueError("Resolve source-only scope disagreements before locking the reference")
        if any(inadequate) and scope != "insufficient_reference":
            raise ValueError("Unreadable pages cannot support positive or negative formal references")
        for c in decision["conditions"]:
            validate_condition(c, source, catalog)
        final = {key(c): c for c in decision["conditions"]}
        inventories = [{key(c): c for c in r["conditions"]} for r in readings]
        if len(final) != len(decision["conditions"]) or any(set(i) != set(final) for i in inventories):
            raise ValueError("Resolve source-only identity differences before lock; never drop conditions")
        for ident, condition in final.items():
            for field in FIELDS:
                a, b = (i[ident]["fields"][field] for i in inventories)
                value = condition["fields"][field]
                if value is not None and (a != value or b != value):
                    raise ValueError("Known reference fields require agreement in both readings")
                if value is None and a is not None and a == b and not condition.get("reconciliation_reason"):
                    raise ValueError("Do not discard an agreed known field without a source rationale")
        if scope == "enterprise_services" and not final:
            if any(not isinstance(r.get("scoped_absence_statement"), str) or not r["scoped_absence_statement"].strip() for r in readings):
                raise ValueError("Empty reference is not absence without two explicit scoped findings")
        base.update(source_pdf_sha256=decision["source_pdf_sha256"], complete=not any(inadequate),
                    first_pass_complete=True, second_pass_complete=True)
        reference_docs.append(base)
    validate_visual_positive_retention(decisions, catalog)
    return {"reference": {"version": VERSION, "locked": True, "reference_type": "automatic_source_review",
        "predictions_seen_before_lock": False, "is_human_or_expert": False, "independent_reviewer": False,
        "independence_note": draft["independence_note"], "documents": reference_docs},
        "source_population_sha256": fingerprint(documents), "coverage_ledger_sha256": fingerprint(ledger),
        "draft_sha256": fingerprint(draft), "visual_evidence_files": evidence_files,
        "reviewer_independence_proven": False, "predictions_generated": False}
