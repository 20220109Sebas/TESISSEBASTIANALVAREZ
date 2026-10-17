"""Source-only reference gates and lossless adapter; never a new extractor.

The artifact runner must bind a reference lock into inference before calling
comparison. Declarations and hashes establish workflow lineage, not independent
annotation, semantic truth, or proof that a person never saw predictions.
"""
from __future__ import annotations

import copy
import re
from decimal import Decimal

from .document_experience_candidate_v1 import VERSION as CANDIDATE_VERSION, validate_document
from .document_experience_dev_v2 import verify_provenance
from .document_experience_units_dev_v1 import FIELDS, checksum, project_units, stable_json
from .reference_evaluation import evaluate, key

VERSION = "document_experience_evaluation_gates_v1"
SCOPES = {"enterprise_services", "outside_consultoria_de_obra", "outside_individual_professional",
          "insufficient_reference", "inaccessible_source"}
ROLES = {"qualification", "evaluation_factor", "tdr_supplier_profile"}


def fingerprint(value):
    return checksum(stable_json(value))


def require_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("A complete SHA-256 is required")


def source_population(documents, ledger):
    if len(ledger) != 50 or len({r["opportunity_id"] for r in ledger}) != 50:
        raise ValueError("Preserve all fifty selected cases in the coverage ledger")
    if any(not r["opportunity_id"].startswith("ocds-") or not r.get("source_status") for r in ledger):
        raise ValueError("Explicit source identity and coverage status required")
    expected = {r["opportunity_id"] for r in ledger if r["source_status"] == "complete_native_view"}
    if len({d["opportunity_id"] for d in documents}) != len(documents):
        raise ValueError("Repeated source document")
    if {d["opportunity_id"] for d in documents} != expected:
        raise ValueError("A complete source was omitted or an unavailable case promoted")
    for document in documents:
        validate_document(document)
    if len({d["source"]["sha256"] for d in documents}) != len(documents):
        raise ValueError("Duplicate PDF must remain excluded by the source pipeline")
    return {d["opportunity_id"]: d for d in documents}


def validate_evidence(ref, source):
    page_number, start, end = ref["page"], ref["char_start"], ref["char_end"]
    if any(type(n) is not int for n in (page_number, start, end)) or not 1 <= page_number <= len(source["pages"]):
        raise ValueError("Invalid literal evidence coordinates")
    page = source["pages"][page_number - 1]
    if (not 0 <= start < end <= len(page["text"]) or page["text"][start:end] != ref["quote"]
            or ref["page_text_sha256"] != page["text_sha256"] or ref["pdf_sha256"] != source["source"]["sha256"]):
        raise ValueError("Literal evidence differs from the fixed source view")


def validate_condition(condition, source):
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
        validate_evidence(ref, source)
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
            validate_evidence(ref, source)
        if value is None:
            continue
        if field == "billed_amount" and (not isinstance(value, str)
                or not re.fullmatch(r"(?:0|[1-9][0-9]*)\.[0-9]{2}", value) or Decimal(value) <= 0):
            raise ValueError("No inferred, rounded or zero billed amount")
        if field == "currency" and value not in ("PEN", "USD"):
            raise ValueError("Unsupported literal currency")
        if field == "lookback_years" and (type(value) is not int or not 1 <= value <= 99):
            raise ValueError("No inferred lookback window")


def validate_reading(reading, source, evidence_files):
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
            validate_evidence(ref, source)
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
        validate_condition(condition, source)
    return inadequate


def build_reference(draft, documents, ledger):
    """Reconcile two archived readings, never predictions. Unknowns stay unknown."""
    sources = source_population(documents, ledger)
    if (draft.get("reference_type") != "automatic_source_review" or draft.get("locked") is not False
            or draft.get("predictions_seen_before_lock") is not False
            or draft.get("is_human_or_expert") is not False or draft.get("independent_reviewer") is not False
            or not draft.get("independence_note")):
        raise ValueError("Disclose automatic non-independent review before inference")
    decisions = draft["documents"]
    if len(decisions) != 50 or len({d["opportunity_id"] for d in decisions}) != 50 or {d["opportunity_id"] for d in decisions} != {r["opportunity_id"] for r in ledger}:
        raise ValueError("Reference must retain the complete fifty-case denominator")
    reference_docs, evidence_files = [], {}
    for decision in decisions:
        ocid, scope = decision["opportunity_id"], decision["scope"]
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
        inadequate = [validate_reading(r, source, evidence_files) for r in readings]
        if any(r["scope"] != scope for r in readings):
            raise ValueError("Resolve source-only scope disagreements before locking the reference")
        if any(inadequate) and scope != "insufficient_reference":
            raise ValueError("Unreadable pages cannot support positive or negative formal references")
        for c in decision["conditions"]:
            validate_condition(c, source)
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
    return {"reference": {"version": VERSION, "locked": True, "reference_type": "automatic_source_review",
        "predictions_seen_before_lock": False, "is_human_or_expert": False, "independent_reviewer": False,
        "independence_note": draft["independence_note"], "documents": reference_docs},
        "source_population_sha256": fingerprint(documents), "coverage_ledger_sha256": fingerprint(ledger),
        "draft_sha256": fingerprint(draft), "visual_evidence_files": evidence_files,
        "reviewer_independence_proven": False, "predictions_generated": False}


def adapt_predictions(candidate, documents):
    """No reference parameter, aliasing, best-match selection or value filling."""
    if (candidate.get("version") != CANDIDATE_VERSION or candidate.get("reference_or_expected_values_used") is not False
            or candidate.get("metrics_computed") is not False or candidate.get("ranking_eligible") is not False):
        raise ValueError("Only fixed source-only candidate outputs are accepted")
    sources = {d["opportunity_id"]: d for d in documents}
    if len(sources) != len(documents):
        raise ValueError("Repeated source document")
    if len({d["source"]["sha256"] for d in documents}) != len(documents):
        raise ValueError("Duplicate PDF source cannot enter comparison")
    outputs = candidate["document_outputs"]
    if len(outputs) != len(sources) or {d["opportunity_id"] for d in outputs} != set(sources):
        raise ValueError("Candidate source population differs")
    for output in outputs:
        source = sources[output["opportunity_id"]]
        validate_document(source)
        if output["source_pdf_sha256"] != source["source"]["sha256"]:
            raise ValueError("Prediction PDF lineage differs")
        verify_provenance(output["result"], source["pages"])
    coverage = [{"opportunity_id": d["opportunity_id"], "physical_pages": len(d["pages"]),
                 "empty_text_pages": [p["page"] for p in d["pages"] if not p["text"].strip()],
                 "unresolved_ocr_pages": [p["page"] for p in d["pages"] if p["selected_ocr_unresolved"]],
                 "all_document_content_proven_recovered": False, "absence_of_requirements_proven": False}
                for d in documents]
    if candidate["coverage"] != coverage:
        raise ValueError("Prediction coverage differs from the fixed source population")
    projected = project_units(outputs)
    if projected != candidate["occurrence_projection"]:
        raise ValueError("Occurrence projection changed or a gate was bypassed")
    if not projected["identity_gate"]["passed"]:
        return {"stage": "stopped_unresolved_identity", "predictions": None, "metrics_computed": False,
                "blocked_unit_ids": projected["identity_gate"]["blocked_unit_ids"],
                "original_occurrences": len(projected["occurrences"]), "candidate_units": len(projected["candidate_units"])}
    predictions = [dict(**{k: u["identity"][k] for k in ("opportunity_id", "subject", "section_role", "item", "branch")},
                        fields=copy.deepcopy(u["fields"])) for u in projected["candidate_units"]]
    if len({key(p) for p in predictions}) != len(predictions):
        raise ValueError("Comparator identity collision after projection")
    return {"stage": "adapted_without_reference_values", "predictions": predictions, "metrics_computed": False,
            "blocked_unit_ids": [], "original_occurrences": len(projected["occurrences"]),
            "candidate_units": len(predictions)}


def compare(locked_reference, candidate, documents, ledger, inference_binding, expected_binding):
    """Artifact runner supplies the reference-first hash binding, not just a flag."""
    source_population(documents, ledger)
    if inference_binding != expected_binding:
        raise ValueError("Inference was not bound to this pre-existing reference/source/protocol lock")
    for name in ("reference_manifest_sha256", "reference_sha256", "source_manifest_sha256", "evaluation_design_manifest_sha256"):
        require_hash(expected_binding[name])
    if (locked_reference["source_population_sha256"] != fingerprint(documents)
            or locked_reference["coverage_ledger_sha256"] != fingerprint(ledger)):
        raise ValueError("Reference and inference populations differ")
    adapted = adapt_predictions(candidate, documents)
    result = {"version": VERSION, "adapter": adapted, "metrics": None, "metrics_computed": False,
              "ranking_eligible": False, "expert_accuracy_proven": False}
    if adapted["predictions"] is None:
        result["stage"] = "stopped_unresolved_identity_no_metrics"
        return result
    reference = locked_reference["reference"]
    # Report streams separately; an unsupported TDR condition is a miss if known,
    # never silently rewritten as qualification. Unknown reference fields ungraded.
    streams = {}
    for role in sorted(ROLES):
        scoped = copy.deepcopy(reference)
        for doc in scoped["documents"]:
            doc["conditions"] = [c for c in doc["conditions"] if c["section_role"] == role]
        streams[role] = evaluate(scoped, [p for p in adapted["predictions"] if p["section_role"] == role], FIELDS)
    result.update(stage="automatic_reference_concordance_only", metrics=streams, metrics_computed=True)
    return result
