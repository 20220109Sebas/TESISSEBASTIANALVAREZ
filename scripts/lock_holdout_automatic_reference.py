"""Reconcile archived source-only readings and lock a bounded automatic reference.

This runner never opens extractor code or prediction files. Explicit decisions
pin both inventories; unknown fields stay ungraded. Old checkpoints are immutable.
"""
import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import validate_source_document, verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes

ROOT = Path(__file__).resolve().parents[1]
EXTRA_REVIEWS = (
    "holdout_scope_second_source_pass_v1.json",
    "holdout_ui_second_source_pass_v1.json",
    "holdout_second_source_pass_v1.json",
)
FIELDS = ("billed_amount", "currency", "lookback_years")
SCOPES = {
    "enterprise_services", "outside_consultoria_de_obra",
    "outside_individual_professional", "insufficient_reference", "inaccessible_source",
}


def fingerprint(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def load_inputs(base, source):
    rows = json.loads((base / "review_progress.json").read_text())
    first_review = json.loads((base / "compiled_source_reviews.json").read_text())
    reviews = json.loads((base / "condition_second_source_reviews.json").read_text())
    reviews.append(json.loads((base / "second_source_review.json").read_text()))
    reviews.extend(json.loads((ROOT / "docs" / name).read_text()) for name in EXTRA_REVIEWS)
    for review in [first_review, *reviews]:
        if any(review.get(flag) is not False for flag in
               ("human_annotation", "expert_annotation", "predictions_consulted", "model_rules_consulted", "reference_locked")):
            raise ValueError("Only unlocked automatic source-only readings are accepted")
    second = {}
    for review in reviews:
        for doc in review["documents"]:
            if doc["opportunity_id"] in second:
                raise ValueError("Duplicate second reading; no silent deduplication")
            second[doc["opportunity_id"]] = (doc, review["independence_note"])
    first = {d["opportunity_id"]: d for d in first_review["documents"]}
    sources = {d["opportunity_id"]: d for d in json.loads((source / "source_documents.json").read_text())}
    evidence = json.loads((base / "second_pass_evidence.json").read_text())
    return rows, first, second, sources, evidence


def canonical_first_value(raw, field):
    if raw is None:
        return None
    if field == "billed_amount":
        value = Decimal(str(raw))
        if not value.is_finite() or value <= 0 or value != value.quantize(Decimal("0.01")):
            raise ValueError("First amount cannot be normalized without rounding")
        return format(value, ".2f")
    return raw


def validate(policy, rows, first, second, sources):
    if (policy.get("version") != "holdout_source_reconciliation_v1"
            or policy.get("reference_type") != "automatic_source_review"
            or policy.get("request_reference_lock") is not True):
        raise ValueError("Explicit bounded automatic lock decision required")
    for flag in ("is_human_or_expert", "independent_reviewer", "predictions_consulted", "model_rules_consulted"):
        if policy.get(flag) is not False:
            raise ValueError("Do not claim independence or consult predictions")
    for name in ("reviewer", "independence_note", "annotation_boundary", "warning_policy", "count_policy", "scope_policy"):
        if not policy.get(name):
            raise ValueError("Disclosure and annotation policies required")
    selected = {r["opportunity_id"]: r for r in rows}
    decisions = policy["documents"]
    if (len(rows) != 50 or len(selected) != 50 or len(decisions) != 50
            or len({d["opportunity_id"] for d in decisions}) != 50
            or {d["opportunity_id"] for d in decisions} != set(selected)):
        raise ValueError("Preserve every case in the fixed fifty-case denominator")
    if len(second) != 27 or set(second) != set(sources):
        raise ValueError("All supplied sources require their second reading")
    for decision in decisions:
        key = decision["opportunity_id"]
        row = selected[key]
        if (decision["sample_number"] != row["sample_number"]
                or decision["source_status"] != row["source_status"]
                or decision["scope"] not in SCOPES or not decision.get("reason")):
            raise ValueError("Case identity, scope and rationale must be explicit")
        if row["reference_locked"] is not False or row["predictions_generated"] is not False or row["historical_ranking_eligible"] is not False:
            raise ValueError("Cannot relabel a used reference or unlock historical ranking")
        prior = first.get(key, {}).get("bidder_experience_conditions", [])
        if fingerprint(prior) != decision["first_conditions_sha256"]:
            raise ValueError("First inventory differs from adjudicated snapshot")
        if key not in sources:
            if (decision["scope"] != "inaccessible_source" or decision["conditions"]
                    or decision["second_document_sha256"] is not None or decision["condition_hashes"]):
                raise ValueError("Inaccessible cases cannot acquire positive or negative references")
            continue
        source = sources[key]
        validate_source_document(source)
        doc = second[key][0]
        if (doc["sample_number"] != row["sample_number"]
                or row["first_pass_status"] != "full_supplied_pdf"
                or row["second_pass_status"] != "full_supplied_pdf"
                or doc.get("full_supplied_document_read") is not True
                or doc.get("entire_procurement_file_reviewed") is not False
                or set(doc["pages_read"]) != {p["page"] for p in source["pages"]}
                or len(doc["pages_read"]) != len(source["pages"])
                or row["physical_pages"] != len(source["pages"])):
            raise ValueError("Two full supplied-document readings required")
        if fingerprint(doc) != decision["second_document_sha256"]:
            raise ValueError("Second reading differs from adjudicated snapshot")
        current = doc["bidder_experience_conditions"]
        hashes = [fingerprint(c) for c in current]
        if hashes != decision["condition_hashes"]:
            raise ValueError("Condition snapshot changed")
        choices = decision["conditions"]
        if sorted(c["second_condition_index"] for c in choices) != list(range(len(current))):
            raise ValueError("Every second condition must be retained exactly once")
        mapped = [c["first_condition_index"] for c in choices]
        if len(mapped) != len(set(mapped)) or any(type(i) is not int or not 0 <= i < len(prior) for i in mapped):
            raise ValueError("Explicit unique first-pass correspondence required")
        if sorted(mapped) != list(range(len(prior))):
            if (decision["unmapped_first_condition_policy"] != "retain_other_roles_ungraded_outside_formal_scope"
                    or decision["scope"] != "outside_consultoria_de_obra"):
                raise ValueError("Cannot silently omit a first-pass condition")
        elif decision["unmapped_first_condition_policy"] != "require_one_to_one":
            raise ValueError("Unexpected mapping exception")
        previous_domain = first.get(key, {}).get("domain", "")
        if str(previous_domain).startswith("consultoria_de_obra"):
            if decision["scope"] != "outside_consultoria_de_obra":
                raise ValueError("Works consultancy is outside the frozen formal services scope")
        elif doc["decision"] == "outside_business_services_domain":
            if decision["scope"] != "outside_individual_professional":
                raise ValueError("Individual-professional finding cannot become a negative")
        elif doc["decision"] in ("abstain_missing_specific_terms", "abstain_unsuitable_source"):
            if decision["scope"] != "insufficient_reference" or current:
                raise ValueError("Insufficient sources must remain ungraded")
        elif decision["scope"] != "enterprise_services":
            raise ValueError("Do not exclude a service merely because its fields are unknown")
        if not current and decision["scope"] == "enterprise_services":
            if (doc["decision"] != "explicit_no_qualification_document_scope"
                    or row["document_absence_confirmed_two_pass"] is not True):
                raise ValueError("A negative reference needs explicit scoped two-pass support")
        identity = set()
        evidence_ids = {e.get("id") for e in doc["evidence"]}
        for choice in choices:
            condition = current[choice["second_condition_index"]]
            if (condition["subject"] != "bidder"
                    or condition["section_role"] not in ("qualification", "tdr_supplier_profile")
                    or condition.get("company_applicability") != "not_determined"):
                raise ValueError("Do not transfer personnel, annex or company eligibility")
            ident = (condition["subject"], condition["section_role"], condition.get("item"), condition["branch"])
            if ident in identity:
                raise ValueError("Duplicate condition identity requires explicit resolution")
            identity.add(ident)
            if not choice.get("identity_resolution") or set(choice["fields"]) != set(FIELDS) or set(choice["first_field_paths"]) != set(FIELDS):
                raise ValueError("Every field and correspondence needs a decision")
            if not condition.get("evidence_ids") or not set(condition["evidence_ids"]) <= evidence_ids:
                raise ValueError("Conditions require their existing literal evidence")
            for field in FIELDS:
                value = choice["fields"][field]
                if value != condition.get(field):
                    raise ValueError("Cannot change reviewed values or fill unknowns at lock")
                allowed_paths = {"billed_amount": {"billed_amount", "amount"},
                                 "currency": {"currency"},
                                 "lookback_years": {"lookback_years", "lookback_years_explicit_in_qualification"}}
                path = choice["first_field_paths"][field]
                if path not in allowed_paths[field]:
                    raise ValueError("Undeclared first-field translation")
                old_value = canonical_first_value(prior[choice["first_condition_index"]].get(path), field)
                if value is not None and old_value is not None and value != old_value:
                    raise ValueError("A known cross-pass disagreement cannot be silently resolved at lock")
                reason = choice["unknown_reasons"].get(field)
                if (value is None) != bool(reason):
                    raise ValueError("Unknown fields need reasons; known values are not unknown")
                if value is None:
                    continue
                if field == "billed_amount" and (not isinstance(value, str) or not re.fullmatch(r"\d+\.\d{2}", value) or Decimal(value) <= 0):
                    raise ValueError("Positive exact decimal required, no zero or rounding")
                if field == "currency" and (not isinstance(value, str) or not re.fullmatch(r"[A-Z]{3}", value)):
                    raise ValueError("Literal currency code required")
                if field == "lookback_years" and (type(value) is not int or value <= 0):
                    raise ValueError("Explicit positive integer window required")


def attach_evidence(source, records):
    for item in records:
        page = source["pages"][item["page"] - 1]
        if (page["text"].count(item["quote"]) != 1
                or page["text"][item["char_start"]:item["char_end"]] != item["quote"]
                or page["text_sha256"] != item["page_text_sha256"]
                or source["source"]["sha256"] != item["pdf_sha256"]
                or item["human_or_expert"] is not False):
            raise ValueError("Archived source evidence changed or lost its literal provenance")
        verify_hashes({item["image_path"]: item["image_sha256"]})
    if not records:
        raise ValueError("A supplied document needs existing visually checked evidence")
    return copy.deepcopy(records)


def build(base, source_dir, policy_path, preflight, output):
    if output.exists():
        raise ValueError("Use a fresh lock directory; never replace a reference")
    inputs = dict(protected_outputs())
    for directory in (base, source_dir, preflight):
        verify_artifact(directory, inputs)
        inputs.update(json.loads((directory / "manifest.json").read_text())["input_sha256"])
    previous = json.loads((base / "summary.json").read_text())
    if any(previous[k] is not False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified")):
        raise ValueError("Cannot reconcile after holdout inference")
    prelock = json.loads((preflight / "evaluation_lock.json").read_text())
    protocol_path = Path(prelock["protocol_path"])
    verify_hashes({str(protocol_path): prelock["protocol_sha256"], **prelock["comparison_code_sha256"]})
    protocol = json.loads(protocol_path.read_text())
    if (protocol["fields"] != list(FIELDS) or protocol["reference_type"] != "automatic_source_review"
            or protocol["predictions_generated"] is not False):
        raise ValueError("Preserve the precommitted three-field comparison protocol")
    model_lock = ROOT / "artifacts/experience_holdout_models_lock_v1/model_lock.json"
    frozen = json.loads(model_lock.read_text())["original_code_sha256"]
    verify_hashes(frozen)
    inputs.update(frozen)
    for path in (policy_path, Path(__file__).resolve(), model_lock, protocol_path):
        inputs[str(path)] = sha(path)
    inputs.update(prelock["comparison_code_sha256"])
    for name in EXTRA_REVIEWS:
        path = ROOT / "docs" / name
        if str(path) not in inputs or inputs[str(path)] != sha(path):
            raise ValueError("Scope/absence readings must be protected by prior lineage")
    policy = json.loads(policy_path.read_text())
    rows, first, second, sources, archived_evidence = load_inputs(base, source_dir)
    validate(policy, rows, first, second, sources)
    documents, ledger = [], []
    for decision in sorted(policy["documents"], key=lambda d: d["sample_number"]):
        key = decision["opportunity_id"]
        source = sources.get(key)
        reviewed = second.get(key)
        doc = reviewed[0] if reviewed else None
        prior = first.get(key, {}).get("bidder_experience_conditions", [])
        evidence = attach_evidence(source, [e for e in archived_evidence if e["opportunity_id"] == key]) if source else []
        conditions, comparisons = [], []
        for choice in decision["conditions"]:
            original = doc["bidder_experience_conditions"][choice["second_condition_index"]]
            old = prior[choice["first_condition_index"]]
            states = {}
            for field in FIELDS:
                path = choice["first_field_paths"][field]
                first_raw = old.get(path)
                first_value = canonical_first_value(first_raw, field)
                current = choice["fields"][field]
                status = ("unknown_ungraded" if current is None else
                          "two_pass_same_value_after_explicit_mapping" if first_value == current else
                          "source_supported_second_reading_not_full_two_pass_field_agreement")
                states[field] = {"value": current, "first_raw": first_raw, "first_field_path": path,
                                 "first_normalized": first_value, "status": status,
                                 "unknown_reason": choice["unknown_reasons"].get(field)}
            cited = [e for e in evidence if e.get("evidence_id") in original["evidence_ids"]]
            if {e.get("evidence_id") for e in cited} != set(original["evidence_ids"]):
                raise ValueError("Missing registered evidence for a condition")
            conditions.append({"opportunity_id": key, "subject": original["subject"],
                "section_role": original["section_role"], "item": original.get("item"), "branch": original["branch"],
                "fields": copy.deepcopy(choice["fields"]), "field_decisions": states,
                "source_annotation": copy.deepcopy(original), "evidence": cited,
                "first_pass_condition": copy.deepcopy(old), "second_condition_sha256": fingerprint(original),
                "identity_resolution": choice["identity_resolution"], "company_applicability": "not_determined"})
            comparisons.append({"first_condition_index": choice["first_condition_index"],
                                "second_condition_index": choice["second_condition_index"], "fields": states})
        eligible = decision["scope"] == "enterprise_services"
        documents.append({"opportunity_id": key, "sample_number": decision["sample_number"],
            "scope": decision["scope"], "complete": bool(source and eligible),
            "first_pass_complete": bool(source), "second_pass_complete": bool(source),
            "annotation_scope": policy["annotation_boundary"], "source_status": decision["source_status"],
            "reason": decision["reason"], "conditions": conditions,
            "source_pdf_path": str((ROOT / source["source"]["file_path"]).resolve()) if source else None,
            "source_pdf_sha256": source["source"]["sha256"] if source else None,
            "reading_coverage": {"pages_read": doc["pages_read"],
                "complete_rendered_pages_read": doc["complete_rendered_pages_read"]} if doc else None,
            "source_warnings": copy.deepcopy(doc.get("source_warnings", [])) if doc else [],
            "second_pass_independence_note": reviewed[1] if reviewed else None,
            "archived_second_reading": copy.deepcopy(doc), "evidence": evidence,
            "entire_procurement_file_reviewed": False, "historical_content_proven": False,
            "ranking_eligible": False, "company_applicability": "not_determined"})
        mapped = {c["first_condition_index"] for c in decision["conditions"]}
        ledger.append({**copy.deepcopy(decision), "field_comparisons": comparisons,
            "original_first_conditions": copy.deepcopy(prior),
            "unmapped_first_conditions_retained": [c for i, c in enumerate(prior) if i not in mapped],
            "previous_progress_row": copy.deepcopy(next(r for r in rows if r["opportunity_id"] == key)),
            "formal_field_comparison_eligible": eligible, "reference_is_expert_gold": False})
    reference = {"version": "holdout_automatic_reference_lock_v1", "locked": True,
        "predictions_seen_before_lock": False, "is_human_or_expert": False,
        "reference_type": "automatic_source_review", "independent_reviewer": False,
        "reviewer": policy["reviewer"], "independence_note": policy["independence_note"],
        "interpretation": "concordance_with_bounded_automatic_reference_not_independent_semantic_accuracy",
        "fields": list(FIELDS), "documents": documents, "predictions_generated": False,
        "historical_ranking_eligibility": False}
    eligible_docs = [d for d in documents if d["scope"] == "enterprise_services"]
    eligible_conditions = [c for d in eligible_docs for c in d["conditions"]]
    summary = {"selected_cases": 50, "supplied_pdfs_with_two_readings": len(sources),
        "scope_counts": dict(Counter(d["scope"] for d in documents)),
        "evaluable_documents": len(eligible_docs), "positive_condition_documents": sum(bool(d["conditions"]) for d in eligible_docs),
        "scoped_absence_documents": sum(not d["conditions"] for d in eligible_docs),
        "all_retained_source_conditions": sum(len(d["conditions"]) for d in documents),
        "formal_reference_conditions": len(eligible_conditions),
        "known_field_support": {f: sum(c["fields"][f] is not None for c in eligible_conditions) for f in FIELDS},
        "unknown_fields": {f: sum(c["fields"][f] is None for c in eligible_conditions) for f in FIELDS},
        "supplier_count_conditions_retained_not_in_three_field_metrics": sum(c["section_role"] == "tdr_supplier_profile" for c in eligible_conditions),
        "physical_pages_read": sum(len(d["pages"]) for d in sources.values()),
        "ocr_pages_with_text": sum(p["ocr_used"] for d in sources.values() for p in d["pages"]),
        "reference_locked": True, "reference_ready_for_frozen_inference": True,
        "predictions_generated": False, "f1_computed": False, "f1_estimable": False,
        "ranking_inputs_modified": False, "semantic_accuracy_independently_proven": False,
        "meaning_of_f1_estimable_false": "No predictions or metric computation in this source-only lock stage; positive support is reported separately."}
    verify_hashes(inputs)
    output.mkdir(parents=True)
    for name, value in (("reference.json", reference), ("reconciliation_ledger.json", ledger), ("summary.json", summary)):
        dump(output / name, value)
    dump(output / "reference_lock.json", {"reference_sha256": sha(output / "reference.json"),
        "reconciliation_policy_sha256": sha(policy_path), "protocol_path": str(protocol_path),
        "protocol_sha256": prelock["protocol_sha256"], "comparison_code_sha256": prelock["comparison_code_sha256"],
        "frozen_model_code_sha256": frozen, "reference_locked": True,
        "predictions_seen_before_lock": False, "predictions_generated": False,
        "unknown_field_policy": protocol["unknown_field_policy"],
        "annotation_boundary": policy["annotation_boundary"], "count_policy": policy["count_policy"]})
    dump(output / "manifest.json", {"input_sha256": inputs, "script_sha256": sha(__file__),
        "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "source", "decisions", "preflight", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base.resolve(), args.source.resolve(), args.decisions.resolve(),
                           args.preflight.resolve(), args.output.resolve())))
