"""Compare selective predictions without removing abstained reference cases."""
import copy

from .document_experience_candidate_v2 import POLICY, VERSION as CANDIDATE_VERSION, decisions
from .document_experience_evaluation_v1 import FIELDS, ROLES, adapt_predictions, fingerprint, require_hash, source_population
from .reference_evaluation import evaluate

VERSION = "document_experience_selective_evaluation_v1"


def adapt(candidate, documents):
    if (candidate.get("version") != CANDIDATE_VERSION
            or candidate.get("reference_or_expected_values_used") is not False
            or candidate.get("metrics_computed") is not False or candidate.get("ranking_eligible") is not False
            or candidate.get("raw_text_parser_changed") is not False):
        raise ValueError("Use the fixed selective candidate without expected values")
    raw = candidate["raw_candidate_output"]
    # This verifies raw provenance, coverage and lossless projection. Its old
    # global stop is retained in raw, not retrospectively relabelled as a pass.
    adapt_predictions(raw, documents)
    expected = decisions(raw)
    if expected != candidate["selective_decisions"]:
        raise ValueError("Selective decisions differ from the frozen all-group policy")
    return expected


def compare(locked_reference, candidate, documents, ledger, inference_binding, expected_binding):
    source_population(documents, ledger)
    if inference_binding != expected_binding:
        raise ValueError("Inference was not bound to this pre-existing reference and design")
    for name in ("reference_manifest_sha256", "reference_sha256", "source_manifest_sha256", "evaluation_design_manifest_sha256"):
        require_hash(expected_binding[name])
    if (locked_reference["source_population_sha256"] != fingerprint(documents)
            or locked_reference["coverage_ledger_sha256"] != fingerprint(ledger)):
        raise ValueError("Reference and inference populations differ")
    selective = adapt(candidate, documents)
    reference = locked_reference["reference"]
    streams = {}
    for role in sorted(ROLES):
        scoped = copy.deepcopy(reference)
        for doc in scoped["documents"]:
            doc["conditions"] = [c for c in doc["conditions"] if c["section_role"] == role]
        streams[role] = evaluate(scoped, [p for p in selective["predictions"] if p["section_role"] == role], FIELDS)
    eligible = {d["opportunity_id"] for d in reference["documents"] if d["scope"] == "enterprise_services"}
    units = candidate["raw_candidate_output"]["occurrence_projection"]["candidate_units"]
    return {"version": VERSION, "policy": POLICY, "stage": "selective_automatic_reference_concordance_only",
            "metrics": streams, "metrics_computed": True,
            "prediction_decisions": {"candidate_groups": selective["candidate_groups"],
                "emitted_groups": len(selective["emitted_unit_ids"]), "abstained_groups": len(selective["abstentions"]),
                "candidate_groups_in_evaluable_scope": sum(u["identity"]["opportunity_id"] in eligible for u in units),
                "emitted_groups_in_evaluable_scope": sum(p["opportunity_id"] in eligible for p in selective["predictions"]),
                "abstained_groups_in_evaluable_scope": sum(a["identity"]["opportunity_id"] in eligible for a in selective["abstentions"]),
                "emitted_known_fields_in_evaluable_scope": {f: sum(p["opportunity_id"] in eligible and p["fields"][f] is not None
                    for p in selective["predictions"]) for f in FIELDS},
                "original_occurrences": selective["original_occurrences"]},
            "reference_documents_removed_due_to_abstention": 0, "reference_conditions_removed_due_to_abstention": 0,
            "uncertain_identity_resolved": False, "expert_accuracy_proven": False, "ranking_eligible": False}
