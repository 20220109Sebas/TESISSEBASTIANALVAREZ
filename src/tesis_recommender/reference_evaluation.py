"""Literal field concordance for a completed automatic reference, not expert accuracy.

No incomplete or single-pass document is silently treated as a gold reference.
This module does not perform matching, extraction or inference on source PDFs.
"""
from collections import Counter

VERSION = "automatic_reference_field_concordance_v1"


def key(row):
    return (row["opportunity_id"], row["subject"], row["section_role"], row.get("item"), row["branch"])


def metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn and 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate(reference, predictions, fields):
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("Fields must be nonempty and unique")
    if (reference.get("locked") is not True
            or reference.get("predictions_seen_before_lock") is not False
            or reference.get("is_human_or_expert") is not False
            or reference.get("reference_type") != "automatic_source_review"):
        raise ValueError("Reference must be locked source-only automatic review")
    docs = reference["documents"]
    if len({d["opportunity_id"] for d in docs}) != len(docs):
        raise ValueError("Duplicate reference document")
    eligible = set()
    for doc in docs:
        if doc["scope"] != "enterprise_services":
            continue
        if any(doc.get(flag) is not True for flag in
               ("complete", "first_pass_complete", "second_pass_complete")):
            raise ValueError("Incomplete enterprise-service reference cannot be evaluated")
        if any(row["opportunity_id"] != doc["opportunity_id"] for row in doc["conditions"]):
            raise ValueError("Reference condition belongs to another document")
        eligible.add(doc["opportunity_id"])
    truth = [r for d in docs if d["opportunity_id"] in eligible for r in d["conditions"]]
    if any(r["opportunity_id"] not in eligible for r in truth):
        raise ValueError("Reference condition belongs to another document")
    # Same item and condition are part of identity, not a score to be optimized.
    if len({key(r) for r in truth}) != len(truth):
        raise ValueError("Duplicate reference identity requires adjudication")
    graded_predictions = [r for r in predictions if r["opportunity_id"] in eligible]
    if len({key(r) for r in graded_predictions}) != len(graded_predictions):
        raise ValueError("Duplicate prediction identity requires explicit handling")
    gold = {key(r): r for r in truth}
    pred = {key(r): r for r in graded_predictions}
    counts = {f: Counter(tp=0, fp=0, fn=0, reference_unknown=0,
                         predictions_ungraded_unknown_reference=0) for f in fields}
    for identity in sorted(set(gold) | set(pred), key=repr):
        g, p = gold.get(identity), pred.get(identity)
        for field in fields:
            known = g is not None and field in g["fields"] and g["fields"][field] is not None
            emitted = p is not None and field in p["fields"] and p["fields"][field] is not None
            if g is not None and not known:
                counts[field]["reference_unknown"] += 1
                counts[field]["predictions_ungraded_unknown_reference"] += int(emitted)
                continue
            if known and emitted and g["fields"][field] == p["fields"][field]:
                counts[field]["tp"] += 1
            else:
                counts[field]["fn"] += int(known)
                counts[field]["fp"] += int(emitted)
    return {"version": VERSION, "metric_interpretation": "concordance_with_automatic_reference_not_expert_accuracy",
        "evaluable_documents": len(eligible), "reference_conditions": len(truth),
        "graded_predictions": len(graded_predictions),
        "predictions_outside_evaluable_scope": len(predictions) - len(graded_predictions),
        "fields": {f: dict(metrics(c["tp"], c["fp"], c["fn"]),
                           reference_unknown=c["reference_unknown"],
                           predictions_ungraded_unknown_reference=c["predictions_ungraded_unknown_reference"])
                   for f, c in counts.items()},
        "semantic_accuracy_independently_proven": False, "ranking_quality_evaluated": False}
