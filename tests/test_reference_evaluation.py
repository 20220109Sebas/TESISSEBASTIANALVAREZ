"""Synthetic controls only. These fixtures are not thesis observations."""
from copy import deepcopy

import pytest

from tesis_recommender.reference_evaluation import evaluate, metrics


def condition(ocid="example", **changes):
    row = {"opportunity_id": ocid, "subject": "bidder", "section_role": "qualification",
           "item": None, "branch": "general", "fields": {"amount": "100.00", "currency": "PEN", "years": 8}}
    row.update(changes)
    return row


def document(ocid="example", conditions=None, **changes):
    row = {"opportunity_id": ocid, "scope": "enterprise_services", "complete": True,
           "first_pass_complete": True, "second_pass_complete": True,
           "conditions": [condition(ocid)] if conditions is None else conditions}
    row.update(changes)
    return row


def reference(documents=None, **changes):
    row = {"locked": True, "predictions_seen_before_lock": False, "is_human_or_expert": False,
           "reference_type": "automatic_source_review", "documents": [document()] if documents is None else documents}
    row.update(changes)
    return row


def test_exact_match_counts_three_fields_not_three_documents():
    result = evaluate(reference(), [condition()], ["amount", "currency", "years"])
    assert result["evaluable_documents"] == result["reference_conditions"] == 1
    assert all(r["tp"] == 1 and r["fp"] == r["fn"] == 0 and r["f1"] == 1 for r in result["fields"].values())
    assert not result["semantic_accuracy_independently_proven"]
    assert not result["ranking_quality_evaluated"]


def test_wrong_value_counts_false_positive_and_false_negative():
    row = condition(fields={"amount": "200.00"})
    result = evaluate(reference(), [row], ["amount"])["fields"]["amount"]
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)
    assert result["precision"] == result["recall"] == result["f1"] == 0


def test_abstention_counts_only_missed_supported_value():
    result = evaluate(reference(), [condition(fields={"amount": None})], ["amount"])["fields"]["amount"]
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)
    assert result["precision"] is None and result["recall"] == result["f1"] == 0


@pytest.mark.parametrize("fields", [{"amount": None}, {}])
def test_unknown_reference_is_ungraded_not_automatically_correct(fields):
    ref = reference([document(conditions=[condition(fields=fields)])])
    result = evaluate(ref, [condition()], ["amount"])["fields"]["amount"]
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 0)
    assert result["reference_unknown"] == result["predictions_ungraded_unknown_reference"] == 1
    assert result["precision"] is result["recall"] is result["f1"] is None


@pytest.mark.parametrize("field,value", [("item", "2"), ("branch", "mype"),
    ("subject", "professional"), ("section_role", "evaluation_factor")])
def test_identity_mismatch_is_not_matched_by_same_amount(field, value):
    result = evaluate(reference(), [condition(**{field: value})], ["amount"])["fields"]["amount"]
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)


def test_complete_empty_reference_counts_spurious_value_but_no_recall_support():
    result = evaluate(reference([document(conditions=[])]), [condition()], ["amount"])["fields"]["amount"]
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 0)
    assert result["precision"] == 0 and result["recall"] is result["f1"] is None


def test_missing_prediction():
    result = evaluate(reference(), [], ["amount"])["fields"]["amount"]
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)


def test_noneligible_document_is_not_a_negative():
    docs = [document(), document("outside", scope="scope_pending", complete=False)]
    result = evaluate(reference(docs), [condition(), condition("outside"), condition("unavailable")], ["amount"])
    assert result["evaluable_documents"] == 1
    assert result["graded_predictions"] == 1
    assert result["predictions_outside_evaluable_scope"] == 2


@pytest.mark.parametrize("flag", ["complete", "first_pass_complete", "second_pass_complete"])
@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_incomplete_reference_never_enters_denominator(flag, value):
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate(reference([document(**{flag: value})]), [], ["amount"])


@pytest.mark.parametrize("changes", [{"locked": False}, {"locked": "true"},
    {"predictions_seen_before_lock": True}, {"predictions_seen_before_lock": None},
    {"is_human_or_expert": True}, {"is_human_or_expert": None}, {"reference_type": "expert"}])
def test_unlocked_or_mislabelled_reference_rejected(changes):
    with pytest.raises(ValueError, match="locked source-only"):
        evaluate(reference(**changes), [], ["amount"])


@pytest.mark.parametrize("flag", ["predictions_seen_before_lock", "is_human_or_expert"])
def test_missing_attestation_rejected(flag):
    row = reference()
    row.pop(flag)
    with pytest.raises(ValueError):
        evaluate(row, [], ["amount"])


def test_duplicate_document_rejected():
    with pytest.raises(ValueError, match="Duplicate reference document"):
        evaluate(reference([document(), document()]), [], ["amount"])


def test_reference_condition_cannot_be_swapped_between_eligible_documents():
    with pytest.raises(ValueError, match="another document"):
        evaluate(reference([document(conditions=[condition("other")]), document("other")]), [], ["amount"])


@pytest.mark.parametrize("target", ["reference", "prediction"])
def test_duplicate_identity_requires_explicit_resolution(target):
    ref = reference([document(conditions=[condition(), condition()])]) if target == "reference" else reference()
    predictions = [condition(), condition()] if target == "prediction" else []
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(ref, predictions, ["amount"])


@pytest.mark.parametrize("fields", [[], ["amount", "amount"]])
def test_invalid_requested_fields(fields):
    with pytest.raises(ValueError, match="Fields"):
        evaluate(reference(), [], fields)


def test_inputs_not_mutated_and_macro_not_invented():
    ref, predictions = reference(), [condition()]
    before = deepcopy((ref, predictions))
    result = evaluate(ref, predictions, ["amount"])
    assert (ref, predictions) == before
    assert "overall_accuracy" not in result and "macro_f1" not in result


def test_empty_denominator_metrics_are_undefined():
    assert metrics(0, 0, 0) == {"tp": 0, "fp": 0, "fn": 0, "precision": None, "recall": None, "f1": None}
