import copy
import json
from pathlib import Path

import pytest

from scripts.lock_holdout_automatic_reference import build, fingerprint, load_inputs, validate
from tesis_recommender.reference_evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v47"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
POLICY = ROOT / "docs/holdout_reference_reconciliation_v1.json"
PREFLIGHT = ROOT / "artifacts/experience_holdout_evaluation_preflight_v1"


def inputs():
    return json.loads(POLICY.read_text()), load_inputs(BASE, SOURCE)


def test_explicit_reconciliation_covers_all_fifty_without_replacement():
    policy, data = inputs()
    validate(policy, *data[:4])
    docs = policy["documents"]
    assert len(docs) == 50
    assert sum(d["scope"] == "enterprise_services" for d in docs) == 16
    assert sum(d["scope"] == "inaccessible_source" for d in docs) == 23
    assert sum(d["scope"] == "insufficient_reference" for d in docs) == 3
    assert sum(d["scope"] == "outside_consultoria_de_obra" for d in docs) == 5
    assert sum(d["scope"] == "outside_individual_professional" for d in docs) == 3
    assert sum(len(d["conditions"]) for d in docs) == 30
    assert all(d["reason"] for d in docs)
    assert not policy["predictions_consulted"] and not policy["independent_reviewer"]


def test_ambiguities_stay_unknown_and_supplier_count_not_facturation():
    policy, _ = inputs()
    docs = {d["sample_number"]: d for d in policy["documents"]}
    assert docs[13]["conditions"][1]["fields"]["lookback_years"] is None
    assert docs[20]["conditions"][0]["fields"]["billed_amount"] is None
    assert docs[41]["conditions"][1]["fields"]["billed_amount"] is None
    assert docs[21]["conditions"][0]["fields"] == dict.fromkeys(("billed_amount", "currency", "lookback_years"))
    assert all(docs[n]["conditions"][0]["fields"]["billed_amount"] is None for n in (1, 11, 18, 34, 50))
    assert "no está vinculado" in docs[17]["reason"]
    assert "no como resolución de precedencia" in docs[2]["reason"]


@pytest.mark.parametrize("mutation", [
    "seen_predictions", "expert", "independent", "missing_case", "duplicate_case",
    "omit_condition", "duplicate_condition", "first_snapshot", "second_snapshot",
    "condition_snapshot", "consultancy_as_service", "individual_as_negative",
    "insufficient_as_negative", "inaccessible_as_negative", "exclude_unknown_service",
    "resolve_ambiguous_money", "inherit_mype_window", "unknown_without_reason",
    "wrong_first_match", "invent_field_translation", "omit_first_roles",
])
def test_lock_rejects_unsafe_reconciliation(mutation):
    policy, data = inputs()
    docs = {d["sample_number"]: d for d in policy["documents"]}
    if mutation == "seen_predictions":
        policy["predictions_consulted"] = True
    elif mutation == "expert":
        policy["is_human_or_expert"] = True
    elif mutation == "independent":
        policy["independent_reviewer"] = True
    elif mutation == "missing_case":
        policy["documents"].pop()
    elif mutation == "duplicate_case":
        policy["documents"][-1] = copy.deepcopy(policy["documents"][0])
    elif mutation == "omit_condition":
        docs[20]["conditions"].pop()
    elif mutation == "duplicate_condition":
        docs[20]["conditions"].append(copy.deepcopy(docs[20]["conditions"][0]))
    elif mutation == "first_snapshot":
        docs[13]["first_conditions_sha256"] = "0" * 64
    elif mutation == "second_snapshot":
        docs[13]["second_document_sha256"] = "0" * 64
    elif mutation == "condition_snapshot":
        docs[13]["condition_hashes"][0] = "0" * 64
    elif mutation == "consultancy_as_service":
        docs[11]["scope"] = "enterprise_services"
    elif mutation == "individual_as_negative":
        docs[7]["scope"] = "enterprise_services"
    elif mutation == "insufficient_as_negative":
        docs[4]["scope"] = "enterprise_services"
    elif mutation == "inaccessible_as_negative":
        docs[3]["scope"] = "enterprise_services"
    elif mutation == "exclude_unknown_service":
        docs[21]["scope"] = "insufficient_reference"
    elif mutation == "resolve_ambiguous_money":
        docs[41]["conditions"][1]["fields"]["billed_amount"] = "21995.00"
    elif mutation == "inherit_mype_window":
        docs[13]["conditions"][1]["fields"]["lookback_years"] = 15
    elif mutation == "unknown_without_reason":
        docs[13]["conditions"][1]["unknown_reasons"] = {}
    elif mutation == "wrong_first_match":
        for choice in docs[2]["conditions"]:
            choice["first_condition_index"] = 1 - choice["first_condition_index"]
    elif mutation == "invent_field_translation":
        docs[16]["conditions"][0]["first_field_paths"]["billed_amount"] = "quantia_context"
    elif mutation == "omit_first_roles":
        docs[18]["unmapped_first_condition_policy"] = "require_one_to_one"
    with pytest.raises(ValueError):
        validate(policy, *data[:4])


def test_normalized_first_mismatch_cannot_be_hidden_by_repinning():
    policy, data = inputs()
    _, first, _, _, _ = data
    d = next(d for d in policy["documents"] if d["sample_number"] == 16)
    first[d["opportunity_id"]]["bidder_experience_conditions"][0]["billed_amount"] = 3000001
    d["first_conditions_sha256"] = fingerprint(first[d["opportunity_id"]]["bidder_experience_conditions"])
    with pytest.raises(ValueError, match="known cross-pass disagreement"):
        validate(policy, *data[:4])


@pytest.fixture(scope="module")
def locked_reference(tmp_path_factory):
    output = tmp_path_factory.mktemp("holdout-lock") / "reference"
    summary = build(BASE, SOURCE, POLICY, PREFLIGHT, output)
    return output, summary, json.loads((output / "reference.json").read_text())


def test_locked_reference_has_exact_support_and_no_real_metrics(locked_reference):
    _, summary, ref = locked_reference
    assert ref["locked"] and not ref["predictions_seen_before_lock"]
    assert summary["evaluable_documents"] == 16
    assert summary["positive_condition_documents"] == 15 and summary["scoped_absence_documents"] == 1
    assert summary["formal_reference_conditions"] == 25 and summary["all_retained_source_conditions"] == 30
    assert summary["known_field_support"] == {"billed_amount": 22, "currency": 24, "lookback_years": 23}
    assert summary["unknown_fields"] == {"billed_amount": 3, "currency": 1, "lookback_years": 2}
    assert summary["physical_pages_read"] == 1667 and summary["ocr_pages_with_text"] == 459
    assert summary["supplier_count_conditions_retained_not_in_three_field_metrics"] == 1
    assert not summary["predictions_generated"] and not summary["f1_computed"] and not summary["f1_estimable"]
    assert not summary["ranking_inputs_modified"] and not summary["semantic_accuracy_independently_proven"]
    assert all(not d["ranking_eligible"] and not d["entire_procurement_file_reviewed"] for d in ref["documents"])


def test_legacy_mapping_and_original_records_preserved(locked_reference):
    output, _, ref = locked_reference
    docs = {d["sample_number"]: d for d in ref["documents"]}
    c = docs[16]["conditions"][0]
    assert c["first_pass_condition"]["billed_amount"] == 3000000
    assert c["fields"]["billed_amount"] == "3000000.00"
    assert c["field_decisions"]["billed_amount"]["status"] == "two_pass_same_value_after_explicit_mapping"
    assert docs[17]["conditions"][0]["item"] is None
    assert docs[17]["conditions"][0]["first_pass_condition"]["item"] == "01"
    assert docs[13]["conditions"][0]["first_pass_condition"]["condition_id"] == "13-general"
    assert docs[41]["conditions"][1]["first_pass_condition"]["id"] == "41-mype"
    assert docs[21]["conditions"][0]["source_annotation"]["minimum_similar_services"] == 3
    assert docs[21]["conditions"][0]["section_role"] == "tdr_supplier_profile"
    assert docs[2]["conditions"][0]["fields"]["lookback_years"] == 15
    assert docs[2]["source_warnings"][0]["code"] == "tdr_qualification_lookback_conflict"
    ledger = json.loads((output / "reconciliation_ledger.json").read_text())
    assert len(next(d for d in ledger if d["sample_number"] == 18)["unmapped_first_conditions_retained"]) == 3
    rows = json.loads((BASE / "review_progress.json").read_text())
    assert all(not r["reference_locked"] for r in rows)
    lock = json.loads((output / "reference_lock.json").read_text())
    prelock = json.loads((PREFLIGHT / "evaluation_lock.json").read_text())
    assert lock["comparison_code_sha256"] == prelock["comparison_code_sha256"]
    assert lock["protocol_sha256"] == prelock["protocol_sha256"]


def test_frozen_comparator_accepts_reference_with_synthetic_only_predictions(locked_reference):
    _, _, ref = locked_reference
    # These are fabricated test rows, never held-out extractor outputs.
    predicted = [{k: c[k] for k in ("opportunity_id", "subject", "section_role", "item", "branch", "fields")}
                 for d in ref["documents"] if d["scope"] == "enterprise_services" for c in d["conditions"]]
    result = evaluate(ref, predicted, ref["fields"])
    assert result["evaluable_documents"] == 16 and result["reference_conditions"] == 25
    assert result["fields"]["billed_amount"]["tp"] == 22
    assert result["fields"]["billed_amount"]["reference_unknown"] == 3
    assert not result["semantic_accuracy_independently_proven"]
    modified = copy.deepcopy(predicted)
    target = next(c for c in modified if c["fields"]["billed_amount"] is None)
    target["fields"]["billed_amount"] = "999999.00"
    result = evaluate(ref, modified, ref["fields"])
    assert result["fields"]["billed_amount"]["predictions_ungraded_unknown_reference"] == 1
    assert result["fields"]["billed_amount"]["fp"] == 0


def test_only_scoped_absence_can_grade_a_synthetic_spurious_prediction(locked_reference):
    _, _, ref = locked_reference
    docs = {d["sample_number"]: d for d in ref["documents"]}
    synthetic = [{"opportunity_id": docs[n]["opportunity_id"], "subject": "bidder", "section_role": "qualification",
                  "item": None, "branch": "fake_test_only", "fields": {"billed_amount": "100.00"}}
                 for n in (3, 4, 7, 11, 45)]
    result = evaluate(ref, synthetic, ref["fields"])
    assert result["graded_predictions"] == 1 and result["predictions_outside_evaluable_scope"] == 4
    assert result["fields"]["billed_amount"]["fp"] == 1


def test_lock_is_byte_reproducible_and_cannot_overwrite(locked_reference, tmp_path):
    original, _, _ = locked_reference
    output = tmp_path / "reproduction"
    build(BASE, SOURCE, POLICY, PREFLIGHT, output)
    assert sorted(p.name for p in original.iterdir()) == sorted(p.name for p in output.iterdir())
    assert all((output / p.name).read_bytes() == p.read_bytes() for p in original.iterdir())
    with pytest.raises(ValueError, match="fresh lock directory"):
        build(BASE, SOURCE, POLICY, PREFLIGHT, output)
