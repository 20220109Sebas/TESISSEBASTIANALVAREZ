import copy

import pytest

from tesis_recommender.document_experience_candidate_v2 import decisions, predict_documents
from tesis_recommender.document_experience_dev_v1 import digest
from tesis_recommender.document_experience_evaluation_v1 import fingerprint
from tesis_recommender.document_experience_selective_evaluation_v1 import adapt, compare

HEADER = "3.2 REQUISITOS DE CALIFICACIÓN\nEXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD\n"
GENERAL = "El postor debe acreditar un monto facturado acumulado equivalente a "
WINDOW = " durante los ocho (8) años anteriores a la fecha de presentación de ofertas."
CLAUSE = GENERAL + "S/ 900.00." + WINDOW
FIELDS = {"billed_amount": "900.00", "currency": "PEN", "lookback_years": 8}


def source(text, ocid="ocds-selective-test", pdf="a" * 64):
    page = {"page": 1, "text": text, "text_sha256": digest(text), "native_text": text,
            "native_text_sha256": digest(text), "ocr_selected": False, "ocr_used": False,
            "selected_ocr_unresolved": False, "extraction_method": "native_pdf_text"}
    return {"opportunity_id": ocid, "source": {"opportunity_id": ocid, "status": "downloaded_pdf",
            "sha256": pdf, "bytes": 1234}, "pages": [page], "total_pages": 1,
            "processed_pages": 1, "truncated_by_page_limit": False,
            "ranking_eligible": False, "historical_content_proven": False}


def gold(document, fields=None):
    ocid = document["opportunity_id"]
    condition = {"opportunity_id": ocid, "subject": "bidder", "section_role": "qualification",
                 "item": None, "branch": "general", "fields": FIELDS if fields is None else fields}
    reference = {"locked": True, "predictions_seen_before_lock": False, "is_human_or_expert": False,
                 "reference_type": "automatic_source_review", "documents": [{"opportunity_id": ocid,
                 "scope": "enterprise_services", "complete": True, "first_pass_complete": True,
                 "second_pass_complete": True, "conditions": [condition]}]}
    ledger = [{"opportunity_id": ocid, "source_status": "complete_native_view"}] + [
              {"opportunity_id": "ocds-unavailable-" + str(n), "source_status": "inaccessible"} for n in range(49)]
    reference["documents"] += [{"opportunity_id": r["opportunity_id"], "scope": "inaccessible_source",
                                "conditions": []} for r in ledger[1:]]
    return {"reference": reference, "source_population_sha256": fingerprint([document]),
            "coverage_ledger_sha256": fingerprint(ledger)}, ledger


def compare_synthetic(document, fields=None):
    reference, ledger = gold(document, fields)
    binding = {name: "b" * 64 for name in ("reference_manifest_sha256", "reference_sha256",
               "source_manifest_sha256", "evaluation_design_manifest_sha256")}
    before = copy.deepcopy(reference)
    result = compare(reference, predict_documents([document]), [document], ledger, binding, binding)
    assert reference == before
    return result


def test_single_assertion_keeps_fields_without_reference():
    document = source(HEADER + CLAUSE)
    before = copy.deepcopy(document)
    result = predict_documents([document])
    selective = adapt(result, [document])
    assert selective["predictions"][0]["fields"] == FIELDS
    assert len(selective["emitted_unit_ids"]) == 1 and not selective["abstentions"]
    assert document == before and result["metrics_computed"] is False


def test_strict_repeat_in_one_context_preserves_both_occurrences():
    result = predict_documents([source(HEADER + CLAUSE + "\n" + CLAUSE)])
    decision = result["selective_decisions"]
    assert decision["original_occurrences"] == 2 and len(decision["predictions"]) == 1
    assert len(decision["source_contexts"]) == 1
    assert len(decision["source_contexts"][0]["occurrence_ids"]) == 2
    assert not decision["abstentions"]


@pytest.mark.parametrize("second", [CLAUSE, CLAUSE.replace("900.00", "901.00"),
    GENERAL + "UNA VEZ EL VALOR ESTIMADO." + WINDOW, CLAUSE.upper()])
def test_uncertain_group_abstains_all_fields_never_selects_or_fills(second):
    result = predict_documents([source(HEADER + CLAUSE + "\n" + HEADER + second)])
    decision = result["selective_decisions"]
    assert not decision["predictions"] and not decision["emitted_unit_ids"]
    abstention, = decision["abstentions"]
    assert abstention["emitted_values"] is None and abstention["identity_resolved"] is False
    assert len(abstention["occurrence_ids"]) == 2 and len(abstention["context_ids"]) == 2
    assert len(result["raw_candidate_output"]["occurrence_projection"]["occurrences"]) == 2
    assert result["raw_candidate_output"]["occurrence_projection"]["identity_gate"]["passed"] is False


def test_context_ids_use_anchors_not_parsed_values():
    result = predict_documents([source(HEADER + CLAUSE + "\n" + HEADER + CLAUSE)])
    raw = copy.deepcopy(result["raw_candidate_output"])
    from tesis_recommender.document_experience_units_dev_v1 import project_units
    raw["document_outputs"][0]["result"]["records"][0]["fields"]["billed_amount"] = "123.00"
    raw["occurrence_projection"] = project_units(raw["document_outputs"])
    assert [c["context_id"] for c in decisions(raw)["source_contexts"]] == [
        c["context_id"] for c in result["selective_decisions"]["source_contexts"]]


def test_items_roles_and_branches_remain_separate():
    mype = ("En el caso de postores que declaren en el Anexo N° 1 tener la condición de micro y pequeña "
            "empresa, se acredita una experiencia de S/ 100.00." + WINDOW)
    document = source(HEADER + "ITEM 1\n" + CLAUSE + "\n" + mype + "\nITEM 2\n" + CLAUSE)
    predictions = predict_documents([document])["selective_decisions"]["predictions"]
    assert {(p["item"], p["branch"]) for p in predictions} == {(1, "general"), (1, "mype"), (2, "general")}


def test_relative_threshold_stays_null_not_inferred_from_other_group():
    result = predict_documents([source(HEADER + GENERAL + "UNA VEZ EL VALOR ESTIMADO." + WINDOW)])
    assert result["selective_decisions"]["predictions"][0]["fields"] == {
        "billed_amount": None, "currency": None, "lookback_years": 8}


def test_abstention_causes_false_negatives_without_removing_any_reference_row():
    result = compare_synthetic(source(HEADER + CLAUSE + "\n" + HEADER + CLAUSE))
    stream = result["metrics"]["qualification"]
    assert stream["evaluable_documents"] == stream["reference_conditions"] == 1
    for field in FIELDS:
        assert stream["fields"][field]["fn"] == 1 and stream["fields"][field]["tp"] == 0
        assert stream["fields"][field]["recall"] == stream["fields"][field]["f1"] == 0
        assert stream["fields"][field]["precision"] is None
    assert result["prediction_decisions"]["abstained_groups"] == 1
    assert result["prediction_decisions"]["candidate_groups_in_evaluable_scope"] == 1
    assert result["prediction_decisions"]["emitted_groups_in_evaluable_scope"] == 0
    assert all(n == 0 for n in result["prediction_decisions"]["emitted_known_fields_in_evaluable_scope"].values())
    assert result["reference_documents_removed_due_to_abstention"] == result["reference_conditions_removed_due_to_abstention"] == 0
    assert result["uncertain_identity_resolved"] is False


def test_unknown_reference_is_not_a_success_or_false_negative():
    result = compare_synthetic(source(HEADER + CLAUSE + "\n" + HEADER + CLAUSE),
                               {"billed_amount": None, "currency": "PEN", "lookback_years": 8})
    amount = result["metrics"]["qualification"]["fields"]["billed_amount"]
    assert amount["reference_unknown"] == 1 and amount["tp"] == amount["fp"] == amount["fn"] == 0
    assert amount["f1"] is None


def test_empty_candidate_does_not_prove_absence_and_keeps_known_misses():
    document = source("No supported predicate in synthetic input.")
    candidate = predict_documents([document])
    assert not candidate["selective_decisions"]["predictions"]
    assert candidate["raw_candidate_output"]["coverage"][0]["absence_of_requirements_proven"] is False
    assert compare_synthetic(document)["metrics"]["qualification"]["fields"]["billed_amount"]["fn"] == 1


@pytest.mark.parametrize("mode", ["drop_abstention", "fill_prediction", "drop_context", "edit_reason", "modify_raw_projection"])
def test_output_tampering_rejected(mode):
    document = source(HEADER + CLAUSE + "\n" + HEADER + CLAUSE)
    candidate = predict_documents([document])
    if mode == "drop_abstention":
        candidate["selective_decisions"]["abstentions"] = []
    elif mode == "fill_prediction":
        candidate["selective_decisions"]["predictions"] = [{"fields": FIELDS}]
    elif mode == "drop_context":
        candidate["selective_decisions"]["source_contexts"] = []
    elif mode == "edit_reason":
        candidate["selective_decisions"]["abstentions"][0]["reasons"] = []
    else:
        candidate["raw_candidate_output"]["occurrence_projection"]["identity_gate"]["passed"] = True
    with pytest.raises(ValueError):
        adapt(candidate, [document])


@pytest.mark.parametrize("text", [HEADER + CLAUSE, "No prediction"])
def test_duplicate_pdf_identity_is_hard_error_even_with_no_predictions(text):
    a, b = source(text), source(text, ocid="ocds-other")
    with pytest.raises(ValueError, match="PDF identity"):
        predict_documents([a, b])


def test_repeated_document_and_changed_source_rejected():
    document = source(HEADER + CLAUSE)
    with pytest.raises(ValueError):
        predict_documents([document, document])
    document["pages"][0]["text"] += "tampered"
    with pytest.raises(ValueError):
        predict_documents([document])


def test_artifact_binding_and_reference_population_are_not_only_flags():
    document = source(HEADER + CLAUSE)
    locked, ledger = gold(document)
    binding = {name: "b" * 64 for name in ("reference_manifest_sha256", "reference_sha256",
               "source_manifest_sha256", "evaluation_design_manifest_sha256")}
    with pytest.raises(ValueError, match="bound"):
        compare(locked, predict_documents([document]), [document], ledger, {}, binding)
    locked["source_population_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="populations"):
        compare(locked, predict_documents([document]), [document], ledger, binding, binding)
