from copy import deepcopy

import pytest

from scripts.assemble_holdout_review_progress import validate_second_pass


def fixture():
    ocid = "test-case"
    quote = "REQUISITOS DE CALIFICACIÓN\nNo corresponde"
    source = {ocid: {"source": {"sample_number": 45}, "pages": [{"page": 1, "text": quote}]}}
    first = {ocid: {"sample_number": 45, "coverage": {"full_pdf_semantically_read": True},
        "decision": "explicit_no_qualification_document_scope", "bidder_experience_conditions": []}}
    review = {"review_kind": "automatic_source_only_second_pass", "human_annotation": False,
        "expert_annotation": False, "predictions_consulted": False, "model_rules_consulted": False,
        "reference_locked": False, "documents": [{"opportunity_id": ocid, "sample_number": 45,
        "pages_read": [1], "complete_rendered_pages_read": [1], "full_supplied_document_read": True,
        "entire_procurement_file_reviewed": False, "decision": "explicit_no_qualification_document_scope",
        "bidder_experience_conditions": [], "evidence": [{"page": 1, "quote": quote}]}]}
    return review, source, first


def test_scoped_source_agreement_is_not_a_locked_reference():
    review, source, first = fixture()
    result = validate_second_pass(review, source, first)
    assert result["test-case"]["entire_procurement_file_reviewed"] is False
    assert review["reference_locked"] is False


@pytest.mark.parametrize("field", ["human_annotation", "expert_annotation", "predictions_consulted",
    "model_rules_consulted", "reference_locked"])
@pytest.mark.parametrize("value", [None, 0, True])
def test_second_pass_flags_require_literal_false(field, value):
    review, source, first = fixture()
    review[field] = value
    with pytest.raises(ValueError):
        validate_second_pass(review, source, first)


@pytest.mark.parametrize("field,value", [("pages_read", []), ("pages_read", [1, 1]),
    ("pages_read", [True]), ("pages_read", [2]), ("complete_rendered_pages_read", []),
    ("full_supplied_document_read", 1), ("entire_procurement_file_reviewed", True),
    ("evidence", []), ("bidder_experience_conditions", [{"amount": 10}]),
    ("decision", "abstain_unsuitable_source"), ("sample_number", 31)])
def test_invalid_coverage_or_unresolved_decision_rejected(field, value):
    review, source, first = fixture()
    review["documents"][0][field] = value
    with pytest.raises(ValueError):
        validate_second_pass(review, source, first)


def test_partial_first_pass_cannot_be_promoted_by_second_pass():
    review, source, first = fixture()
    first["test-case"]["coverage"]["full_pdf_semantically_read"] = False
    with pytest.raises(ValueError):
        validate_second_pass(review, source, first)


def test_duplicate_documents_stop_consolidation():
    review, source, first = fixture()
    review["documents"].append(deepcopy(review["documents"][0]))
    with pytest.raises(ValueError):
        validate_second_pass(review, source, first)


def test_empty_conditions_need_explicit_absence_not_missing_data():
    review, source, first = fixture()
    text = "No se encontró el documento"
    source["test-case"]["pages"][0]["text"] = text
    review["documents"][0]["evidence"][0]["quote"] = text
    with pytest.raises(ValueError):
        validate_second_pass(review, source, first)


def test_unsuitable_source_agreement_is_not_an_absence_judgment():
    review, source, first = fixture()
    first["test-case"]["decision"] = "abstain_unsuitable_source"
    review["documents"][0]["decision"] = "abstain_unsuitable_source"
    assert validate_second_pass(review, source, first)["test-case"]["decision"] == "abstain_unsuitable_source"
