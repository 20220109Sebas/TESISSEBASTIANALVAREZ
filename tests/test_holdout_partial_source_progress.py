from copy import deepcopy

import pytest

from scripts.record_holdout_full_source_pass import FLAGS
from scripts.record_holdout_partial_source_pass import apply_partial_pass, prepare_review


def fixture():
    text = "Experiencia del postor por cien mil soles"
    source = {"case": {"source": {"sample_number": 2, "sha256": "pdfhash", "file_path": "source.pdf"},
        "pages": [{"page": 1, "text": text, "native_text": "", "ocr_used": True},
                  {"page": 2, "text": "unread", "native_text": "unread", "ocr_used": False}]}}
    doc = {"opportunity_id": "case", "sample_number": 2, "pdf_sha256": "pdfhash", "pdf_path": "source.pdf",
        "total_pages": 2, "entire_procurement_file_reviewed": False, "historical_content_proven": False,
        "ranking_eligible": False, "decision": "candidate_partial",
        "coverage": {"native_pages_read": [], "ocr_pages_read": [1], "complete_rendered_pages_read": [1],
            "unreviewed_pages": [2], "full_pdf_semantically_read": False},
        "rendered_pages": [{"page": 1, "render_path": "page.png"}],
        "bidder_experience_conditions": [{"subject": "bidder", "evidence_ids": ["clause"],
            "document_reference_complete": False}],
        "evidence": [{"id": "clause", "page": 1, "quote": text, "render_path": "page.png"}],
        "visual_observations": [{"page": 1, "replaces_ocr_text": False}]}
    review = {"review_kind": "automatic_source_only_partial_reading_notes", "reviewer": "automatic",
        **{flag: False for flag in FLAGS}, "documents": [doc]}
    return review, source


def rows():
    return [{"opportunity_id": "case" if n == 0 else f"other{n}", "sample_number": 2 if n == 0 else n+100,
        "source_status": "downloaded_unique_pdf", "physical_pages": 2, "first_pass_status": "not_performed",
        "first_pass_decision": None, "second_pass_status": "not_performed", "reference_locked": False,
        "predictions_generated": False, "historical_ranking_eligible": False} for n in range(50)]


def test_offsets_and_coverage_do_not_create_a_full_pass():
    raw, source = fixture()
    result = prepare_review(raw, source)
    evidence = result["documents"][0]["evidence"][0]
    assert evidence["char_start"] == 0
    assert evidence["char_end"] == len(evidence["quote"])
    assert evidence["quote_basis"] == "ocr_text_verbatim_not_error_free"
    assert "char_start" not in raw["documents"][0]["evidence"][0]
    assert result["documents"][0]["coverage"]["unreviewed_pages"] == [2]


@pytest.mark.parametrize("flag", FLAGS)
def test_claims_of_human_or_final_review_rejected(flag):
    review, source = fixture()
    review[flag] = True
    with pytest.raises(ValueError):
        prepare_review(review, source)


@pytest.mark.parametrize("field,value", [("unreviewed_pages", []), ("full_pdf_semantically_read", True),
    ("ocr_pages_read", [True]), ("ocr_pages_read", [1,1]), ("ocr_pages_read", [3])])
def test_inconsistent_coverage_rejected(field, value):
    review, source = fixture()
    review["documents"][0]["coverage"][field] = value
    with pytest.raises(ValueError):
        prepare_review(review, source)


@pytest.mark.parametrize("field,value", [("pdf_sha256", "wrong"), ("pdf_path", "wrong.pdf"),
    ("sample_number", 1), ("total_pages", 1), ("ranking_eligible", True),
    ("historical_content_proven", True), ("entire_procurement_file_reviewed", True)])
def test_wrong_identity_and_overstated_scope_rejected(field, value):
    review, source = fixture()
    review["documents"][0][field] = value
    with pytest.raises(ValueError):
        prepare_review(review, source)


def test_invalid_literal_or_condition_link_rejected():
    review, source = fixture()
    review["documents"][0]["evidence"][0]["quote"] = "invented"
    with pytest.raises(ValueError):
        prepare_review(review, source)
    review, source = fixture()
    review["documents"][0]["bidder_experience_conditions"][0]["evidence_ids"] = ["missing"]
    with pytest.raises(ValueError):
        prepare_review(review, source)


def test_visual_ocr_replacement_not_permitted():
    review, source = fixture()
    review["documents"][0]["visual_observations"][0]["replaces_ocr_text"] = True
    with pytest.raises(ValueError):
        prepare_review(review, source)


def test_unavailable_ocr_and_missing_render_rejected():
    review, source = fixture()
    source["case"]["pages"][0]["ocr_used"] = False
    with pytest.raises(ValueError):
        prepare_review(review, source)
    review, source = fixture()
    review["documents"][0]["rendered_pages"] = []
    with pytest.raises(ValueError):
        prepare_review(review, source)


def test_existing_rows_and_judgments_preserved():
    review = prepare_review(*fixture())
    previous = rows()
    result = apply_partial_pass(previous, review)
    assert result[0]["first_pass_status"] == "partial"
    assert result[0]["first_pass_decision"] is None
    assert result[0]["second_pass_status"] == "not_performed"
    assert result[0]["reference_locked"] is False
    assert result[1:] == previous[1:]
    assert previous[0]["first_pass_status"] == "not_performed"


@pytest.mark.parametrize("field,value", [("first_pass_status", "full_supplied_pdf"),
    ("second_pass_status", "full_supplied_pdf"), ("reference_locked", True),
    ("predictions_generated", True), ("historical_ranking_eligible", True),
    ("source_status", "too_large_partial_file"), ("sample_number", 3), ("physical_pages", 99)])
def test_complete_or_wrong_progress_cannot_be_replaced(field, value):
    review = prepare_review(*fixture())
    previous = rows()
    previous[0][field] = value
    with pytest.raises(ValueError):
        apply_partial_pass(previous, review)


def test_fifty_cases_and_unique_documents_required():
    review, source = fixture()
    review["documents"].append(deepcopy(review["documents"][0]))
    with pytest.raises(ValueError):
        prepare_review(review, source)
    with pytest.raises(ValueError):
        apply_partial_pass(rows()[:-1], prepare_review(*fixture()))
