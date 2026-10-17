from copy import deepcopy
import hashlib

import pytest

from scripts.record_holdout_full_source_pass import FLAGS, apply_first_pass, merge_source_reviews, prepare_review, register_render_inventory, current_block_counts, preserved_auxiliary_inventories


def fixture():
    text = "EXPERIENCIA DEL POSTOR: cien mil soles"
    source = {"case": {"source": {"sample_number": 8, "sha256": "pdfhash", "file_path": "source.pdf"},
        "pages": [{"page": 1, "text": text, "native_text": text, "ocr_used": False}]}}
    doc = {"opportunity_id": "case", "sample_number": 8, "pdf_sha256": "pdfhash", "pdf_path": "source.pdf",
        "total_pages": 1, "entire_procurement_file_reviewed": False, "decision": "candidate_boundary_pending",
        "coverage": {"native_pages_read": [1], "ocr_pages_read": [], "complete_rendered_pages_read": [1],
            "unreviewed_pages": [], "full_pdf_semantically_read": True},
        "bidder_experience_conditions": [{"billed_amount": "100000", "lookback_years": None}],
        "evidence": [{"id": "clause", "page": 1, "quote": text, "render_path": "page.png"}]}
    review = {"review_kind": "automatic_source_only_full_reading_notes", "reviewer": "automatic",
        **{flag: False for flag in FLAGS}, "documents": [doc]}
    return review, source


def rows():
    return [{"opportunity_id": "case" if n == 0 else f"other{n}", "sample_number": 8 if n == 0 else n + 100,
        "source_status": "downloaded_unique_pdf", "physical_pages": 1, "first_pass_status": "partial",
        "first_pass_decision": "prior", "second_pass_status": "not_performed", "two_pass_agreement": None,
        "reference_locked": False, "predictions_generated": False, "historical_ranking_eligible": False}
        for n in range(50)]


def test_offsets_are_source_derived_without_closing_reference():
    raw, source = fixture()
    result = prepare_review(raw, source)
    e = result["documents"][0]["evidence"][0]
    assert e["char_start"] == 0
    assert e["char_end"] == len(e["quote"])
    assert e["text_sha256"] == hashlib.sha256(e["quote"].encode()).hexdigest()
    assert "char_start" not in raw["documents"][0]["evidence"][0]
    assert result["reference_locked"] is False
    assert result["documents"][0]["bidder_experience_conditions"][0]["lookback_years"] is None


def test_full_pass_retains_partial_and_second_pass_notes(tmp_path):
    import json
    partial = {"documents": [{"opportunity_id": "case", "coverage": [1]}]}
    second = {"independent_reviewer": False, "documents": [{"opportunity_id": "old"}]}
    for name, value in (("partial_source_reviews.json", partial), ("second_source_review.json", second)):
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "manifest.json").write_text('{"not":"an inventory"}')
    result = preserved_auxiliary_inventories(tmp_path)
    assert result == {"partial_source_reviews.json": partial, "second_source_review.json": second}
    assert "manifest.json" not in result


def test_legacy_full_pass_without_auxiliary_files_remains_supported(tmp_path):
    assert preserved_auxiliary_inventories(tmp_path) == {}


@pytest.mark.parametrize("flag", FLAGS)
def test_first_pass_rejects_claimed_or_missing_finality(flag):
    value, source = fixture()
    value[flag] = None
    with pytest.raises(ValueError):
        prepare_review(value, source)


@pytest.mark.parametrize("key,value", [("native_pages_read", [True]), ("native_pages_read", [1, 1]),
    ("native_pages_read", [2]), ("unreviewed_pages", [1]), ("full_pdf_semantically_read", 1),
    ("complete_rendered_pages_read", [])])
def test_no_invalid_or_inconsistent_coverage(key, value):
    review, source = fixture()
    review["documents"][0]["coverage"][key] = value
    with pytest.raises(ValueError):
        prepare_review(review, source)


def test_unread_page_cannot_be_promoted():
    review, source = fixture()
    review["documents"][0]["total_pages"] = 2
    source["case"]["pages"].append({"page": 2, "text": "unread", "native_text": "unread", "ocr_used": False})
    with pytest.raises(ValueError):
        prepare_review(review, source)


@pytest.mark.parametrize("field,value", [("pdf_sha256", "wrong"), ("pdf_path", "other.pdf"),
    ("sample_number", 9), ("total_pages", 2), ("entire_procurement_file_reviewed", True)])
def test_identity_and_supplied_document_scope(field, value):
    review, source = fixture()
    review["documents"][0][field] = value
    with pytest.raises(ValueError):
        prepare_review(review, source)


def test_quotation_must_be_literal_and_unique():
    review, source = fixture()
    review["documents"][0]["evidence"][0]["quote"] = "invented quote"
    with pytest.raises(ValueError):
        prepare_review(review, source)


def test_empty_native_page_requires_a_visual_reading():
    review, source = fixture()
    source["case"]["pages"].append({"page": 2, "native_text": "", "text": "", "ocr_used": False})
    doc = review["documents"][0]
    doc["total_pages"] = 2
    doc["coverage"]["native_pages_read"].append(2)
    with pytest.raises(ValueError):
        prepare_review(review, source)


def test_unavailable_ocr_rejected():
    review, source = fixture()
    review["documents"][0]["coverage"]["ocr_pages_read"] = [1]
    with pytest.raises(ValueError):
        prepare_review(review, source)


def test_second_pass_not_created_and_old_decisions_preserved():
    review = prepare_review(*fixture())
    previous = rows()
    result = apply_first_pass(previous, [review])
    assert previous[0]["first_pass_status"] == "partial"
    assert result[0]["first_pass_status"] == "full_supplied_pdf"
    assert result[0]["previous_first_pass_decision"] == "prior"
    assert result[0]["second_pass_status"] == "not_performed"
    assert result[0]["two_pass_agreement"] is None
    assert result[0]["reference_locked"] is False
    assert result[1:] == previous[1:]


@pytest.mark.parametrize("field,value", [("second_pass_status", "full_supplied_pdf"), ("reference_locked", True),
    ("predictions_generated", True), ("historical_ranking_eligible", True), ("source_status", "too_large_partial_file"),
    ("physical_pages", 99), ("sample_number", 20)])
def test_do_not_replace_existing_decisions_or_use_wrong_source(field, value):
    review = prepare_review(*fixture())
    previous = rows()
    previous[0][field] = value
    with pytest.raises(ValueError):
        apply_first_pass(previous, [review])


def test_duplicate_reviews_are_not_independent_passes():
    review = prepare_review(*fixture())
    with pytest.raises(ValueError):
        apply_first_pass(rows(), [review, deepcopy(review)])


def test_all_fifty_cases_remain_in_denominator():
    review = prepare_review(*fixture())
    with pytest.raises(ValueError):
        apply_first_pass(rows()[:-1], [review])


def test_previous_evidence_and_reviewers_are_preserved():
    review = prepare_review(*fixture())
    previous = [{"opportunity_id": "old", "sample_number": 3, "reviewer": "Earlier reviewer",
        "evidence": [{"quote": "earlier source"}]}]
    result = merge_source_reviews(previous, [review])
    assert result[0] == previous[0]
    assert result[1] == review["documents"][0]
    result[0]["evidence"][0]["quote"] = "changed copy"
    assert previous[0]["evidence"][0]["quote"] == "earlier source"


def test_extended_first_pass_replaces_inventory_not_second_pass():
    review = prepare_review(*fixture())
    previous = [{"opportunity_id": "case", "sample_number": 8, "reviewer": "Partial reviewer"}]
    result = merge_source_reviews(previous, [review])
    assert len(result) == 1
    assert result[0]["reviewer"] == "automatic"


def test_inventory_rejects_duplicate_prior_documents():
    with pytest.raises(ValueError):
        merge_source_reviews([{"opportunity_id": "case"}, {"opportunity_id": "case"}], [])


def test_inventory_rejects_duplicate_new_readings():
    review = prepare_review(*fixture())
    with pytest.raises(ValueError):
        merge_source_reviews([], [review, deepcopy(review)])


@pytest.mark.parametrize("renders", [[], [{"page":1,"render_path":"page.png"}]*2,
    [{"page":2,"render_path":"page.png"}], [{"page":True,"render_path":"page.png"}],
    [{"page":1,"render_path":""}], [{"page":1,"render_path":"different.png"}]])
def test_explicit_render_inventory_matches_all_attested_pages(renders):
    review, source = fixture()
    review["documents"][0]["rendered_pages"] = renders
    with pytest.raises(ValueError):
        prepare_review(review, source)


@pytest.mark.parametrize("page,replaces", [(2,False),(1,True),(1,None)])
def test_explicit_inventory_checks_visual_notes_without_rewriting_ocr(page,replaces):
    review, source = fixture()
    doc = review["documents"][0]
    doc["rendered_pages"] = [{"page":1,"render_path":"page.png"}]
    doc["visual_observations"] = [{"page":page,"replaces_ocr_text":replaces}]
    with pytest.raises(ValueError):
        prepare_review(review, source)


def test_all_explicit_renders_are_hashed_even_without_quotes(tmp_path,monkeypatch):
    monkeypatch.setattr("scripts.record_holdout_full_source_pass.ROOT",tmp_path)
    page = tmp_path / "page.png"
    page.write_bytes(b"test image fixture")
    doc = {"rendered_pages":[{"page":2,"render_path":"page.png"}],"evidence":[]}
    inputs = {}
    register_render_inventory(doc,inputs)
    expected = hashlib.sha256(page.read_bytes()).hexdigest()
    assert inputs == {str(page):expected}
    assert doc["rendered_pages"][0]["render_sha256"] == expected


def test_explicit_render_hash_mismatch_is_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr("scripts.record_holdout_full_source_pass.ROOT",tmp_path)
    (tmp_path / "page.png").write_bytes(b"test image fixture")
    doc = {"rendered_pages":[{"page":1,"render_path":"page.png","render_sha256":"wrong"}]}
    with pytest.raises(ValueError,match="Rendered image changed"):
        register_render_inventory(doc,{})


def test_full_block_resets_inherited_partial_activity_without_second_pass():
    review,_ = fixture()
    summary = {"partial_readings_recorded_in_this_block":5,"new_second_passes_in_this_block":3}
    summary.update(current_block_counts([review]))
    assert summary == {"completed_first_readings_in_this_block":1,
        "partial_readings_recorded_in_this_block":0,"new_second_passes_in_this_block":0}
