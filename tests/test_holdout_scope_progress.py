from copy import deepcopy
import json

import pytest

from scripts.extend_holdout_review_progress import validate_scope_review, update_second_pass_summary


def fixture():
    quote = "Perfil Académico - Profesional Requerido"
    source = {"case": {"source": {"sample_number": 14}, "pages": [{"page": 1, "text": quote}]}}
    first = {"case": {"sample_number": 14, "coverage": {"full_pdf_semantically_read": True},
        "domain": "consultor_individual", "decision": "outside_business_services_domain"}}
    review = {"review_kind": "automatic_source_only_scope_second_pass", "human_annotation": False,
        "expert_annotation": False, "predictions_consulted": False, "model_rules_consulted": False,
        "reference_locked": False, "documents": [{"opportunity_id": "case", "sample_number": 14,
            "pages_read": [1], "complete_rendered_pages_read": [1], "full_supplied_document_read": True,
            "entire_procurement_file_reviewed": False, "decision": "outside_business_services_domain",
            "bidder_experience_conditions": [], "evidence": [{"page": 1, "quote": quote}]}]}
    return review, source, first


def test_individual_professional_scope_agreement():
    validate_scope_review(*fixture())


@pytest.mark.parametrize("flag", ["human_annotation", "expert_annotation", "predictions_consulted", "model_rules_consulted", "reference_locked"])
def test_no_positive_or_missing_attestations(flag):
    review, source, first = fixture()
    review[flag] = None
    with pytest.raises(ValueError):
        validate_scope_review(review, source, first)


@pytest.mark.parametrize("field,value", [("pages_read", [True]), ("pages_read", []), ("pages_read", [1, 1]),
    ("complete_rendered_pages_read", []), ("entire_procurement_file_reviewed", True),
    ("full_supplied_document_read", 1), ("decision", "explicit_no_qualification_document_scope"),
    ("bidder_experience_conditions", [{"amount": 1}])])
def test_no_incomplete_readings_or_outside_scope_negatives(field, value):
    review, source, first = fixture()
    review["documents"][0][field] = value
    with pytest.raises(ValueError):
        validate_scope_review(review, source, first)


def test_natural_person_status_alone_is_insufficient():
    review, source, first = fixture()
    quote = "personas naturales"
    source["case"]["pages"][0]["text"] = quote
    review["documents"][0]["evidence"][0]["quote"] = quote
    with pytest.raises(ValueError):
        validate_scope_review(review, source, first)


def test_repeated_review_rejected():
    review, source, first = fixture()
    review["documents"].append(deepcopy(review["documents"][0]))
    with pytest.raises(ValueError):
        validate_scope_review(review, source, first)


def test_first_pass_partial_does_not_become_agreement():
    review, source, first = fixture()
    first["case"]["coverage"]["full_pdf_semantically_read"] = False
    with pytest.raises(ValueError):
        validate_scope_review(review, source, first)


def ui_fixture():
    review, source, first = fixture()
    quotes = ["método de Consultores Individuales", "comparación de Currículos Vitae"]
    source["case"]["pages"][0]["text"] = "\n".join(quotes)
    review["documents"][0]["evidence"] = [{"page":1,"quote":q,"image":"page.png"} for q in quotes]
    return review, source, first


def test_explicit_individual_method_and_cv_support_scope():
    validate_scope_review(*ui_fixture())


@pytest.mark.parametrize("missing", [0,1])
def test_ui_scope_needs_both_selection_and_cv_evidence(missing):
    review, source, first = ui_fixture()
    review["documents"][0]["evidence"].pop(missing)
    with pytest.raises(ValueError):
        validate_scope_review(review, source, first)


def test_complete_render_inventory_accepted():
    review, source, first = ui_fixture()
    review["documents"][0]["rendered_pages"] = [{"page":1,"render_path":"page.png"}]
    validate_scope_review(review, source, first)


@pytest.mark.parametrize("renders", [[], [{"page":True,"render_path":"page.png"}],
    [{"page":1,"render_path":"page.png"}]*2, [{"page":2,"render_path":"page.png"}],
    [{"page":1,"render_path":""}], [{"page":1,"render_path":"other.png"}]])
def test_second_pass_render_inventory_cannot_overstate_images(renders):
    review, source, first = ui_fixture()
    review["documents"][0]["rendered_pages"] = renders
    with pytest.raises(ValueError):
        validate_scope_review(review, source, first)


def test_second_summary_keeps_other_warnings_and_exposes_same_reviewer():
    prior = {"sample_number":99,"second_full_source_pass_pending":True,"source_warnings":["old"]}
    summary = {"completed_first_readings_in_this_block":1,"partial_readings_recorded_in_this_block":1,
        "unresolved_cross_review_issues":[deepcopy(prior),{"sample_number":14,"second_full_source_pass_pending":True}]}
    review, _, _ = fixture()
    review["independent_reviewer"] = False
    review["documents"][0]["source_warnings"] = [{"code":"unresolved_ocr"}]
    rows = [{"second_pass_status":"full_supplied_pdf","outside_business_scope_confirmed_two_pass":True}]
    update_second_pass_summary(summary, rows, review)
    assert summary["unresolved_cross_review_issues"][0] == prior
    completed = summary["unresolved_cross_review_issues"][1]
    assert completed["second_full_source_pass_pending"] is False
    assert completed["independent_reviewer"] is False
    assert completed["source_warnings"] == [{"code":"unresolved_ocr"}]
    assert summary["new_second_passes_in_this_block"] == 1
    assert summary["completed_first_readings_in_this_block"] == 0
    assert summary["partial_readings_recorded_in_this_block"] == 0


def test_second_build_preserves_inventories_prior_rows_and_evidence(tmp_path, monkeypatch):
    import scripts.extend_holdout_review_progress as module
    review, sources, first = ui_fixture()
    review.update(reviewer="same reviewer", independent_reviewer=False, independence_note="not independent")
    image = tmp_path / "page.png"
    image.write_bytes(b"fixture image")
    for e in review["documents"][0]["evidence"]:
        e["image"] = str(image)
    review["documents"][0]["rendered_pages"] = [{"page":1,"render_path":str(image)}]
    sources["case"]["source"]["sha256"] = "fixture pdf hash"
    sources["case"]["pages"][0]["extraction_method"] = "native"
    rows = [{"opportunity_id":"case" if i==0 else f"other-{i}",
        "sample_number":14 if i==0 else i+100, "second_pass_status":"not_performed"} for i in range(50)]
    base, source_dir = tmp_path / "base", tmp_path / "source"
    base.mkdir()
    source_dir.mkdir()
    for directory in (base, source_dir):
        (directory / "manifest.json").write_text(json.dumps({"input_sha256":{}}))
    (source_dir / "source_documents.json").write_text(json.dumps([
        {"opportunity_id":key, **value} for key,value in sources.items()]))
    (base / "review_progress.json").write_text(json.dumps(rows))
    old_evidence = [{"quote":"earlier second reading","opportunity_id":"other-1"}]
    (base / "second_pass_evidence.json").write_text(json.dumps(old_evidence))
    (base / "summary.json").write_text(json.dumps({"reference_locked":False,"f1_estimable":False}))
    inventories = {"compiled_source_reviews.json":b'{"earlier":"full inventory"}\n',
        "partial_source_reviews.json":b'{"earlier":"partial inventory"}\n'}
    for filename,content in inventories.items():
        (base / filename).write_bytes(content)
    first_path, review_path = tmp_path / "first.json", tmp_path / "second.json"
    first_path.write_text(json.dumps({"documents":[{"opportunity_id":key,**value} for key,value in first.items()]}))
    review_path.write_text(json.dumps(review))
    monkeypatch.setattr(module,"protected_outputs",lambda:{})
    monkeypatch.setattr(module,"verify_artifact",lambda directory,inputs:None)
    out = tmp_path / "out"
    summary = module.build(base,source_dir,first_path,review_path,out)
    current = json.loads((out / "review_progress.json").read_text())
    assert current[1:] == rows[1:]
    assert current[0]["second_pass_independent_reviewer"] is False
    assert json.loads((out / "second_pass_evidence.json").read_text())[0] == old_evidence[0]
    for filename,content in inventories.items():
        assert (out / filename).read_bytes() == content
    manifest = json.loads((out / "manifest.json").read_text())
    assert str(image) in manifest["input_sha256"]
    assert set(inventories) <= set(manifest["output_sha256"])
    assert summary["reference_locked"] is False
    assert summary["f1_estimable"] is False
    with pytest.raises(ValueError,match="new progress directory"):
        module.build(base,source_dir,first_path,review_path,out)
