import copy
import json
from pathlib import Path

import pytest

from tesis_recommender.document_experience_dev_v1 import digest
from tesis_recommender.document_experience_candidate_v1 import (
    MAX_BYTES, MAX_DOCUMENTS, MAX_PAGES, predict_documents, validate_document,
)

ROOT = Path(__file__).resolve().parents[1]
HEADER = "REQUISITOS DE CALIFICACIÓN\nEXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD\n"
CLAUSE = ("El postor debe acreditar un monto facturado acumulado equivalente a S/ 900.00 "
          "durante los ocho (8) años anteriores a la fecha de presentación de ofertas.")


def source(*texts, ocid="ocds-candidate-test"):
    pages = [{"page": n, "text": text, "text_sha256": digest(text), "native_text": text,
              "native_text_sha256": digest(text), "ocr_selected": False, "ocr_used": False,
              "selected_ocr_unresolved": False, "extraction_method": "native_pdf_text"}
             for n, text in enumerate(texts or (HEADER + CLAUSE,), 1)]
    return {"opportunity_id": ocid, "source": {"opportunity_id": ocid, "status": "downloaded_pdf",
        "sha256": "a" * 64, "bytes": 1234}, "pages": pages, "total_pages": len(pages),
        "processed_pages": len(pages), "truncated_by_page_limit": False,
        "ranking_eligible": False, "historical_content_proven": False}


def test_native_input_combines_extraction_provenance_and_unit_identity():
    document = source()
    original = copy.deepcopy(document)
    result = predict_documents([document])
    unit, = result["occurrence_projection"]["candidate_units"]
    assert unit["fields"] == {"billed_amount": "900.00", "currency": "PEN", "lookback_years": 8}
    assert result["occurrence_projection"]["identity_gate"]["passed"]
    assert document == original
    for flag in ("metrics_computed", "reference_or_expected_values_used", "pdf_downloads_performed", "new_ocr_performed", "ranking_eligible"):
        assert result[flag] is False
    assert not result["number_words_semantic_consistency_proven"]


def test_ocr_source_requires_full_lineage_and_preserves_native_view():
    document = source("Scanned native header only")
    page = document["pages"][0]
    text = HEADER + CLAUSE
    page.update(text=text, text_sha256=digest(text), ocr_text=text, ocr_text_sha256=digest(text),
                ocr_selected=True, ocr_used=True, extraction_method="apple_vision_local_ocr",
                ocr_json_path="synthetic-test-only.json", ocr_json_sha256="b" * 64)
    result = predict_documents([document])
    assert len(result["occurrence_projection"]["occurrences"]) == 1
    span = result["document_outputs"][0]["result"]["records"][0]["clause_evidence"]["source_spans"][0]
    assert span["ocr_json_sha256"] == "b" * 64
    assert document["pages"][0]["native_text"] == "Scanned native header only"


def test_unresolved_and_empty_ocr_is_coverage_gap_not_absence_or_negative():
    document = source("")
    document["pages"][0].update(ocr_selected=True, selected_ocr_unresolved=True,
                              extraction_method="native_pdf_text_incomplete_ocr_unresolved")
    result = predict_documents([document])
    assert result["coverage"][0]["unresolved_ocr_pages"] == [1]
    assert result["coverage"][0]["empty_text_pages"] == [1]
    assert not result["coverage"][0]["absence_of_requirements_proven"]
    assert not result["occurrence_projection"]["candidate_units"]


@pytest.mark.parametrize("key,value", [
    ("total_pages", 2), ("processed_pages", 2), ("truncated_by_page_limit", True),
    ("ranking_eligible", True), ("historical_content_proven", True), ("total_pages", True),
])
def test_partial_documents_and_eligibility_relabeling_rejected(key, value):
    document = source()
    document[key] = value
    with pytest.raises(ValueError):
        predict_documents([document])


@pytest.mark.parametrize("bytes_", [0, -1, MAX_BYTES + 1, True, "1234"])
def test_invalid_pdf_size_rejected(bytes_):
    document = source()
    document["source"]["bytes"] = bytes_
    with pytest.raises(ValueError):
        validate_document(document)


@pytest.mark.parametrize("key,value", [
    ("sha256", "bad"), ("status", "partial_pdf"), ("opportunity_id", "ocds-other-source"),
])
def test_source_identity_rejected(key, value):
    document = source()
    document["source"][key] = value
    with pytest.raises(ValueError):
        validate_document(document)


@pytest.mark.parametrize("key,value", [
    ("text", "Changed archived source"), ("native_text", "Changed native source"),
    ("extraction_method", "unknown_ocr"), ("ocr_used", "false"), ("ocr_selected", True),
    ("selected_ocr_unresolved", True), ("page", True),
])
def test_page_text_or_method_flags_rejected(key, value):
    document = source()
    document["pages"][0][key] = value
    with pytest.raises(ValueError):
        validate_document(document)


@pytest.mark.parametrize("key,value", [
    ("ocr_json_path", ""), ("ocr_json_sha256", "bad"), ("ocr_text", "different"),
    ("ocr_text_sha256", "c" * 64), ("selected_ocr_unresolved", True), ("ocr_used", False),
])
def test_bad_ocr_lineage_rejected(key, value):
    document = source()
    page = document["pages"][0]
    page.update(ocr_selected=True, ocr_used=True, extraction_method="apple_vision_local_ocr",
                ocr_json_path="test-only.json", ocr_json_sha256="b" * 64,
                ocr_text=page["text"], ocr_text_sha256=page["text_sha256"])
    page[key] = value
    with pytest.raises(ValueError):
        validate_document(document)


@pytest.mark.parametrize("change", ["reorder", "duplicate", "gap"])
def test_physical_page_sequence_rejected(change):
    document = source("one", "two")
    if change == "reorder":
        document["pages"].reverse()
    elif change == "duplicate":
        document["pages"][1]["page"] = 1
    else:
        document["pages"][1]["page"] = 3
    with pytest.raises(ValueError):
        validate_document(document)


def test_fixed_document_page_and_batch_limits():
    assert validate_document(source(*(["page"] * MAX_PAGES)))
    with pytest.raises(ValueError):
        validate_document(source(*(["page"] * (MAX_PAGES + 1))))
    with pytest.raises(ValueError):
        predict_documents([])
    with pytest.raises(ValueError):
        predict_documents([source(ocid=f"ocds-batch-{n}") for n in range(MAX_DOCUMENTS + 1)])
    with pytest.raises(ValueError):
        predict_documents([source(), source()])


def test_whole_number_words_conflict_is_not_claimed_semantically_validated():
    document = source(HEADER + CLAUSE.replace("S/ 900.00", "S/ 900.00 (mil soles)"))
    result = predict_documents([document])
    assert not result["number_words_semantic_consistency_proven"]
    assert not result["metrics_computed"]


def test_identity_conflict_stops_projection_without_erasing_occurrences():
    result = predict_documents([source(HEADER + CLAUSE + "\n" + HEADER + CLAUSE)])
    projection = result["occurrence_projection"]
    assert len(projection["occurrences"]) == 2
    assert not projection["identity_gate"]["passed"]
    assert projection["candidate_units"][0]["fields"] is None


def test_all_observed_pages_pass_contract_with_original_outputs_unchanged():
    documents = json.loads((ROOT / "artifacts/experience_holdout_source_view_v1/source_documents.json").read_text())
    prior = json.loads((ROOT / "artifacts/document_experience_development_v2/document_candidates.json").read_text())
    result = predict_documents(documents)
    assert len(result["document_outputs"]) == 27
    by_id = {d["opportunity_id"]: d["result"] for d in prior}
    assert all(d["result"] == by_id[d["opportunity_id"]] for d in result["document_outputs"])
    assert len(result["occurrence_projection"]["occurrences"]) == 27
    assert not result["occurrence_projection"]["identity_gate"]["passed"]


@pytest.fixture(scope="module")
def candidate_lock(tmp_path_factory):
    from scripts.freeze_document_experience_candidate_v1 import freeze
    output = tmp_path_factory.mktemp("candidate-model-lock") / "lock"
    return output, freeze(output)


def test_frozen_model_and_declared_open_evaluation_gates(candidate_lock):
    from scripts.freeze_document_experience_candidate_v1 import validate_lock
    output, summary = candidate_lock
    lock = validate_lock(output)
    assert summary["model_code_files_frozen"] == 8 and summary["model_test_files_frozen"] == 4
    assert lock["model_interface_locked"]
    for flag in ("sampling_design_locked", "source_acquisition_and_ocr_locked", "new_reference_locked", "new_sample_selected", "ready_for_new_evaluation"):
        assert lock[flag] is False


def test_candidate_snapshot_bytes_reproduce_and_overwrite_is_rejected(candidate_lock, tmp_path):
    from scripts.freeze_document_experience_candidate_v1 import freeze
    original, _ = candidate_lock
    output = tmp_path / "replay"
    freeze(output)
    assert sorted(str(p.relative_to(original)) for p in original.rglob("*") if p.is_file()) == sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file())
    for path in original.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (output / path.relative_to(original)).read_bytes()
    with pytest.raises(ValueError):
        freeze(output)


def test_lock_runtime_mismatch_is_rejected_without_editing_real_files(candidate_lock, monkeypatch):
    import scripts.freeze_document_experience_candidate_v1 as module
    output, _ = candidate_lock
    original = module.runtime_identity()
    monkeypatch.setattr(module, "runtime_identity", lambda: dict(original, python_version="different"))
    with pytest.raises(ValueError, match="Runtime changed"):
        module.validate_lock(output)


def test_nonfixed_contract_rejected():
    from scripts.freeze_document_experience_candidate_v1 import CONTRACT, check_contract
    contract = json.loads(CONTRACT.read_text())
    contract["extraction"]["max_clause_characters"] = 100000
    with pytest.raises(ValueError):
        check_contract(contract)


def test_import_origin_change_is_rejected(candidate_lock, monkeypatch):
    from scripts.freeze_document_experience_candidate_v1 import validate_lock
    import tesis_recommender.document_experience_candidate_v1 as candidate
    output, _ = candidate_lock
    monkeypatch.setattr(candidate, "__file__", "/tmp/not-the-fixed-module.py")
    with pytest.raises(ValueError, match="different path"):
        validate_lock(output)


def test_verify_regression_does_not_mislabel_known_sources_as_unseen(candidate_lock, tmp_path):
    from scripts.verify_document_experience_candidate_v1 import verify
    lock_dir, _ = candidate_lock
    output = tmp_path / "regression"
    summary = verify(lock_dir, output)
    assert summary["documents"] == 27 and summary["original_occurrences"] == 27
    assert summary["all_documents_previously_inspected"]
    assert not summary["new_sample_inference_performed"] and not summary["metrics_computed"]
    replay = tmp_path / "replay"
    verify(lock_dir, replay)
    for path in output.iterdir():
        assert path.read_bytes() == (replay / path.name).read_bytes()
