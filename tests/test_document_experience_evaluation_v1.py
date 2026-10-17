"""Synthetic workflow tests, not a newly observed documentary evaluation."""
import copy
import json
from pathlib import Path

import pytest

from tesis_recommender.document_experience_candidate_v1 import predict_documents
from tesis_recommender.document_experience_dev_v1 import digest
from tesis_recommender.document_experience_evaluation_v1 import (
    FIELDS, adapt_predictions, build_reference, compare, source_population,
)

ROOT = Path(__file__).resolve().parents[1]
HEADER = "REQUISITOS DE CALIFICACIÓN\nEXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD\n"
CLAUSE = ("El postor debe acreditar un monto facturado acumulado equivalente a S/ 900.00 "
          "durante los ocho (8) años anteriores a la fecha de presentación de ofertas.")


def source(text=HEADER + CLAUSE, ocid="ocds-evaluation-synthetic"):
    page = {"page": 1, "text": text, "native_text": text, "text_sha256": digest(text),
            "native_text_sha256": digest(text), "ocr_selected": False, "ocr_used": False,
            "selected_ocr_unresolved": False, "extraction_method": "native_pdf_text"}
    return {"opportunity_id": ocid, "source": {"opportunity_id": ocid, "status": "downloaded_pdf",
        "sha256": digest(ocid), "bytes": 1234}, "pages": [page], "total_pages": 1, "processed_pages": 1,
        "truncated_by_page_limit": False, "ranking_eligible": False, "historical_content_proven": False}


def condition(document, role="qualification"):
    page = document["pages"][0]
    ref = {"page": 1, "char_start": 0, "char_end": len(page["text"]), "quote": page["text"],
           "page_text_sha256": page["text_sha256"], "pdf_sha256": document["source"]["sha256"]}
    return {"opportunity_id": document["opportunity_id"], "subject": "bidder", "section_role": role,
            "item": None, "branch": "general", "identity_reason": "Synthetic explicit source heading",
            "fields": {"billed_amount": "900.00", "currency": "PEN", "lookback_years": 8},
            "unknown_reasons": {f: None for f in FIELDS}, "identity_evidence": [ref],
            "field_evidence": {f: [copy.deepcopy(ref)] for f in FIELDS}}


def packet(document=None):
    document = source() if document is None else document
    ledger = [{"opportunity_id": document["opportunity_id"], "source_status": "complete_native_view"}]
    ledger += [{"opportunity_id": f"ocds-unavailable-synthetic-{n}", "source_status": "source_access_not_attempted"} for n in range(49)]
    cond = condition(document)
    readings = [{"pass_number": n, "reviewer": "automatic-synthetic-test-only", "scope": "enterprise_services",
                 "full_supplied_document_read": True, "predictions_consulted": False, "model_rules_consulted": False,
                 "page_reviews": [{"page": 1, "page_text_sha256": document["pages"][0]["text_sha256"], "content_status": "readable"}],
                 "conditions": [copy.deepcopy(cond)]} for n in (1, 2)]
    decisions = [{"opportunity_id": document["opportunity_id"], "scope": "enterprise_services",
                  "reason": "Synthetic fixture; no new PDF viewed", "source_pdf_sha256": document["source"]["sha256"],
                  "conditions": [cond], "readings": readings}]
    decisions += [{"opportunity_id": r["opportunity_id"], "scope": "inaccessible_source",
                   "reason": "No source, not a negative", "source_pdf_sha256": None, "conditions": [], "readings": []} for r in ledger[1:]]
    draft = {"reference_type": "automatic_source_review", "locked": False, "predictions_seen_before_lock": False,
             "is_human_or_expert": False, "independent_reviewer": False, "independence_note": "Automatic test, not independent/expert", "documents": decisions}
    return [document], ledger, draft


def binding():
    return {k: digest(k) for k in ("reference_manifest_sha256", "reference_sha256", "source_manifest_sha256", "evaluation_design_manifest_sha256")}


def test_adapter_preserves_exact_identity_fields_and_does_not_mutate():
    documents, _, _ = packet()
    candidate = predict_documents(documents)
    before = copy.deepcopy(candidate)
    adapted = adapt_predictions(candidate, documents)
    assert adapted["predictions"] == [{"opportunity_id": documents[0]["opportunity_id"], "subject": "bidder",
        "section_role": "qualification", "item": None, "branch": "general",
        "fields": {"billed_amount": "900.00", "currency": "PEN", "lookback_years": 8}}]
    assert candidate == before and not adapted["metrics_computed"]


def test_conflicts_stop_whole_comparison_even_for_outside_scope():
    documents, ledger, draft = packet(source(HEADER + CLAUSE + "\n" + HEADER + CLAUSE))
    draft["documents"][0]["scope"] = "outside_consultoria_de_obra"
    for reading in draft["documents"][0]["readings"]:
        reading["scope"] = "outside_consultoria_de_obra"
        reading["scope_evidence"] = copy.deepcopy(reading["conditions"][0]["identity_evidence"])
    locked = build_reference(draft, documents, ledger)
    candidate = predict_documents(documents)
    result = compare(locked, candidate, documents, ledger, binding(), binding())
    assert result["metrics"] is None and not result["metrics_computed"]
    assert result["adapter"]["original_occurrences"] == 2
    assert result["adapter"]["blocked_unit_ids"]


@pytest.mark.parametrize("change", ["gate", "drop_unit", "drop_occurrence", "amount", "source_pdf", "source_population", "quote", "coverage"])
def test_adapter_rejects_changed_projection_or_lineage(change):
    documents, _, _ = packet()
    candidate = predict_documents(documents)
    p = candidate["occurrence_projection"]
    if change == "gate":
        p["identity_gate"]["passed"] = False
    elif change == "drop_unit":
        p["candidate_units"] = []
    elif change == "drop_occurrence":
        p["occurrences"] = []
    elif change == "amount":
        p["candidate_units"][0]["fields"]["billed_amount"] = "901.00"
    elif change == "source_pdf":
        candidate["document_outputs"][0]["source_pdf_sha256"] = "b" * 64
    elif change == "source_population":
        candidate["document_outputs"] = []
    elif change == "coverage":
        candidate["coverage"][0]["unresolved_ocr_pages"] = [1]
    else:
        candidate["document_outputs"][0]["result"]["records"][0]["clause_evidence"]["evidence"] = "Changed"
    with pytest.raises(ValueError):
        adapt_predictions(candidate, documents)


def test_reference_preserves_fifty_cases_and_is_not_expert_truth():
    documents, ledger, draft = packet()
    before = copy.deepcopy(draft)
    locked = build_reference(draft, documents, ledger)
    assert len(locked["reference"]["documents"]) == 50
    assert sum(d["scope"] == "enterprise_services" for d in locked["reference"]["documents"]) == 1
    assert not locked["reference"]["is_human_or_expert"] and not locked["reviewer_independence_proven"]
    assert before == draft


@pytest.mark.parametrize("change", ["drop_case", "duplicate_case", "missing_complete_source", "extra_source", "duplicate_pdf"])
def test_population_gates(change):
    documents, ledger, _ = packet()
    if change == "drop_case":
        ledger.pop()
    elif change == "duplicate_case":
        ledger[1] = ledger[0]
    elif change == "missing_complete_source":
        documents = []
    elif change == "extra_source":
        documents.append(source(ocid="ocds-extra"))
    else:
        extra = source(ocid=ledger[1]["opportunity_id"])
        extra["source"]["sha256"] = documents[0]["source"]["sha256"]
        documents.append(extra)
        ledger[1]["source_status"] = "complete_native_view"
    with pytest.raises(ValueError):
        source_population(documents, ledger)


@pytest.mark.parametrize("field,value", [("locked", True), ("predictions_seen_before_lock", True),
    ("is_human_or_expert", True), ("independent_reviewer", True), ("reference_type", "gold"), ("independence_note", "")])
def test_no_post_inference_relabel_or_expert_claim(field, value):
    documents, ledger, draft = packet()
    draft[field] = value
    with pytest.raises(ValueError):
        build_reference(draft, documents, ledger)


@pytest.mark.parametrize("change", ["one_pass", "wrong_pass", "duplicate_page", "different_page_hash", "predictions", "model_rules", "different_pdf", "missing_identity", "wrong_quote", "duplicate_condition", "unreadable"])
def test_reference_source_reading_and_evidence_gates(change):
    documents, ledger, draft = packet()
    doc = draft["documents"][0]
    reading = doc["readings"][1]
    if change == "one_pass":
        doc["readings"].pop()
    elif change == "wrong_pass":
        reading["pass_number"] = 1
    elif change == "duplicate_page":
        reading["page_reviews"] *= 2
    elif change == "different_page_hash":
        reading["page_reviews"][0]["page_text_sha256"] = "b" * 64
    elif change == "predictions":
        reading["predictions_consulted"] = True
    elif change == "model_rules":
        reading["model_rules_consulted"] = True
    elif change == "different_pdf":
        doc["source_pdf_sha256"] = "b" * 64
    elif change == "missing_identity":
        reading["conditions"] = []
    elif change == "wrong_quote":
        doc["conditions"][0]["identity_evidence"][0]["quote"] = "wrong"
    elif change == "duplicate_condition":
        doc["conditions"] *= 2
    else:
        reading["page_reviews"][0]["content_status"] = "unreadable"
    with pytest.raises(ValueError):
        build_reference(draft, documents, ledger)


@pytest.mark.parametrize("field,value", [("billed_amount", "0.00"), ("billed_amount", "900"),
    ("billed_amount", 900), ("currency", "SOL"), ("lookback_years", True), ("lookback_years", 0)])
def test_no_reference_value_coercion(field, value):
    documents, ledger, draft = packet()
    draft["documents"][0]["conditions"][0]["fields"][field] = value
    with pytest.raises(ValueError):
        build_reference(draft, documents, ledger)


def test_known_values_require_literal_evidence_and_both_passes_agree():
    documents, ledger, draft = packet()
    draft["documents"][0]["conditions"][0]["field_evidence"]["currency"] = []
    with pytest.raises(ValueError, match="literal"):
        build_reference(draft, documents, ledger)
    documents, ledger, draft = packet()
    draft["documents"][0]["readings"][1]["conditions"][0]["fields"]["billed_amount"] = "901.00"
    with pytest.raises(ValueError, match="agreement"):
        build_reference(draft, documents, ledger)


def test_unknown_reference_field_is_ungraded_not_false_positive():
    documents, ledger, draft = packet()
    doc = draft["documents"][0]
    for c in [doc["conditions"][0], *(r["conditions"][0] for r in doc["readings"])]:
        c["fields"]["billed_amount"] = None
        c["unknown_reasons"]["billed_amount"] = "Synthetic unresolved source amount"
    result = compare(build_reference(draft, documents, ledger), predict_documents(documents), documents, ledger, binding(), binding())
    amount = result["metrics"]["qualification"]["fields"]["billed_amount"]
    assert amount["tp"] == amount["fp"] == amount["fn"] == 0
    assert amount["reference_unknown"] == amount["predictions_ungraded_unknown_reference"] == 1
    assert amount["f1"] is None


def test_unsupported_tdr_not_aliased_and_separate_from_qualification():
    documents, ledger, draft = packet()
    doc = draft["documents"][0]
    for c in [doc["conditions"][0], *(r["conditions"][0] for r in doc["readings"])]:
        c["section_role"] = "tdr_supplier_profile"
    result = compare(build_reference(draft, documents, ledger), predict_documents(documents), documents, ledger, binding(), binding())
    assert result["metrics"]["tdr_supplier_profile"]["fields"]["currency"]["fn"] == 1
    assert result["metrics"]["qualification"]["fields"]["currency"]["fp"] == 1


def test_empty_reference_needs_two_scoped_absence_findings():
    documents, ledger, draft = packet()
    doc = draft["documents"][0]
    doc["conditions"] = []
    for r in doc["readings"]:
        r["conditions"] = []
    with pytest.raises(ValueError, match="absence"):
        build_reference(draft, documents, ledger)
    for r in doc["readings"]:
        r["scoped_absence_statement"] = "Synthetic supplied-document-only absence; not whole procurement file"
    locked = build_reference(draft, documents, ledger)
    assert locked["reference"]["documents"][0]["conditions"] == []


def test_unavailable_cases_never_become_negative_reference():
    documents, ledger, draft = packet()
    draft["documents"][1]["scope"] = "enterprise_services"
    with pytest.raises(ValueError, match="negative"):
        build_reference(draft, documents, ledger)


def test_unreadable_pages_remain_ungraded_insufficient_reference():
    documents, ledger, draft = packet()
    doc = draft["documents"][0]
    doc["scope"] = "insufficient_reference"
    for reading in doc["readings"]:
        reading["scope"] = "insufficient_reference"
    doc["readings"][1]["page_reviews"][0]["content_status"] = "unreadable"
    locked = build_reference(draft, documents, ledger)
    result = compare(locked, predict_documents(documents), documents, ledger, binding(), binding())
    stream = result["metrics"]["qualification"]
    assert stream["evaluable_documents"] == 0 and stream["predictions_outside_evaluable_scope"] == 1
    assert stream["fields"]["currency"]["f1"] is None


def test_blank_page_requires_archived_visual_evidence_and_two_absence_findings():
    documents, ledger, draft = packet(source(""))
    doc = draft["documents"][0]
    doc["conditions"] = []
    for reading in doc["readings"]:
        reading["conditions"] = []
        reading["scoped_absence_statement"] = "Synthetic visually reviewed blank, supplied document scope only"
        reading["page_reviews"][0].update(content_status="visually_verified_blank",
            image_evidence={"path": "/tmp/synthetic-blank-only.png", "sha256": "a" * 64})
    locked = build_reference(draft, documents, ledger)
    assert locked["visual_evidence_files"] == {"/tmp/synthetic-blank-only.png": "a" * 64}
    # Pure validation records a prerequisite; the artifact runner hashes the real
    # image before saving. This fixture never creates a visual review artifact.
    doc["readings"][0]["page_reviews"][0]["image_evidence"]["sha256"] = "bad"
    with pytest.raises(ValueError):
        build_reference(draft, documents, ledger)


def test_missing_reference_reason_and_inferred_item_are_rejected():
    documents, ledger, draft = packet()
    condition_ = draft["documents"][0]["conditions"][0]
    condition_["fields"]["currency"] = None
    with pytest.raises(ValueError, match="reasons"):
        build_reference(draft, documents, ledger)
    documents, ledger, draft = packet()
    draft["documents"][0]["conditions"][0]["item"] = True
    with pytest.raises(ValueError, match="Item"):
        build_reference(draft, documents, ledger)


def test_scope_cannot_be_changed_or_excluded_without_two_source_findings():
    documents, ledger, draft = packet()
    doc = draft["documents"][0]
    doc["scope"] = "outside_individual_professional"
    with pytest.raises(ValueError, match="scope disagreements"):
        build_reference(draft, documents, ledger)
    for reading in doc["readings"]:
        reading["scope"] = doc["scope"]
    with pytest.raises(ValueError, match="scope evidence"):
        build_reference(draft, documents, ledger)


@pytest.mark.parametrize("change", ["reference", "source", "design", "population"])
def test_reference_first_binding_and_population_checks(change):
    documents, ledger, draft = packet()
    locked = build_reference(draft, documents, ledger)
    actual = binding()
    if change == "population":
        locked["source_population_sha256"] = "b" * 64
    else:
        name = {"reference": "reference_sha256", "source": "source_manifest_sha256", "design": "evaluation_design_manifest_sha256"}[change]
        actual[name] = "b" * 64
    with pytest.raises(ValueError):
        compare(locked, predict_documents(documents), documents, ledger, actual, binding())


@pytest.fixture(scope="module")
def design(tmp_path_factory):
    from scripts.run_document_experience_evaluation_v1 import freeze
    directory = tmp_path_factory.mktemp("evaluation-design") / "lock"
    freeze(directory)
    return directory


def test_design_freeze_reproduces_registered_files_without_selecting(design, tmp_path):
    from scripts.run_document_experience_evaluation_v1 import freeze, validate_design
    lock = validate_design(design)
    assert lock["comparison_adapter_locked"] and lock["reference_first_gates_locked"]
    assert not lock["ready_to_select_new_evaluation_sample"] and not lock["new_sample_selected"]
    output = tmp_path / "replay"
    freeze(output)
    manifest = json.loads((design / "manifest.json").read_text())
    for name in (*manifest["output_sha256"], "manifest.json"):
        assert (design / name).read_bytes() == (output / name).read_bytes()
    with pytest.raises(ValueError):
        freeze(output)


def source_artifact(directory, design, documents, ledger):
    from scripts.register_browser_experience_sources import dump, sha
    from scripts.run_document_experience_evaluation_v1 import save
    lock = json.loads((design / "evaluation_design_lock.json").read_text())
    source_design = Path(lock["source_design_path"])
    save(directory, {"source_documents.json": documents, "coverage_ledger.json": ledger,
                    "summary.json": {"predictions_generated": False, "reference_created": False}},
         {str(source_design / "manifest.json"): sha(source_design / "manifest.json")})
    draft_path = directory.parent / "draft.json"
    dump(draft_path, packet(documents[0])[2])
    return draft_path


def test_artifact_workflow_reference_first_reproduces_and_model_gets_sources_only(design, tmp_path, monkeypatch):
    import scripts.run_document_experience_evaluation_v1 as run
    documents, ledger, _ = packet()
    source_dir = tmp_path / "source"
    draft_path = source_artifact(source_dir, design, documents, ledger)
    reference = tmp_path / "reference"
    run.lock_reference(design, source_dir, draft_path, reference)
    original = run.predict_documents
    passed = []
    def spy(value):
        passed.append(copy.deepcopy(value))
        return original(value)
    monkeypatch.setattr(run, "predict_documents", spy)
    inference = tmp_path / "inference"
    run.infer(design, source_dir, reference, inference)
    assert passed == [documents]
    output = tmp_path / "comparison"
    summary = run.evaluate_run(design, source_dir, reference, inference, output)
    assert summary["selected_cases"] == 50 and summary["metrics_computed"]
    assert not summary["expert_accuracy_proven"]
    replay = tmp_path / "replay"
    run.evaluate_run(design, source_dir, reference, inference, replay)
    for path in output.iterdir():
        assert path.read_bytes() == (replay / path.name).read_bytes()
    with pytest.raises(ValueError):
        run.lock_reference(design, source_dir, draft_path, reference)


def test_inference_without_reference_artifact_fails_before_model(design, tmp_path, monkeypatch):
    import scripts.run_document_experience_evaluation_v1 as run
    documents, ledger, _ = packet()
    source_dir = tmp_path / "source"
    source_artifact(source_dir, design, documents, ledger)
    monkeypatch.setattr(run, "predict_documents", lambda _: pytest.fail("Model must not run"))
    with pytest.raises(FileNotFoundError):
        run.infer(design, source_dir, tmp_path / "missing-reference", tmp_path / "output")


def test_self_reported_binding_without_manifest_prerequisite_is_rejected(design, tmp_path):
    import scripts.run_document_experience_evaluation_v1 as run
    documents, ledger, _ = packet()
    source_dir = tmp_path / "source"
    draft_path = source_artifact(source_dir, design, documents, ledger)
    reference = tmp_path / "reference"
    run.lock_reference(design, source_dir, draft_path, reference)
    inference = tmp_path / "forged-inference"
    declared = run.checked_reference(reference, design, source_dir, {})
    run.save(inference, {"candidate_predictions.json": predict_documents(documents),
                        "summary.json": {"binding": declared, "reference_values_passed_to_model": False}}, {})
    with pytest.raises(ValueError, match="prerequisite"):
        run.evaluate_run(design, source_dir, reference, inference, tmp_path / "comparison")


def test_observed_source_adapter_regression_stops_without_new_metrics(design, tmp_path):
    from scripts.run_document_experience_evaluation_v1 import regression
    output = tmp_path / "observed-regression"
    summary = regression(design, output)
    assert summary["documents"] == 27 and summary["original_occurrences"] == 27
    assert summary["candidate_units"] == 26 and summary["blocked_units"] == 1
    assert not summary["metrics_computed"] and not summary["old_case15_resolution_generalized"]
    replay = tmp_path / "replay-regression"
    regression(design, replay)
    for path in output.iterdir():
        assert path.read_bytes() == (replay / path.name).read_bytes()


def test_previously_observed_case_cannot_become_new_validation_source(design, tmp_path):
    import scripts.run_document_experience_evaluation_v1 as run
    lock = json.loads((design / "evaluation_design_lock.json").read_text())
    registry = json.loads((Path(lock["source_design_path"]) / "exclusion_registry.json").read_text())
    documents, ledger, _ = packet(source(ocid=registry["excluded_ocids"][0]))
    directory = tmp_path / "source"
    source_artifact(directory, design, documents, ledger)
    with pytest.raises(ValueError, match="Previously observed"):
        run.load_sources(directory, {}, design)
