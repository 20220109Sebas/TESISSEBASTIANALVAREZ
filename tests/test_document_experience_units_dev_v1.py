import copy
import json
from pathlib import Path

import pytest

from tesis_recommender.document_experience_dev_v1 import digest
from tesis_recommender.document_experience_dev_v2 import extract_development
from tesis_recommender.document_experience_units_dev_v1 import clause_key, project_units

ROOT = Path(__file__).resolve().parents[1]
HEADER = "3.2 REQUISITOS DE CALIFICACIÓN\nEXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD\n"
GENERAL = "El postor debe acreditar un monto facturado acumulado equivalente a "
WINDOW = " durante los ocho (8) años anteriores a la fecha de presentación de ofertas."
CLAUSE = GENERAL + "S/ 900.00." + WINDOW
MYPE = ("En el caso de postores que declaren en el Anexo N° 1 tener la condición de micro y pequeña empresa, "
        "se acredita una experiencia de S/ 100.00." + WINDOW)


def document(text, *, ocid="ocds-test-a", pdf="a" * 64):
    pages = [{"page": 1, "text": text, "text_sha256": digest(text), "extraction_method": "synthetic_native_test"}]
    return {"opportunity_id": ocid, "source_pdf_sha256": pdf, "result": extract_development(pages)}


def test_exact_whitespace_repeat_in_same_context_is_lossless():
    doc = document(HEADER + CLAUSE + "\n" + CLAUSE.replace("monto facturado", "monto\nfacturado"))
    before = copy.deepcopy(doc)
    result = project_units([doc])
    unit, = result["candidate_units"]
    assert unit["status"] == "exact_repeat_consolidated"
    assert unit["fields"] == {"billed_amount": "900.00", "currency": "PEN", "lookback_years": 8}
    assert len(unit["member_occurrence_ids"]) == len(result["occurrences"]) == 2
    assert result["identity_gate"]["passed"]
    assert doc == before
    assert not result["metrics_computed"] and not result["new_evaluation_ready"]


def test_same_explicit_item_with_repeated_context_can_consolidate():
    doc = document(HEADER + "ITEM 1\n" + CLAUSE + "\n" + HEADER + "ITEM 1\n" + CLAUSE)
    unit, = project_units([doc])["candidate_units"]
    assert unit["identity"]["item"] == 1 and unit["status"] == "exact_repeat_consolidated"


def test_unknown_item_across_sections_is_not_assumed_same_unit():
    doc = document(HEADER + CLAUSE + "\n" + HEADER + CLAUSE)
    result = project_units([doc])
    unit, = result["candidate_units"]
    assert unit["fields"] is None
    assert unit["reasons"] == ["unknown_item_across_distinct_context_anchors"]
    assert not result["identity_gate"]["passed"]
    assert result["identity_gate"]["action"] == "stop_before_field_comparison"


def test_equal_values_with_different_words_are_not_duplicate_proof():
    doc = document(HEADER + CLAUSE + "\n" + CLAUSE + " Únicamente para servicios eléctricos.")
    unit, = project_units([doc])["candidate_units"]
    assert "different_clause_text_same_candidate_identity" in unit["reasons"]
    assert unit["fields"] is None and len(unit["member_occurrence_ids"]) == 2


@pytest.mark.parametrize("change", [str.upper, lambda s: s.replace("900.00", "900,00"), lambda s: s.replace("ofertas.", "ofertas!")])
def test_case_punctuation_and_numeric_format_not_silently_normalized(change):
    doc = document(HEADER + CLAUSE + "\n" + change(CLAUSE))
    unit, = project_units([doc])["candidate_units"]
    assert "different_clause_text_same_candidate_identity" in unit["reasons"]


def test_only_whitespace_is_collapsed_for_clause_equivalence():
    assert clause_key("  uno\n\tdos\u00a0tres  ") == "uno dos tres"
    assert clause_key("Año") != clause_key("Ano") != clause_key("ano")


@pytest.mark.parametrize("field,new", [("billed_amount", "950.00"), ("currency", "USD"), ("lookback_years", 15)])
def test_conflicting_known_values_stop_without_choosing_a_value(field, new):
    doc = document(HEADER + CLAUSE + "\n" + CLAUSE)
    doc["result"]["records"][1]["fields"][field] = new
    result = project_units([doc])
    unit, = result["candidate_units"]
    assert unit["fields"] is None
    assert "conflicting_known_field_values" in unit["reasons"]
    assert new in unit["known_value_conflicts"][field]
    assert len(result["occurrences"]) == 2 and not result["identity_gate"]["passed"]


@pytest.mark.parametrize("field", ["billed_amount", "currency", "lookback_years"])
def test_known_null_disagreement_never_fills_from_other_occurrence(field):
    doc = document(HEADER + CLAUSE + "\n" + CLAUSE)
    record = doc["result"]["records"][1]
    record["fields"][field] = None
    if field == "billed_amount":
        record["amount_status"] = "ambiguous_number_abstained"
    unit, = project_units([doc])["candidate_units"]
    assert "known_vs_null_field_disagreement" in unit["reasons"]
    assert unit["fields"] is None


def test_equal_nulls_remain_abstentions_instead_of_becoming_values():
    relative = GENERAL + "UNA VEZ LA CUANTÍA DEL PROCEDIMIENTO." + WINDOW
    unit, = project_units([document(HEADER + relative + "\n" + relative)])["candidate_units"]
    assert unit["status"] == "exact_repeat_consolidated"
    assert unit["fields"] == {"billed_amount": None, "currency": None, "lookback_years": 8}


def test_same_fields_with_incompatible_warnings_stop():
    doc = document(HEADER + CLAUSE + "\n" + CLAUSE)
    doc["result"]["records"][1]["warnings"] = ["conflicting_literal"]
    unit, = project_units([doc])["candidate_units"]
    assert unit["reasons"] == ["inconsistent_parser_diagnostics"]
    assert unit["fields"] is None


def test_items_and_general_mype_branches_stay_distinct():
    doc = document(HEADER + "ITEM 1\n" + CLAUSE + "\n" + MYPE + "\nITEM 2\n" + CLAUSE)
    units = project_units([doc])["candidate_units"]
    assert {(u["identity"]["item"], u["identity"]["branch"]) for u in units} == {(1, "general"), (1, "mype"), (2, "general")}
    assert len(units) == 3 and all(u["status"] == "single_occurrence" for u in units)


def test_qualification_and_evaluation_factor_not_merged():
    doc = document(HEADER + CLAUSE + "\nFACTORES DE EVALUACIÓN\nEXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD\n" + CLAUSE)
    units = project_units([doc])["candidate_units"]
    assert {u["identity"]["section_role"] for u in units} == {"qualification", "evaluation_factor"}


def test_same_ocid_with_different_pdfs_requires_source_identity_resolution():
    result = project_units([document(HEADER + CLAUSE), document(HEADER + CLAUSE, pdf="b" * 64)])
    assert len(result["candidate_units"]) == 2
    assert all("multiple_pdf_sources_for_opportunity_unresolved" in u["reasons"] for u in result["candidate_units"])
    assert not result["identity_gate"]["passed"]


def test_same_pdf_under_different_ocids_not_merged_or_assumed_correct():
    result = project_units([document(HEADER + CLAUSE), document(HEADER + CLAUSE, ocid="ocds-other")])
    assert len(result["candidate_units"]) == 2
    assert all("same_pdf_under_multiple_opportunities_unresolved" in u["reasons"] for u in result["candidate_units"])


def test_distinct_ocid_and_pdf_remain_separate_and_order_is_deterministic():
    documents = [document(HEADER + CLAUSE), document(HEADER + CLAUSE, ocid="ocds-other", pdf="b" * 64)]
    result = project_units(documents)
    assert result["identity_gate"]["passed"] and len(result["candidate_units"]) == 2
    assert result == project_units(list(reversed(documents)))


@pytest.mark.parametrize("mode", ["document", "prediction_id", "location"])
def test_repeated_input_is_rejected_not_counted_as_source_repetition(mode):
    doc = document(HEADER + CLAUSE)
    documents = [doc]
    if mode == "document":
        documents.append(copy.deepcopy(doc))
    else:
        repeated = copy.deepcopy(doc["result"]["records"][0])
        if mode == "location":
            repeated["prediction_id"] += ":new-id-same-source"
        doc["result"]["records"].append(repeated)
    with pytest.raises(ValueError):
        project_units(documents)


@pytest.mark.parametrize("field,value", [("item", True), ("item", 0), ("item", 2), ("subject", "key_personnel"),
    ("section_role", "tdr_supplier_profile"), ("branch", "other"), ("ranking_eligible", True)])
def test_invalid_scope_and_item_identity_rejected(field, value):
    doc = document(HEADER + "ITEM 1\n" + CLAUSE)
    doc["result"]["records"][0][field] = value
    with pytest.raises(ValueError):
        project_units([doc])


@pytest.mark.parametrize("field,value", [("billed_amount", "0900.00"), ("billed_amount", 900),
    ("billed_amount", "0.00"), ("currency", "soles"), ("lookback_years", True), ("lookback_years", -1)])
def test_invalid_field_representation_rejected(field, value):
    doc = document(HEADER + CLAUSE)
    doc["result"]["records"][0]["fields"][field] = value
    with pytest.raises(ValueError):
        project_units([doc])


def test_changed_clause_hash_and_missing_context_are_rejected():
    for field in ("section_evidence", "subject_evidence", "clause_evidence"):
        doc = document(HEADER + CLAUSE)
        if field == "clause_evidence":
            doc["result"]["records"][0][field]["evidence"] += "changed"
        else:
            doc["result"]["records"][0][field] = None
        with pytest.raises(ValueError):
            project_units([doc])


def test_empty_predictions_do_not_prove_no_requirements():
    result = project_units([document("No supported clause in archived text.")])
    assert not result["candidate_units"] and not result["occurrences"]
    assert result["identity_gate"]["passed"] and not result["new_evaluation_ready"]


@pytest.fixture(scope="module")
def observed_documents():
    return json.loads((ROOT / "artifacts/document_experience_development_v2/document_candidates.json").read_text())


def test_observed_case15_is_not_auto_resolved_by_old_authorization(observed_documents):
    result = project_units(observed_documents)
    assert len(result["occurrences"]) == 27
    assert len(result["candidate_units"]) == 26
    blocked = [u for u in result["candidate_units"] if u["status"] == "unresolved_blocks_comparison"]
    unit, = blocked
    assert unit["identity"]["opportunity_id"] == next(d["opportunity_id"] for d in observed_documents if d["sample_number"] == 15)
    assert unit["reasons"] == ["unknown_item_across_distinct_context_anchors"]
    assert len(unit["member_occurrence_ids"]) == 2 and unit["fields"] is None
    assert not result["metrics_computed"]


@pytest.fixture(scope="module")
def unit_artifact(tmp_path_factory):
    from scripts.run_document_experience_units_development_v1 import build
    output = tmp_path_factory.mktemp("occurrence-policy") / "output"
    summary = build(ROOT / "artifacts/document_experience_development_v2",
                    ROOT / "artifacts/experience_holdout_source_view_v1", output)
    return output, summary


def test_persisted_projection_and_development_limits(unit_artifact):
    output, summary = unit_artifact
    assert summary["original_occurrences"] == 27 and summary["candidate_identity_groups"] == 26
    assert summary["unit_status_counts"] == {"single_occurrence": 25, "unresolved_blocks_comparison": 1}
    assert not summary["identity_gate_passed"] and not summary["prior_user_resolution_generalized"]
    assert not summary["metrics_computed"] and not summary["frozen_evaluation_modified"]
    assert (output / "occurrence_projection.json").exists()


def test_projection_reproduces_all_bytes_and_rejects_overwrite(unit_artifact, tmp_path):
    from scripts.run_document_experience_units_development_v1 import build
    original, _ = unit_artifact
    output = tmp_path / "replay"
    args = ROOT / "artifacts/document_experience_development_v2", ROOT / "artifacts/experience_holdout_source_view_v1"
    build(*args, output)
    for path in original.iterdir():
        assert path.read_bytes() == (output / path.name).read_bytes()
    with pytest.raises(ValueError):
        build(*args, output)
