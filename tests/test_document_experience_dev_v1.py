import copy
import json
from pathlib import Path

import pytest

from tesis_recommender.document_experience_dev_v1 import (
    digest, extract_development, money_at, parse_number, verify_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
GENERAL = "El postor debe acreditar un monto facturado acumulado equivalente a "
MYPE = ("En el caso de postores que declaren en el Anexo N° 1 tener la condición de micro y pequeña empresa, "
        "se acredita una experiencia de ")
WINDOW = " durante los ocho (8) años anteriores a la fecha de presentación de ofertas."
HEADER = "3.2 REQUISITOS DE CALIFICACIÓN\nA. EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD\n"


def pages(*texts):
    return [{"page": i, "text": text, "text_sha256": digest(text), "extraction_method": "synthetic_native_test"}
            for i, text in enumerate(texts, 1)]


@pytest.mark.parametrize("raw,expected", [
    ("100000", "100000.00"), ("100,000.00", "100000.00"), ("100.000,00", "100000.00"),
    ("14 000,00", "14000.00"), ("300 000", "300000.00"), ("39\u00a0000,00", "39000.00"),
    ("1´200,000.00", "1200000.00"), ("1’200’000.00", "1200000.00"),
    ("1 00,000.00", "100000.00"), ("1\n000,00", "1000.00"),
    ("7.50", "7.50"), ("7,50", "7.50"), ("1,234", "1234.00"),
])
def test_complete_supported_money_formats(raw, expected):
    assert parse_number(raw) == expected
    result = money_at("S/ " + raw + " (importe)", 0)
    assert result["billed_amount"] == expected and result["currency"] == "PEN"
    assert result["status"] == "complete_token_normalized"
    assert ("S/ " + raw)[result["start"]:result["end"]] == "S/ " + raw


@pytest.mark.parametrize("raw", [
    "1 00 000", "12 34", "1000.123", "1.234", "1,23,456.00", "1..200.00", "0", "-10",
    "12 34,567.00", "100.00.00", "NaN", "1000,1234", "12\n34", "1O00.00",
])
def test_malformed_numbers_are_not_repaired_or_rounded(raw):
    assert parse_number(raw) is None
    assert money_at("S/ " + raw + " (importe)", 0)["billed_amount"] is None


@pytest.mark.parametrize("text", ["S/ 100/200", "S/ 100+200", "S/ 100-200", "S/ 100O", "S/ 1000.123"])
def test_never_accepts_numeric_prefix_of_invalid_token(text):
    assert money_at(text, 0)["billed_amount"] is None


def test_currency_brackets_and_explicit_centavos_conflict():
    assert money_at("[USD 1,500.25]", 0)["currency"] == "USD"
    assert money_at("[USD 1,500.25]", 0)["billed_amount"] == "1500.25"
    row = money_at("S/. 21,995.00 (Veintiún mil novecientos noventa y cinco CON 50/100)", 0)
    assert row["billed_amount"] is None and row["currency"] == "PEN"
    assert row["warnings"] == ["explicit_centavos_conflict_digits_vs_parenthetical"]
    assert money_at("S/ 12.50 (doce CON 50/100)", 0)["billed_amount"] == "12.50"


def test_two_explicit_items_and_mype_are_kept_separate():
    source = pages(HEADER + "ITEM 1\n" + GENERAL + "S/ 1 00 000 (ciento cien mil)." + WINDOW + "\n" +
        MYPE + "S/ 14 000,00 (catorce mil con 00/100)." + WINDOW + "\nAcreditación:\nTexto.\n",
        "Otra información de acreditación.\nITEM 2\n" + GENERAL + "S/ 300 000 (trescientos mil)." + WINDOW +
        "\n" + MYPE + "S/ 39 000,00 (treinta y nueve mil con 00/100)." + WINDOW + "\nAcreditación:\nTexto.")
    before = copy.deepcopy(source)
    result = extract_development(source)
    assert [(r["item"], r["branch"]) for r in result["records"]] == [(1, "general"), (1, "mype"), (2, "general"), (2, "mype")]
    assert [r["fields"]["billed_amount"] for r in result["records"]] == [None, "14000.00", "300000.00", "39000.00"]
    assert all(r["item_evidence"] for r in result["records"])
    assert verify_provenance(result, source) and source == before
    assert not result["automatic_deduplication_performed"] and not result["f1_computed"]


def test_item_in_general_table_does_not_assign_experience_item():
    source = pages("ITEM 01\nTabla general.\n" + HEADER + GENERAL + "S/ 900.00." + WINDOW)
    result = extract_development(source)
    assert result["records"][0]["item"] is None
    assert result["records"][0]["item_evidence"] is None


def test_role_change_resets_item_and_personnel_or_annex_is_not_bidder():
    source = pages(HEADER + "ITEM 1\n" + GENERAL + "S/ 900.00." + WINDOW +
        "\nB. EXPERIENCIA DEL PERSONAL CLAVE\n" + GENERAL + "S/ 800.00." + WINDOW +
        "\nANEXO N° 11\nEXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD\n" + GENERAL + "S/ 700.00." + WINDOW)
    result = extract_development(source)
    assert len(result["records"]) == 1
    assert len(result["skipped_contexts"]) == 2
    assert result["records"][0]["fields"]["billed_amount"] == "900.00"


def test_unknown_heading_role_and_inline_reference_do_not_become_qualification():
    source = pages("Consultar requisitos de calificación.\nEXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD\n" + GENERAL + "S/ 900.00." + WINDOW)
    result = extract_development(source)
    assert not result["records"] and len(result["skipped_contexts"]) == 1


def test_mype_variant_does_not_inherit_general_or_annex_window():
    variant = ("En caso de postores que declaren en el anexo N° 1 tener condición de micro o pequeña empresa.\n"
               "se acreditara\nuna experiencia de S/ 37,500.00 (con 00/100).")
    source = pages(HEADER + GENERAL + "S/ 150,000.00 (con 00/100)." + WINDOW + "\n" + variant +
        "\nSe consideran servicios similares a los siguientes: servicios.\nAcreditación:\n" + WINDOW)
    result = extract_development(source)
    assert len(result["records"]) == 2
    general, mype = result["records"]
    assert general["fields"]["lookback_years"] == 8
    assert mype["fields"]["billed_amount"] == "37500.00"
    assert mype["fields"]["lookback_years"] is None and mype["window_evidence"] is None


def test_cross_page_window_and_source_methods_preserved():
    source = pages(HEADER + GENERAL + "S/ 1´200,000.00 (Un Millón\n",
        "Cabecera de página\nDoscientos Mil y 00/100), por servicios." + WINDOW +
        "\n" + MYPE + "S/ 118,000.00." + WINDOW)
    source[1].update(extraction_method="synthetic_OCR_test", ocr_json_path="test-only.json", ocr_json_sha256="a" * 64)
    result = extract_development(source)
    assert result["records"][0]["fields"]["billed_amount"] == "1200000.00"
    assert result["records"][0]["fields"]["lookback_years"] == 8
    assert len(result["records"][0]["clause_evidence"]["source_spans"]) == 2
    assert verify_provenance(result, source)
    altered = copy.deepcopy(result)
    altered["records"][0]["window_evidence"]["source_spans"][0]["extraction_method"] = "fake"
    with pytest.raises(ValueError):
        verify_provenance(altered, source)


def test_relative_requirement_kept_without_budget_or_currency_inference():
    source = pages("Cuantía S/ 700000.00\n" + HEADER + GENERAL + "UNA VEZ LA CUANTÍA DEL PROCEDIMIENTO DE SELECCIÓN." + WINDOW)
    row = extract_development(source)["records"][0]
    assert row["fields"] == {"billed_amount": None, "currency": None, "lookback_years": 8}
    assert row["amount_status"] == "no_literal_absolute_money"


@pytest.mark.parametrize("mutation", ["hash", "duplicate", "out_of_order", "gap"])
def test_bad_source_layout_is_rejected(mutation):
    source = pages(HEADER, GENERAL + "S/ 100.00." + WINDOW)
    if mutation == "hash":
        source[0]["text"] += "changed"
    elif mutation == "duplicate":
        source[1]["page"] = 1
    elif mutation == "out_of_order":
        source.reverse()
    elif mutation == "gap":
        source[1]["page"] = 3
    with pytest.raises(ValueError):
        extract_development(source)


@pytest.fixture(scope="module")
def observed_sources():
    return {d["sample_number"]: d for d in json.loads((ROOT / "artifacts/experience_holdout_source_view_v1/source_documents.json").read_text())}


def test_observed_development_regressions_not_unseen_validation(observed_sources):
    # These sources motivated this version; successful regression is not test accuracy.
    for number, amount in ((8, "100000.00"), (48, "1200000.00")):
        doc = observed_sources[number]
        result = extract_development(doc["pages"])
        general = [r for r in result["records"] if r["branch"] == "general"]
        assert any(r["fields"]["billed_amount"] == amount for r in general)
        assert verify_provenance(result, doc["pages"])
    doc = observed_sources[20]
    result = extract_development(doc["pages"])
    assert [(r["item"], r["branch"], r["fields"]["billed_amount"]) for r in result["records"]] == [
        (1, "general", None), (1, "mype", "14000.00"), (2, "general", "300000.00"), (2, "mype", "39000.00")]
    assert verify_provenance(result, doc["pages"])
    mype = next(r for r in extract_development(observed_sources[13]["pages"])["records"] if r["branch"] == "mype")
    assert mype["fields"]["lookback_years"] is None
    mype = next(r for r in extract_development(observed_sources[41]["pages"])["records"] if r["branch"] == "mype")
    assert mype["fields"]["billed_amount"] is None


def test_repeated_source_conditions_are_not_silently_deduplicated(observed_sources):
    result = extract_development(observed_sources[15]["pages"])
    assert [r["page"] for r in result["records"]] == [42, 45]
    assert len({r["prediction_id"] for r in result["records"]}) == 2
    assert not result["automatic_deduplication_performed"]


def test_all_observed_sources_keep_traceability_and_development_flags(observed_sources):
    assert len(observed_sources) == 27
    for doc in observed_sources.values():
        result = extract_development(doc["pages"])
        assert verify_provenance(result, doc["pages"])
        assert result["stage"] == "development_on_previously_inspected_sources"
        assert not result["f1_computed"] and not result["ranking_eligible"]
        assert not result["independent_semantic_accuracy_proven"] and not result["archived_text_modified"]


@pytest.fixture(scope="module")
def development_artifact(tmp_path_factory):
    from scripts.run_document_experience_development_v1 import build
    output = tmp_path_factory.mktemp("document-development") / "candidates"
    source = ROOT / "artifacts/experience_holdout_source_view_v1"
    summary = build(source, output)
    return output, summary


def test_persisted_development_is_not_relabelled_as_independent_accuracy(development_artifact):
    output, summary = development_artifact
    assert summary["documents_processed"] == 27 and summary["physical_pages"] == 1667
    assert summary["all_documents_previously_inspected"] and summary["traceability_verified"]
    assert summary["original_frozen_evaluation_preserved"]
    assert all(not summary[k] for k in ("f1_computed", "independent_validation_performed", "generalization_proven",
        "reference_or_expected_values_used_for_inference", "new_ocr_performed", "archived_text_modified", "ranking_inputs_modified"))
    docs = json.loads((output / "document_candidates.json").read_text())
    records = [r for d in docs for r in d["result"]["records"]]
    assert summary["candidate_occurrences"] == len(records)
    assert summary["complete_numeric_tokens"] == sum(r["fields"]["billed_amount"] is not None for r in records)
    original = json.loads((ROOT / "artifacts/experience_holdout_concordance_v1/summary.json").read_text())
    assert original["field_metrics"]["accepted_branches"]["billed_amount"]["f1"] == pytest.approx(0.631578947368421)


def test_development_artifact_reproduces_without_replacing_frozen_code(development_artifact, tmp_path):
    from scripts.run_document_experience_development_v1 import build
    original, _ = development_artifact
    source = ROOT / "artifacts/experience_holdout_source_view_v1"
    output = tmp_path / "reproduction"
    build(source, output)
    assert sorted(p.name for p in output.iterdir()) == sorted(p.name for p in original.iterdir())
    assert all((output / p.name).read_bytes() == p.read_bytes() for p in original.iterdir())
    with pytest.raises(ValueError, match="fresh development directory"):
        build(source, output)
