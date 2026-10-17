import copy
import json
from pathlib import Path

import pytest

from tesis_recommender.document_experience_dev_v1 import digest, extract_development as extract_v1
from tesis_recommender.document_experience_dev_v2 import (
    VERSION, extract_development, heading_key, verify_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
GENERAL = "El postor debe acreditar un monto facturado acumulado equivalente a "
MYPE = ("En el caso de postores que declaren en el Anexo N° 1 tener la condición de micro y pequeña empresa, "
        "se acredita una experiencia de ")
WINDOW = " durante los ocho (8) años anteriores a la fecha de presentación de ofertas."
QUAL = "3.2 REQUISITOS DE CALIFICACIÓN\n"
BIDDER = "A. EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD\n"


def pages(*texts):
    return [{"page": n, "text": text, "text_sha256": digest(text), "extraction_method": "synthetic_native_test"}
            for n, text in enumerate(texts, 1)]


@pytest.mark.parametrize("header", [
    "REQUISITOS DE LA CALIFICACIÓN", "3.2. REQUISITOS DE LA CALIFICACION:",
    "REQUISITOS DE CALIFICACIÓN OBLIGATORIOS", "REQUISITOS DE CALIFICACIO\u0301N",
])
def test_explicit_full_line_qualification_variants(header):
    source = pages(header + "\n" + BIDDER + GENERAL + "S/ 900.00." + WINDOW)
    before = copy.deepcopy(source)
    result = extract_development(source)
    row, = result["records"]
    assert row["section_role"] == "qualification"
    assert row["section_evidence"]["evidence"] == header + "\n"
    assert row["fields"]["billed_amount"] == "900.00"
    assert verify_provenance(result, source) and before == source


@pytest.mark.parametrize("header", [
    "EXPERIENCIA DEL POSTOR EN LĄ ESPECIALIDAD", "EXPERIÉNCIA DEL PÓSTOR EN LA ESPECIALIDAD",
    "EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD", "EXPERIENCIA DEL POSTOR EN LA ESPECIALIDA\u0301D",
])
def test_subject_diacritics_preserve_unmodified_source_line(header):
    source = pages(QUAL + header + "\n" + GENERAL + "S/ 900.00." + WINDOW)
    row, = extract_development(source)["records"]
    assert row["subject_evidence"]["evidence"] == header + "\n"
    assert row["subject_evidence"]["source_spans"][0]["page_text_sha256"] == source[0]["text_sha256"]
    assert heading_key(header) == "EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD"


@pytest.mark.parametrize("header", [
    "Consultar REQUISITOS DE LA CALIFICACIÓN.", "Según los requisitos de la calificación:",
    "REQUISISTOS DE CALIFICACION", "REQUISITOS DE LA CALIFICACIÓN; VER ANEXO",
])
def test_inline_mentions_and_ocr_word_errors_do_not_promote_role(header):
    result = extract_development(pages(header + "\n" + BIDDER + GENERAL + "S/ 900.00." + WINDOW))
    assert not result["records"]
    assert len(result["skipped_contexts"]) == 1


@pytest.mark.parametrize("subject", [
    "EXPERIENCIA DEL PERSONAL CLAVE", "EXPERIENCIA DEL POST0R EN LA ESPECIALIDAD",
    "Según EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD",
])
def test_never_fuzzily_repairs_subject_or_confuses_personnel(subject):
    result = extract_development(pages(QUAL + subject + "\n" + GENERAL + "S/ 900.00." + WINDOW))
    assert not result["records"]


@pytest.mark.parametrize("header", [
    "7.1. REQUISITOS DEL PROVEEDOR", "VI. REQUISITOS DEL PROVEEDOR Y/O PERSONAL (OBLIGATORIO)",
    "PERFIL DEL PROVEEDOR:",
])
def test_supplier_role_is_explicit_but_not_accepted_as_qualification(header):
    source = pages(QUAL + BIDDER + GENERAL + "S/ 900.00." + WINDOW + "\n" + header + "\n" +
                   BIDDER + GENERAL + "S/ 800.00." + WINDOW)
    result = extract_development(source)
    assert len(result["records"]) == 1
    row, = result["skipped_contexts"]
    assert row["role"] == "tdr_supplier_profile"
    assert row["section_evidence"]["evidence"] == header + "\n"
    assert verify_provenance(result, source)


def test_annex_and_chapter_reset_subject_and_item_context():
    result = extract_development(pages(QUAL + BIDDER + "ITEM 1\n" + GENERAL + "S/ 900.00." + WINDOW +
        "\nANEXO N° 11\n" + BIDDER + GENERAL + "S/ 800.00." + WINDOW +
        "\nCAPÍTULO IV\n" + BIDDER + GENERAL + "S/ 700.00." + WINDOW))
    assert [r["item"] for r in result["records"]] == [1]
    assert [r["role"] for r in result["skipped_contexts"]] == ["annex_form", "unknown"]


def test_diacritics_in_body_and_malformed_money_are_not_repaired():
    result = extract_development(pages(QUAL + BIDDER + GENERAL + "S/ 5,000.000.00." + WINDOW))
    row, = result["records"]
    assert row["fields"]["billed_amount"] is None
    assert "5,000.000.00" in row["amount_evidence"]["evidence"]
    # Heading normalization is not silently applied to the predicate body.
    assert not extract_development(pages(QUAL + BIDDER + GENERAL.replace("acreditar", "acręditar") + "S/ 900.00."))["records"]


@pytest.mark.parametrize("ref", ["section_evidence", "subject_evidence", "predicate_evidence"])
def test_context_lineage_tampering_is_rejected(ref):
    source = pages(QUAL + BIDDER + GENERAL + "S/ 900.00." + WINDOW)
    result = extract_development(source)
    result["records"][0][ref]["source_spans"][0]["evidence"] += "modified"
    with pytest.raises(ValueError):
        verify_provenance(result, source)


@pytest.fixture(scope="module")
def observed_sources():
    return {d["sample_number"]: d for d in json.loads((ROOT / "artifacts/experience_holdout_source_view_v1/source_documents.json").read_text())}


def test_observed_heading_regressions_not_unseen_accuracy(observed_sources):
    source = observed_sources[36]["pages"]
    result = extract_development(source)
    row = next(r for r in result["records"] if r["page"] == 43)
    assert "LĄ ESPECIALIDAD" in row["subject_evidence"]["evidence"]
    assert row["fields"]["billed_amount"] is None
    assert "5,000.000.00" in row["amount_evidence"]["evidence"]
    # Recognition of the heading does not supply missing/unsupported fields.
    assert row["fields"]["lookback_years"] is None
    row = next(r for r in extract_development(observed_sources[39]["pages"])["records"] if r["page"] == 27)
    assert "REQUISITOS DE LA CALIFICACIÓN" in row["section_evidence"]["evidence"]
    assert row["fields"] == {"billed_amount": "700000.00", "currency": "PEN", "lookback_years": 8}


def test_observed_tdr_remains_separate(observed_sources):
    for number in (41, 48):
        result = extract_development(observed_sources[number]["pages"])
        assert len(result["skipped_contexts"]) == 2
        assert all(r["role"] == "tdr_supplier_profile" for r in result["skipped_contexts"])
        assert verify_provenance(result, observed_sources[number]["pages"])


def test_all_prior_emissions_unchanged_and_source_provenance_valid(observed_sources):
    for doc in observed_sources.values():
        source = doc["pages"]
        old = extract_v1(source)
        result = extract_development(source)
        current = {(r["page"], r["clause_evidence"]["source_spans"][0]["char_start"], r["branch"]): r for r in result["records"]}
        for prior in old["records"]:
            key = prior["page"], prior["clause_evidence"]["source_spans"][0]["char_start"], prior["branch"]
            row = current[key]
            for field in ("fields", "item", "amount_status", "clause_evidence", "warnings"):
                assert row[field] == prior[field]
        assert verify_provenance(result, source)
        assert result["version"] == VERSION and not result["f1_computed"]
        assert not result["automatic_deduplication_performed"] and not result["ranking_eligible"]


def test_missing_archived_ocr_clause_cannot_be_filled_from_reference(observed_sources):
    source = observed_sources[16]["pages"]
    assert not extract_development(source)["records"]
    page = next(p for p in source if p["page"] == 36)
    assert "MILLONES CON 00/100" in page["text"] and GENERAL not in page["text"]


@pytest.fixture(scope="module")
def development_artifact(tmp_path_factory):
    from scripts.run_document_experience_development_v2 import build
    output = tmp_path_factory.mktemp("document-development-v2") / "candidates"
    return output, build(ROOT / "artifacts/experience_holdout_source_view_v1", output)


def test_persisted_review_keeps_all_skipped_source_locations(development_artifact):
    output, summary = development_artifact
    review = json.loads((output / "skipped_context_review.json").read_text())
    assert len(review) == 6
    assert summary["context_review_counts"] == {"recognized_full_line_heading_context": 2, "unsupported_role_retained": 4}
    assert summary["candidate_occurrences"] == 27
    assert not summary["f1_computed"] and not summary["independent_validation_performed"]
    assert all(not row["semantic_accuracy_adjudicated"] for row in review)


def test_development_v2_reproduces_bytes_and_rejects_overwrite(development_artifact, tmp_path):
    from scripts.run_document_experience_development_v2 import build
    original, _ = development_artifact
    replay = tmp_path / "replay"
    build(ROOT / "artifacts/experience_holdout_source_view_v1", replay)
    assert sorted(p.name for p in original.iterdir()) == sorted(p.name for p in replay.iterdir())
    for path in original.iterdir():
        assert path.read_bytes() == (replay / path.name).read_bytes()
    with pytest.raises(ValueError):
        build(ROOT / "artifacts/experience_holdout_source_view_v1", replay)
