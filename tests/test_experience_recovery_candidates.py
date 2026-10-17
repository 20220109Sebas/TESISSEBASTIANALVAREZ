import copy
import pytest
from tesis_recommender.experience_conditions import digest
from tesis_recommender.experience_segments import extract_experience_segments
from tesis_recommender.experience_recovery_candidates import recover_candidates
from scripts.recover_unresolved_experience import verify_candidate

PREFIX = "REQUISITOS DE CALIFICACIÓN\nC. EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD\nRequisitos:\n"
RELATIVE = "El postor debe acreditar un monto facturado acumulado equivalente a UNA VEZ LA CUANTÍA DEL\nPROCEDIMIENTO DE SELECCIÓN durante los veinte (20) años anteriores.\n"
DAMAGED = "postor debe acreditar un monto facturado acumulado equivalente a S/ 12,000.00\nSELLO\npor servicios durante los ocho (8) años anteriores.\n"


def run(text):
    pages = [{"page": 1, "text": text, "text_sha256": digest(text), "extraction_method": "native_pdf_text"}]
    frozen = extract_experience_segments(pages)
    return pages, frozen, recover_candidates(pages, frozen)


def test_relative_amount_remains_symbolic():
    pages, _, result = run(PREFIX + RELATIVE)
    row = result["records"][0]
    assert row["multiplier"] == 1
    assert row["absolute_amount"] is None
    assert row["currency"] is None
    assert row["lookback_years"] == 20
    verify_candidate(row, {1: pages[0]})


def test_damaged_predicate_keeps_literal_without_repair():
    pages, _, result = run(PREFIX + DAMAGED)
    row = result["records"][0]
    assert row["billed_amount"] == "12000.00"
    assert row["lookback_years"] == 8
    assert "postor debe" in row["context"]["evidence"]
    assert row["status"] == "requires_separate_review"
    verify_candidate(row, {1: pages[0]})


@pytest.mark.parametrize("prefix", ["ANEXO Nº 1\n", "OTRA SECCIÓN\n", "FACTORES DE EVALUACIÓN\n"])
def test_annex_unknown_and_factor_roles(prefix):
    _, _, result = run(prefix + PREFIX.split("\n", 1)[1] + RELATIVE)
    if prefix.startswith("FACTORES"):
        assert result["records"][0]["section_role"] == "evaluation_factor"
    else:
        assert not result["records"]


def test_personnel_not_bidder():
    _, _, result = run((PREFIX + DAMAGED).replace("EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD", "EXPERIENCIA DEL PERSONAL CLAVE"))
    assert not result["records"]


def test_accepted_frozen_clause_not_duplicated():
    _, frozen, result = run(PREFIX + DAMAGED.replace("postor debe", "El postor debe"))
    assert frozen["qualification_clauses"]
    assert not result["records"]


def test_mype_window_not_borrowed():
    text = PREFIX + DAMAGED.replace("durante los ocho (8) años anteriores", "antes")
    text += "En el caso de postores que declaren en el Anexo Nº 1 tener la condición de micro y pequeña empresa, se acredita una experiencia de S/ 100.00 durante los ocho (8) años anteriores."
    _, _, result = run(text)
    assert result["records"][0]["lookback_years"] is None


@pytest.mark.parametrize("boundary", ["Acreditación:", "D. EXPERIENCIA DEL PERSONAL CLAVE", "ANEXO Nº 8"])
def test_explicit_boundary_stops_context(boundary):
    _, _, result = run(PREFIX + boundary + "\n" + DAMAGED)
    assert not result["records"]


@pytest.mark.parametrize("amount", ["91,360.50.00", "CONSIGNAR MONTO", "0.00"])
def test_bad_money_never_repaired(amount):
    _, _, result = run(PREFIX + DAMAGED.replace("12,000.00", amount))
    assert not result["records"]


def test_saved_values_and_evidence_tampering_rejected():
    pages, _, result = run(PREFIX + RELATIVE)
    for key, value in (("multiplier", 3), ("absolute_amount", "100"), ("lookback_years", 8)):
        row = copy.deepcopy(result["records"][0])
        row[key] = value
        with pytest.raises(ValueError):
            verify_candidate(row, {1: pages[0]})


def test_no_mutation_or_validation_claim():
    pages, frozen, _ = run(PREFIX + DAMAGED)
    before = copy.deepcopy((pages, frozen))
    result = recover_candidates(pages, frozen)
    assert (pages, frozen) == before
    row = result["records"][0]
    assert not row["ranking_eligible"]
    assert not row["independently_validated"]
    assert not row["historical_content_proven"]
    assert not result["f1_estimable"]
