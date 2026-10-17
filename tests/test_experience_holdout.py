import pytest
from scripts.prepare_experience_holdout import select_rows
from scripts.build_experience_holdout_packet import access_status


def row(ocid, **changes):
    return dict(opportunity_id=ocid, object_type="services", publication_date="2025-05-01",
                title="Convocatoria", description="Limpieza de instalaciones", **changes)


def test_sample_is_deterministic_and_independent_of_input_order():
    rows = [row(str(i)) for i in range(100)]
    assert select_rows(rows, set(), 50) == select_rows(list(reversed(rows)), set(), 50)


def test_excludes_development_ids():
    chosen, total = select_rows([row(str(i)) for i in range(60)], {str(i) for i in range(10)}, 50)
    assert total == 50
    assert not set(chosen) & {str(i) for i in range(10)}


def test_excludes_works_other_years_and_explicit_work_consultancy():
    rows = [row(str(i)) for i in range(3)]
    rows[0]["object_type"] = "works"
    rows[1]["publication_date"] = "2024-01-01"
    rows[2]["description"] = "CONSULTORÍA DE OBRA"
    rows.append(row("keep"))
    assert select_rows(rows, set(), 1) == (["keep"], 1)


def test_does_not_select_by_experience_or_award():
    rows = [row(str(i)) for i in range(10)]
    before = select_rows(rows, set(), 5)
    for item in rows:
        item["description"] = "Experiencia requerida ocho años"
        item["awarded_supplier_id"] = "futuro"
    assert before == select_rows(rows, set(), 5)


def test_duplicates_rejected():
    with pytest.raises(ValueError):
        select_rows([row("a"), row("a")], set(), 1)


def test_insufficient_universe_not_silently_replaced():
    with pytest.raises(ValueError):
        select_rows([row("a")], {"a"}, 1)


def test_duplicate_development_pdf_not_reference():
    assert access_status({"status": "downloaded_pdf", "sha256": "known"}, {"known"}, set()) == "excluded_duplicate_development_pdf"


def test_same_pdf_under_two_holdout_ocids_not_counted_twice():
    seen = set()
    item = {"status": "downloaded_pdf", "sha256": "new"}
    assert access_status(item, set(), seen) == "downloaded_unique_pdf"
    assert access_status(item, set(), seen) == "excluded_duplicate_holdout_pdf"


def test_partial_file_not_accepted_even_with_hash():
    assert access_status({"status": "too_large_partial_file", "sha256": "any"}, set(), set()) == "too_large_partial_file"
