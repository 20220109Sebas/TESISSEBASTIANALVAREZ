import pytest
from scripts.build_holdout_reference_draft import check_literal_amount, ref


@pytest.mark.parametrize("raw,value", [("S/ 700,000.00", "700000.00"),
    ("S/ 14 000,00", "14000.00"), ("S/ 300 000", "300000.00")])
def test_reference_literal_formats(raw, value):
    check_literal_amount(raw, value)


def test_ambiguous_amount_can_remain_unresolved():
    check_literal_amount("S/ 1 00 000", None)


@pytest.mark.parametrize("raw,value", [("S/ 91,360.50.00", "91360.50"),
    ("S/ 300 000", "300"), ("S/ 0.00", "0.00"), ("CONSIGNAR MONTO", "100")])
def test_invalid_or_mismatched_value_rejected(raw, value):
    with pytest.raises(ValueError):
        check_literal_amount(raw, value)


def test_reference_keeps_exact_page_offsets():
    result = ref("antes S/ 100.00 después", 6, 15, 44)
    assert result["literal"] == "S/ 100.00"
    assert result["page"] == 44
    assert result["char_start"] == 6
