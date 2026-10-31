import pytest

from tesis_recommender.review_context_v2 import facts


def inputs():
    h = [dict(award_id=str(i), award_date="2024-01-01", category=c, amount=a, currency=u)
         for i,(c,a,u) in enumerate([("Mantenimiento rutinario", 100, "PEN"),
             ("Transporte", 200, "PEN"), ("Mantenimiento rutinario", 10000, "USD"),
             ("Mantenimiento rutinario", 0, "PEN"), ("Transporte", 999, "")])]
    return dict(history=h, cutoff="2025-01-01"), dict(category="MANTENIMIENTO RUTINÁRIO", amount=1000, currency="PEN")


def test_same_currency_only_and_exact_category_evidence():
    r = facts(*inputs())
    assert r["total"] == 300
    assert r["maximum"] == 200
    assert r["same_category_maximum"] == 100
    assert r["ratio_to_maximum"] == 5
    assert r["exact_category_antecedents"] == [1, 3, 4]
    assert r["other_currency_history_count"] == 1
    assert r["missing_or_nonpositive_history_count"] == 2
    assert r["relevance_grade_suggested"] is None
    assert r["duration_verified"] is False


def test_routine_and_periodic_are_not_equated():
    c,o=inputs(); o["category"]="Mantenimiento periódico"
    assert facts(c,o)["exact_category_antecedents"] == []


@pytest.mark.parametrize("amount", [None, 0, -5, float("nan"), True])
def test_missing_amount_is_not_economic_incompatibility(amount):
    c,o=inputs(); o["amount"]=amount
    r=facts(c,o)
    assert r["ratio_to_maximum"] is None
    assert r["opportunity_amount_comparable"] is False


def test_no_history_in_same_currency_has_no_ratio_or_sum():
    c,o=inputs();o["currency"]="EUR"
    r=facts(c,o)
    assert r["total"] is None and r["ratio_to_maximum"] is None


def test_no_duplicate_or_future_history_in_summary():
    c,o=inputs();c["history"].append(c["history"][0])
    with pytest.raises(ValueError): facts(c,o)
    c,o=inputs();c["history"][0]["award_date"]="2025-01-01"
    with pytest.raises(ValueError): facts(c,o)
