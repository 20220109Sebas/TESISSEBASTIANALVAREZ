import pytest

from tesis_recommender.human_relevance_v1 import evaluate_pool


def fixture():
    pool = [dict(pair_id=str(i), supplier_id="company", opportunity_id=str(i)) for i in range(12)]
    ranks = {"a": {"company": [str(i) for i in range(10)]},
             "b": {"company": [str(i) for i in range(2, 12)]}}
    judgments = [dict(pair_id=str(i), grade=2 if i < 5 else 0, rationale="Servicios compatibles") for i in range(12)]
    return pool, ranks, judgments


def test_incomplete_review_blocks_metrics():
    p, r, j = fixture()
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate_pool(p, r, j[:-1])


def test_fixed_denominators_and_macro():
    result = evaluate_pool(*fixture())
    assert result["macro"]["a"]["precision_clear@10"] == .5
    assert result["macro"]["b"]["precision_clear@10"] == .3
    assert result["macro"]["a"]["ndcg_pool@10"] == 1
    assert result["macro"]["b"]["recall_clear_pool@10"] == .6


def test_NE_is_unknown_with_bounds_not_negative():
    p, r, j = fixture()
    j[0]["grade"] = "NE"
    result = evaluate_pool(p, r, j)["macro"]["a"]
    assert result["precision_clear@10"] is None
    assert result["precision_clear_lower@10"] == .4
    assert result["precision_clear_upper@10"] == .5
    assert result["ndcg_pool@10"] is None


def test_unknown_outside_top10_still_blocks_pool_metrics():
    p, r, j = fixture()
    j[11]["grade"] = "NE"
    result = evaluate_pool(p, r, j)["macro"]["a"]
    assert result["precision_clear@10"] == .5
    assert result["ndcg_pool@10"] is None


@pytest.mark.parametrize("grade", [True, 3, -1, "2", None])
def test_invalid_grade_rejected(grade):
    p, r, j = fixture()
    j[0]["grade"] = grade
    with pytest.raises(ValueError):
        evaluate_pool(p, r, j)


def test_duplicate_unknown_and_unexplained_rejected():
    p, r, j = fixture()
    for bad in (j + [j[0]], j + [dict(pair_id="extra", grade=0, rationale="no")],
                [dict(v, rationale="") for v in j]):
        with pytest.raises(ValueError):
            evaluate_pool(p, r, bad)


def test_zero_relevance_has_no_invented_ndcg_or_recall():
    p, r, j = fixture()
    result = evaluate_pool(p, r, [dict(v, grade=0) for v in j])["macro"]["a"]
    assert result["precision_clear@10"] == 0
    assert result["ndcg_pool@10"] is None
    assert result["recall_clear_pool@10"] is None
