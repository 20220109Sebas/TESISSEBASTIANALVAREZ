import math

import pytest

from tesis_recommender.rank_fusion_v1 import ordered_ids, reciprocal_rank_fusion


def test_tie_and_exact_contributions():
    ranked, scores, positions = reciprocal_rank_fusion({"lexical": ["b", "a", "c"], "semantic": ["a", "b", "c"]})
    assert ranked == ["a", "b", "c"]
    assert scores["a"] == scores["b"] == 1 / 61 + 1 / 62
    assert positions["lexical"]["a"] == 2


def test_fusion_preserves_population_and_does_not_mutate_inputs():
    rankings = {"a": ["x", "y", "z"], "b": ["z", "y", "x"]}
    before = {k: v[:] for k, v in rankings.items()}
    ranked, _, _ = reciprocal_rank_fusion(rankings)
    assert set(ranked) == {"x", "y", "z"} and len(ranked) == 3
    assert rankings == before


def test_input_score_scale_does_not_change_rank_fusion():
    ids = ["a", "b", "c"]
    one = ordered_ids(ids, [0.01, 0.2, 0.9])
    two = ordered_ids(ids, [10, 200, 900])
    assert one == two
    assert reciprocal_rank_fusion({"x": one}) == reciprocal_rank_fusion({"x": two})


@pytest.mark.parametrize("rankings", [{}, {"a": []}, {"a": ["x", "x"]},
    {"a": ["x", "y"], "b": ["x"]}, {"a": [""]}])
def test_reject_missing_or_duplicate_population(rankings):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(rankings)


@pytest.mark.parametrize("constant", [0, -1, True, 0.5])
def test_reject_invalid_constant(constant):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"a": ["x"]}, constant)


@pytest.mark.parametrize("ids,scores", [(["a", "a"], [1, 2]), (["a"], [math.nan]),
    (["a"], [math.inf]), (["a"], []), ([], [])])
def test_invalid_component_inputs(ids, scores):
    with pytest.raises(ValueError):
        ordered_ids(ids, scores)
