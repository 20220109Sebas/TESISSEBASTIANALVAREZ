import pytest

from tesis_recommender.reranking_v1 import bounded_pair, rerank_prefix


class Encoding:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1] * len(ids)

    def truncate(self, size, direction):
        self.ids = self.ids[:size]


class Tokenizer:
    @property
    def backend_tokenizer(self):
        return self

    def encode(self, text, add_special_tokens=False):
        return Encoding(list(range(len(text.split()))))

    def post_process(self, query, document, **kwargs):
        return Encoding([0] + query.ids + [2, 2] + document.ids + [2])


def test_budget_records_truncation_and_keeps_both_inputs():
    encoded, audit = bounded_pair(Tokenizer(), "x " * 200, "y " * 600)
    assert len(encoded["input_ids"]) == 512
    assert audit == dict(history_tokens=200, opportunity_tokens=600, history_truncated=True,
                        opportunity_truncated=True, input_tokens=512)


def test_empty_pair_is_rejected():
    with pytest.raises(ValueError):
        bounded_pair(Tokenizer(), "", "service")


def test_prefix_keeps_complete_population_and_tail():
    original = ["b", "a", "d", "c"]
    assert rerank_prefix(original, {"b": 1., "a": 1.}) == ["a", "b", "d", "c"]
    assert original == ["b", "a", "d", "c"]


@pytest.mark.parametrize("scores", [{}, {"b": float("nan")}, {"x": 1.}, {"a": 2.}])
def test_invalid_scores_do_not_silently_drop_candidates(scores):
    with pytest.raises(ValueError):
        rerank_prefix(["b", "a", "c"], scores)
