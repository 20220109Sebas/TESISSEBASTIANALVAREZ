import math
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tesis_recommender.semantic_aggregation import SourceVectors, OrderedVectors, maximum_similarity, rank_components, SIGNALS, text_hash


class SemanticAggregationTests(unittest.TestCase):
    def test_multiple_business_lines_max_and_duplicate_invariance(self):
        history = [[1., 0.], [0., 1.]]
        candidates = np.asarray([[1., 0.], [0., 1.], [-1., 0.]])
        raw, normalized, support = maximum_similarity(history, candidates, 1)
        np.testing.assert_array_equal(raw, [1., 1., 0.])
        np.testing.assert_array_equal(normalized, [1., 1., 0.])
        np.testing.assert_array_equal(support, [0, 1, 1])
        repeated = maximum_similarity(history + [history[0]], candidates, 2)
        for left, right in zip((raw, normalized, support), repeated):
            np.testing.assert_array_equal(left, right)

    def test_tie_selects_first_ordered_award(self):
        raw, _, support = maximum_similarity([[1., 0.], [0., 1.]], np.asarray([[2 ** -.5, 2 ** -.5]]))
        self.assertEqual(support.tolist(), [0])
        self.assertEqual(raw.tolist(), [2 ** -.5])

    def test_constant_minmax_matches_original_policy(self):
        for candidate, expected in (([1., 0.], 1.), ([0., 1.], 0.), ([-1., 0.], 0.)):
            _, normalized, _ = maximum_similarity([[1., 0.]], np.asarray([candidate, candidate]))
            self.assertEqual(normalized.tolist(), [expected, expected])

    def test_memmap_order_wrapper(self):
        matrix = np.asarray([[1., 0.], [0., 1.], [-1., 0.]])
        ordered = OrderedVectors(matrix, [2, 0, 1])
        raw, _, _ = maximum_similarity([[1., 0.]], ordered, 2)
        self.assertEqual(raw.tolist(), [-1., 1., 0.])
        with self.assertRaises(ValueError):
            OrderedVectors(matrix, [-1])

    def test_reject_nonfinite_empty_or_mismatched(self):
        for history, candidates in (([], np.ones((2, 2))), ([[1., 0.]], np.ones((2, 3))),
                                    ([[math.nan, 0.]], np.ones((2, 2))), ([[1., 0.]], np.asarray([[1., math.inf]])),
                                    ([[1., 0.]], np.empty((0, 2)))):
            with self.assertRaises(ValueError):
                maximum_similarity(history, candidates)

    def test_reject_zero_and_nonunit_vectors_in_every_block(self):
        for history, candidates in (([[0., 0.]], [[1., 0.]]),
                                    ([[2., 0.]], [[1., 0.]]),
                                    ([[1., 0.]], [[1., 0.], [0., 0.]]),
                                    ([[1., 0.]], [[1., 0.], [2., 0.]])):
            with self.assertRaises(ValueError):
                maximum_similarity(history, np.asarray(candidates), block_size=1)

    def test_preserve_cached_float32_approximate_unit_norm_without_renormalizing(self):
        value = float(np.float32(2 ** -.5))
        vector = [value, value]
        raw, _, _ = maximum_similarity([vector], np.asarray([vector]))
        self.assertEqual(raw[0], value * value + value * value)
        self.assertNotEqual(raw[0], 1.)

    def test_only_semantic_changes_python_sum_and_tie(self):
        components = {s: np.asarray([.1, .1]) for s in SIGNALS}
        before = {s: v.copy() for s, v in components.items()}
        weights = dict(zip(SIGNALS, [.35, .50, .08, .04, .03]))
        scores, order = rank_components(components, weights, ["z", "a"], np.asarray([.7, .7]))
        self.assertEqual(order, [1, 0])
        self.assertEqual(scores[0], sum(weights[s] * (.7 if s == "semantic" else .1) for s in SIGNALS))
        for s in SIGNALS:
            np.testing.assert_array_equal(components[s], before[s])

    def test_invalid_signal_and_weights(self):
        components = {s: np.asarray([.1]) for s in SIGNALS}
        for weights, order, semantic in (({"semantic": -1.}, ["a"], [0.]), ({"unknown": 1.}, ["a"], [0.]),
                                        ({"semantic": 1.}, ["a", "a"], [0., 0.]), ({"semantic": 1.}, ["a"], [math.nan])):
            with self.assertRaises(ValueError):
                rank_components(components, weights, order, semantic)

    def test_source_index_and_block_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            block = root / "block.npy"
            np.save(block, np.asarray([[1., 0.]], dtype=np.float32))
            fingerprint = dict.fromkeys(("model_name", "model_revision", "profile_strategy", "opportunity_strategy"), "fixture")
            source = {"fingerprint": fingerprint, "blocks": {"block.npy": {"shape": [1, 2], "sha256": hashlib.sha256(block.read_bytes()).hexdigest()}},
                      "texts": {text_hash("award"): {"block": "block.npy", "row": 0}}}
            snapshot = root / "source_store_index.json"
            snapshot.write_text(json.dumps(source))
            metadata = {**fingerprint, "source_store_index_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}
            (root / "metadata.json").write_text(json.dumps(metadata))
            store = SourceVectors(root, root)
            vector, evidence = store.get("award")
            np.testing.assert_array_equal(vector, [1., 0.])
            self.assertEqual(evidence["text_sha256"], text_hash("award"))
            with self.assertRaises(KeyError):
                store.get("unknown")
            np.save(block, np.asarray([[0., 1.]], dtype=np.float32))
            with self.assertRaises(ValueError):
                SourceVectors(root, root).get("award")
            snapshot.write_text("{}")
            with self.assertRaises(ValueError):
                SourceVectors(root, root)


if __name__ == "__main__":
    unittest.main()
