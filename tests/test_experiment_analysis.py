import json
import tempfile
import unittest
from pathlib import Path

from tesis_recommender.experiment_analysis import query_metrics, paired_interval, analyze_experiment
from tesis_recommender.validation import write_csv


class ExperimentAnalysisTests(unittest.TestCase):
    def test_paired_constant_difference_and_short_ranking(self):
        interval = paired_interval([0.2, 0.2], [[0, 1], [1, 1], [0, 0]])
        self.assertEqual(interval, {"mean": 0.2, "ci95_low": 0.2, "ci95_high": 0.2})
        metrics = query_metrics(["positive"], {"positive"})
        self.assertEqual(metrics["precision@10"], 0.1)
        self.assertEqual(metrics["recall@10"], 1)
        with self.assertRaises(ValueError):
            query_metrics([], set())

    def test_analysis_preserves_unlabelled_profiles_and_checks_archived_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ranking = [{"opportunity_id": str(i)} for i in range(10)]
            metrics = query_metrics([str(i) for i in range(10)], {"0"})
            result = {"runs": {"bm25": {"top_k": {"p": ranking, "unlabelled": ranking}, "metrics": metrics}}}
            result_path = root / "results.json"
            result_path.write_text(json.dumps(result))
            labels = root / "labels.csv"
            awards = root / "awards.csv"
            write_csv(labels, [{"supplier_id": "p", "opportunity_id": "0", "label": 1}], ["supplier_id", "opportunity_id", "label"])
            write_csv(awards, [{"supplier_id": "p"}, {"supplier_id": "unlabelled"}], ["supplier_id"])
            report = analyze_experiment(result_path, labels, awards, root / "analysis", resamples=100)
            self.assertEqual(report["profiles_evaluable"], 1)
            self.assertEqual(report["profiles_without_known_positives"], 1)
            result["runs"]["bm25"]["metrics"]["recall@10"] = 0
            result_path.write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                analyze_experiment(result_path, labels, awards, root / "analysis", resamples=100)
