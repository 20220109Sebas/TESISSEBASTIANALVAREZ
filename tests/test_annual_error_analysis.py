"""Controles sintéticos del diagnóstico, separados de los resultados científicos."""
import contextlib
import csv
import gzip
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_path = Path(__file__).resolve().parents[1] / "scripts/analyze_annual_errors.py"
_spec = importlib.util.spec_from_file_location("annual_error_analysis", _path)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


class AnnualErrorAnalysisTests(unittest.TestCase):
    def test_metrics_only_count_observed_pairs_and_use_k(self):
        result = audit.metrics(["unlabelled", "positive"], {"positive", "not_retrieved"}, 5)
        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["precision"], .2)
        self.assertEqual(result["recall"], .5)
        with self.assertRaises(ValueError):
            audit.metrics(["unlabelled"], set(), 5)

    def test_transition_does_not_call_unlabelled_candidates_negatives(self):
        result = audit.transition(["p", "u1"], ["q", "u2"], {"p", "q"}, 2)
        self.assertEqual(result["gained"], ["p"])
        self.assertEqual(result["lost"], ["q"])

    def test_deterministic_ties_and_exact_python_sum(self):
        components = {s: np.array([.91, .91]) for s in audit.SIGNALS}
        weights = dict(zip(audit.SIGNALS, (.35, .50, .08, .04, .03)))
        scores, ranking = audit.rank_components(components, weights, ["z", "a"])
        self.assertEqual(ranking, [1, 0])
        self.assertEqual(float(scores[0]), sum(weights[s] * float(components[s][0]) for s in audit.SIGNALS))

    def test_category_bonus_can_overturn_text_advantage(self):
        components = {"bm25": np.array([1., .95]), "semantic": np.array([1., .95]),
                      "category": np.array([0., 1.]), "region": np.zeros(2), "amount": np.zeros(2)}
        weights = dict(zip(audit.SIGNALS, (.35, .50, .08, .04, .03)))
        _, full = audit.rank_components(components, weights, ["target", "other"])
        _, removed = audit.rank_components(components, {**weights, "category": 0.}, ["target", "other"])
        self.assertEqual(full, [1, 0])
        self.assertEqual(removed, [0, 1])

    def test_nonfinite_scores_rejected(self):
        components = {s: np.array([float("nan")]) for s in audit.SIGNALS}
        with self.assertRaises(ValueError):
            audit.rank_components(components, {"bm25": 1.}, ["o"])

    def build_fixture(self, root):
        experiment, cohort = root / "experiment", root / "cohort"
        experiment.mkdir()
        cohort.mkdir()
        (experiment / "components").mkdir()
        (experiment / "rankings").mkdir()
        order = [f"o{i:02}" for i in range(20)]
        models = {"bm25": {"bm25": 1.}, "labse": {"semantic": 1.},
                  "bm25_labse": {"bm25": .412, "semantic": .588},
                  "hybrid_full": dict(zip(audit.SIGNALS, (.35, .50, .08, .04, .03)))}
        profiles = ["evaluable", "no_observed_positives"]
        paths_rows = [(cohort / "test_opportunities.csv", [{"opportunity_id": o, "amount": "0", "currency": "PEN", "category": "services", "region": "LIMA"} for o in order]),
                      (cohort / "historical_awards.csv", [{"supplier_id": sid} for sid in profiles]),
                      (root / "labels.csv", [{"supplier_id": "evaluable", "opportunity_id": "o15", "label": "1"}])]
        for path, rows in paths_rows:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        manifest = {"status": "complete", "models": models,
                    "input_sha256": {str(path): audit.digest(path) for path, _ in paths_rows},
                    "component_sha256": {}, "ranking_sha256": {}}
        runs = {m: {"top_k": {}, "metrics": {}} for m in models}
        for sid in profiles:
            components = {"bm25": np.arange(20, 0, -1, dtype=float) / 20,
                          "semantic": np.arange(20, dtype=float) / 20,
                          "category": np.arange(20) % 2, "region": np.ones(20), "amount": np.zeros(20)}
            component_path = experiment / "components" / f"{sid}.npz"
            np.savez_compressed(component_path, **components)
            manifest["component_sha256"][str(component_path.relative_to(experiment))] = audit.digest(component_path)
            for model, weights in models.items():
                scores, ranking = audit.rank_components(components, weights, order)
                top = [order[i] for i in ranking[:10]]
                runs[model]["top_k"][sid] = [{"opportunity_id": o} for o in top]
                if sid == "evaluable":
                    for k in (5, 10):
                        measures = audit.metrics(top, {"o15"}, k)
                        runs[model]["metrics"].update({f"{name}@{k}": measures[name] for name in ("precision", "recall", "f1")})
                path = experiment / "rankings" / f"{sid}__{model}.csv.gz"
                with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["supplier_id", "model", "rank", "opportunity_id", "score"])
                    writer.writerows([sid, model, pos, order[i], repr(float(scores[i]))] for pos, i in enumerate(ranking, 1))
                manifest["ranking_sha256"][str(path.relative_to(experiment))] = audit.digest(path)
        (experiment / "results.json").write_text(json.dumps({"runs": runs}))
        manifest["results_sha256"] = audit.digest(experiment / "results.json")
        (experiment / "candidate_order.json").write_text(json.dumps(order))
        manifest["candidate_order_sha256"] = audit.digest(experiment / "candidate_order.json")
        (experiment / "manifest.json").write_text(json.dumps(manifest))
        return SimpleNamespace(experiment=experiment, cohort=cohort, labels=root / "labels.csv", output=root / "diagnostic")

    def test_end_to_end_preserves_inputs_and_excludes_unknown_profiles_from_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            before = audit.digest(args.labels)
            with contextlib.redirect_stdout(io.StringIO()):
                report = audit.diagnose(args)
            self.assertEqual(report["profiles"], 2)
            self.assertEqual(report["profiles_with_known_positives"], 1)
            self.assertEqual(report["profiles_without_known_positives"], 1)
            self.assertEqual(report["revalidated_ranking_rows"], 80)
            self.assertEqual(before, audit.digest(args.labels))
            with self.assertRaises(ValueError):
                audit.diagnose(args)

    def test_changed_input_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            args.labels.write_text(args.labels.read_text() + "changed\n")
            with self.assertRaises(ValueError):
                audit.diagnose(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
