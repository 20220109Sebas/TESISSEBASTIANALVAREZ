"""Pruebas de la variante opt-in y su comparación reproducible de desarrollo."""
import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tesis_recommender.category_ranking import CategoryCodeSignal, SIGNALS, VARIANTS, rank_components, scoped_keys
from tesis_recommender.category_review import extract_categories
from tesis_recommender.experiment_analysis import query_metrics

_path = Path(__file__).resolve().parents[1] / "scripts/run_category_contrast.py"
_spec = importlib.util.spec_from_file_location("category_contrast", _path)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)
_verify_path = _path.with_name("verify_category_contrast.py")
_verify_spec = importlib.util.spec_from_file_location("category_verification", _verify_path)
verification = importlib.util.module_from_spec(_verify_spec)
_verify_spec.loader.exec_module(verification)


def feature(ocid, code="76122003", cubso="7612200300232630", supplier=None):
    value = extract_categories({"items": [{"classification": {"scheme": "CUBSO", "id": cubso, "description": "Categoría " + ocid},
                                          "additionalClassifications": [{"scheme": "UNSPSC", "id": code}]}]})
    value.update({"ocid": ocid, "source_sha256": "f" * 64, "source_uri": "https://example.test/" + ocid,
                  "release_published_at": "2024-01-01T00:00:00-05:00" if supplier else "2025-03-01T00:00:00-05:00"})
    if supplier:
        value.update({"supplier_id": supplier, "award_id": f"{ocid}:award:{supplier}"})
    return value


class CategoryRankingTests(unittest.TestCase):
    def test_shared_code_with_different_labels_is_explained(self):
        signal = CategoryCodeSignal([feature("h", supplier="p")], [feature("o", cubso="7612200300391331")], "UNSPSC", True)
        self.assertEqual(signal.scores("p").tolist(), [1.])
        explanation = signal.explanation("p", 0)
        self.assertEqual(explanation["shared_category_codes"], ["76122003"])
        self.assertEqual(explanation["historical_category_evidence"][0]["award_id"], "h:award:p")

    def test_unavailable_is_not_observed_no_match(self):
        history = feature("h", supplier="p")
        missing = feature("m", code="")
        different = feature("d", code="77111602")
        signal = CategoryCodeSignal([history], [missing, different], "UNSPSC", True)
        self.assertEqual(signal.scores("p").tolist(), [0., 0.])
        self.assertEqual(signal.explanation("p", 0)["category_status"], "unavailable")
        self.assertEqual(signal.explanation("p", 1)["category_status"], "observed_no_exact_match")
        with self.assertRaises(ValueError):
            signal.scores("unknown_supplier")

    def test_any_item_is_a_separate_rule(self):
        opportunity = feature("o", code="77111602")
        extra = feature("extra")["classifications"]
        opportunity["item_count"] = 2
        opportunity["classifications"].extend([{**entry, "item_index": 1} for entry in extra])
        history = [feature("h", supplier="p")]
        self.assertEqual(CategoryCodeSignal(history, [opportunity], "UNSPSC", True).scores("p").tolist(), [0.])
        self.assertEqual(CategoryCodeSignal(history, [opportunity], "UNSPSC", False).scores("p").tolist(), [1.])

    def test_invalid_flag_or_item_position_is_rejected(self):
        value = feature("o")
        value["classifications"][0]["valid_supported_code"] = False
        with self.assertRaises(ValueError):
            scoped_keys(value, "CUBSO", True)
        value = feature("o")
        value["classifications"][0]["item_index"] = 2
        with self.assertRaises(ValueError):
            scoped_keys(value, "CUBSO", True)

    def test_replacement_changes_only_category_and_preserves_python_sum(self):
        components = {s: np.array([.91, .91]) for s in SIGNALS}
        weights = dict(zip(SIGNALS, (.35, .50, .08, .04, .03)))
        original_category = components["category"].copy()
        scores, ranking = rank_components(components, weights, ["z", "a"], [1., 0.])
        self.assertEqual(ranking, [0, 1])
        self.assertEqual(float(scores[0]), sum(weights[s] * (1. if s == "category" else .91) for s in SIGNALS))
        self.assertTrue(np.array_equal(components["category"], original_category))
        _, ties = rank_components(components, weights, ["z", "a"])
        self.assertEqual(ties, [1, 0])

    def test_nonbinary_nonfinite_or_misaligned_signals_rejected(self):
        components = {s: np.ones(2) for s in SIGNALS}
        for value in ([.5, 1.], [float("nan"), 0.], [1.]):
            with self.assertRaises(ValueError):
                rank_components(components, {"bm25": 1.}, ["a", "b"], value)
        with self.assertRaises(ValueError):
            rank_components(components, {"bm25": .8}, ["a", "b"])

    def build_fixture(self, root):
        original, categories, cohort = [root / name for name in ("original", "categories", "cohort")]
        for path in (original, categories, cohort, original / "components", original / "rankings"):
            path.mkdir()
        features = [feature(f"o{i:02}", cubso="7612200300391331" if i % 2 else "7612200300232630") for i in range(12)]
        history = [feature("h1", supplier="p1"), feature("h2", supplier="p2")]
        order = [r["ocid"] for r in features]

        def csv_write(path, values):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)

        csv_write(cohort / "test_opportunities.csv", [{"opportunity_id": r["ocid"], "category": r["legacy_category"], "source_url": "https://example.test/record/" + r["ocid"]} for r in features])
        csv_write(cohort / "historical_awards.csv", [{"award_id": r["award_id"], "supplier_id": r["supplier_id"], "category": r["legacy_category"]} for r in history])
        labels = root / "labels.csv"
        csv_write(labels, [{"supplier_id": "p1", "opportunity_id": "o11", "label": "1"}])
        audit.save_json(original / "candidate_order.json", order)
        for name, values in (("opportunity_categories.jsonl", features), ("history_categories.jsonl", history)):
            (categories / name).write_text("".join(json.dumps(r) + "\n" for r in values))
        weights = dict(zip(SIGNALS, (.35, .50, .08, .04, .03)))
        models = {"bm25": {"bm25": 1.}, "labse": {"semantic": 1.}, "bm25_labse": {"bm25": .412, "semantic": .588}, "hybrid_full": weights}
        runs = {m: {"top_k": {}, "metrics": {}} for m in models}
        manifest = {"status": "complete", "models": models, "component_sha256": {}, "ranking_sha256": {}, "input_sha256": {}}
        for sid in ("p1", "p2"):
            components = {"bm25": np.arange(12, dtype=float) / 12, "semantic": np.arange(12, 0, -1, dtype=float) / 12,
                          "category": np.zeros(12), "region": np.ones(12), "amount": np.zeros(12)}
            path = original / "components" / f"{sid}.npz"
            np.savez_compressed(path, **components)
            manifest["component_sha256"][str(path.relative_to(original))] = audit.sha256(path)
            for model, coeffs in models.items():
                scores, ranking = rank_components(components, coeffs, order)
                top = [{"opportunity_id": order[i]} for i in ranking[:10]]
                runs[model]["top_k"][sid] = top
                if sid == "p1":
                    runs[model]["metrics"] = query_metrics([x["opportunity_id"] for x in top], {"o11"})
                if model == "hybrid_full":
                    path = original / "rankings" / f"{sid}__hybrid_full.csv.gz"
                    audit.save_ranking(path, sid, model, scores, ranking, order)
                    manifest["ranking_sha256"][str(path.relative_to(original))] = audit.sha256(path)
        audit.save_json(original / "results.json", {"runs": runs})
        frozen = root / "frozen_plan_with_code.json"
        audit.save_json(frozen, {"cutoff": "2025-01-01", "candidate_end": "2025-12-31"})
        manifest["input_sha256"] = {str(p): audit.sha256(p) for p in (cohort / "test_opportunities.csv", cohort / "historical_awards.csv", labels, frozen)}
        manifest.update({"results_sha256": audit.sha256(original / "results.json"), "candidate_order_sha256": audit.sha256(original / "candidate_order.json")})
        audit.save_json(original / "manifest.json", manifest)
        audit.save_json(categories / "manifest.json", {"labels_read": False, "rankings_run": False,
            "input_sha256": {str(p): audit.sha256(p) for p in (cohort / "test_opportunities.csv", cohort / "historical_awards.csv")},
            "output_sha256": {name: audit.sha256(categories / name) for name in ("opportunity_categories.jsonl", "history_categories.jsonl")}})
        return SimpleNamespace(original=original, categories=categories, cohort=cohort, labels=labels, output=root / "run")

    def test_full_workflow_is_opt_in_excludes_unknown_profiles_from_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            before = audit.sha256(args.original / "results.json")
            plan = audit.prepare(args)
            self.assertFalse(plan["labels_read_during_preparation"])
            self.assertEqual(plan["variants"], VARIANTS)
            with contextlib.redirect_stdout(io.StringIO()):
                result = audit.run(args.output / "plan.json")
            self.assertEqual(result["population"]["profiles_with_known_positives"], 1)
            self.assertEqual(result["population"]["profiles_without_known_positives"], 1)
            self.assertEqual(audit.sha256(args.original / "results.json"), before)
            manifest = audit.read_json(args.output / "manifest.json")
            self.assertEqual(manifest["reference_rows_reproduced_exactly"], 24)
            self.assertEqual(manifest["new_ranking_rows"], 96)
            self.assertFalse(manifest["default_ranker_changed"])
            self.assertEqual(len(result["comparisons"]), 84)
            with contextlib.redirect_stdout(io.StringIO()):
                verified = verification.verify(args.output)
            self.assertEqual(verified["new_ranking_rows_checked"], 96)
            self.assertEqual(verified["top10_explanations_checked"], 80)
            with self.assertRaises(ValueError):
                audit.run(args.output / "plan.json")

    def test_supplier_identity_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            path = args.categories / "history_categories.jsonl"
            records = [json.loads(x) for x in path.read_text().splitlines()]
            records[0]["supplier_id"] = "another_supplier"
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            manifest = audit.read_json(args.categories / "manifest.json")
            manifest["output_sha256"][path.name] = audit.sha256(path)
            audit.save_json(args.categories / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "otro proveedor"):
                audit.prepare(args)
            self.assertFalse(args.output.exists())

    def test_plan_changes_are_rejected_before_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            audit.prepare(args)
            path = args.output / "plan.json"
            path.write_text(path.read_text() + " ")
            with self.assertRaises(ValueError):
                audit.run(path)
            self.assertFalse((args.output / "rankings").exists())

    def test_post_cutoff_history_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            path = args.categories / "history_categories.jsonl"
            records = [json.loads(x) for x in path.read_text().splitlines()]
            records[0]["release_published_at"] = "2025-03-01T00:00:00-05:00"
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            manifest = audit.read_json(args.categories / "manifest.json")
            manifest["output_sha256"][path.name] = audit.sha256(path)
            audit.save_json(args.categories / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "antes del corte"):
                audit.prepare(args)

    def test_tampered_original_temporal_plan_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            path = Path(directory) / "frozen_plan_with_code.json"
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "Archivo modificado"):
                audit.prepare(args)
            self.assertFalse(args.output.exists())

    def test_verifier_recomputes_category_instead_of_trusting_new_components(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            audit.prepare(args)
            with contextlib.redirect_stdout(io.StringIO()):
                audit.run(args.output / "plan.json")
            path = args.output / "category_components/p1.npz"
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name] for name in VARIANTS}
            arrays["hybrid_unspsc_first"] = np.zeros(12)
            np.savez_compressed(path, **arrays)
            manifest = audit.read_json(args.output / "manifest.json")
            manifest["category_component_sha256"][str(path.relative_to(args.output))] = audit.sha256(path)
            audit.save_json(args.output / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "códigos publicados"):
                verification.verify(args.output)


if __name__ == "__main__":
    unittest.main()
