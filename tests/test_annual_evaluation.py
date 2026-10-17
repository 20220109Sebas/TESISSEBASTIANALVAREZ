import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import asdict

from tesis_recommender.annual_evaluation import iter_snapshot_records, classify_positive, MemmapEmbedder
from tesis_recommender.embedding_cache import build_embedding_cache
from tesis_recommender.hybrid import HybridRanker
from tesis_recommender.pipeline import ABLATIONS
from tesis_recommender.profiles import build_profiles
from tesis_recommender.schemas import Award, Opportunity
from tesis_recommender.semantic import DeterministicTestEmbedder, PrecomputedEmbedder
from tesis_recommender.validation import sha256, write_csv, read_csv


class AnnualEvaluationTests(unittest.TestCase):
    def test_source_label_audit_preserves_pool_and_profiles_without_positives(self):
        source = Path(__file__).resolve().parents[1] / "scripts/audit_annual_positives.py"
        spec = importlib.util.spec_from_file_location("audit_annual_positives_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cohort, corpus, cache, preparation = [root / name for name in ("cohort", "corpus", "cache", "preparation")]
            for p in (cohort, corpus, preparation):
                p.mkdir()
            awards = [Award(f"{p}-{i}", p, date(2024, 1, i + 1), "mantenimiento") for p in ("p1", "p2") for i in range(2)]
            opportunities = [Opportunity(ocid, date(2025, 3, 1), "mantenimiento") for ocid in ("o1", "o2")]
            write_csv(cohort / "historical_awards.csv", [asdict(a) for a in awards], list(asdict(awards[0])))
            write_csv(cohort / "test_opportunities.csv", [asdict(o) for o in opportunities], list(asdict(opportunities[0])))
            availability = [{"ocid": ocid, "status": "verified" if ocid != "o3" else "outside_publication_window",
                "text_published_at": "2025-03-01T12:00:00-05:00"} for ocid in ("o1", "o2", "o3")]
            write_csv(cohort / "opportunity_availability.csv", availability, list(availability[0]))
            cohort_hashes = {p.name: sha256(p) for p in cohort.glob("*.csv")}
            (cohort / "candidate_validation_report.json").write_text(json.dumps({"retrieval_complete": True, "evaluation_ready": True, "output_sha256": cohort_hashes}))
            build_embedding_cache(awards, opportunities, cache, DeterministicTestEmbedder())
            metadata = json.loads((cache / "metadata.json").read_text())
            plan = {field: metadata[field] for field in ("model_name", "model_revision", "profile_strategy", "opportunity_strategy")}
            source_root = source.parent.parent / "src/tesis_recommender"
            plan.update({"cutoff": "2025-01-01", "candidate_end": "2025-12-31", "evaluation_status": "test_only",
                "models": ABLATIONS, "source_sha256": {p.name: sha256(p) for p in source_root.glob("*.py")},
                "metrics": [f"{m}@{k}" for k in (5, 10) for m in ("precision", "recall", "f1")]})
            amendment = root / "test_amendment.json"
            amendment.write_text(json.dumps({"ranking_weights_unchanged": ABLATIONS,
                "updated_source_sha256": {"semantic.py": sha256(source_root / "semantic.py")}}))
            metadata["embedding_amendment"] = {"path": str(amendment), "sha256": sha256(amendment)}
            (cache / "metadata.json").write_text(json.dumps(metadata))
            (preparation / "frozen_plan_with_code.json").write_text(json.dumps(plan))
            (cache / "verification.json").write_text(json.dumps({"output_sha256": {name: sha256(cache / name) for name in ("metadata.json", "index.json", "embeddings.npy")}}))
            snapshot = root / "raw/2025/01/snapshot.json"
            snapshot.parent.mkdir(parents=True)
            observed = [("o1", "p1", "2025-03-02"), ("o2", "p1", "2025-02-28"), ("o3", "p2", "2025-03-03")]
            snapshot.write_text(json.dumps({"records": [{"ocid": ocid, "compiledRelease": {"ocid": ocid, "date": "2025-12-31",
                "awards": [{"id": "award", "date": award_date, "status": "active", "suppliers": [{"id": supplier}]}]}} for ocid, supplier, award_date in observed]}))
            write_csv(corpus / "lineage.csv", [{"ocid": ocid, "selected_snapshot_id": "2025-01", "compiled_release_date": "2025-12-31"} for ocid, _, _ in observed], ["ocid", "selected_snapshot_id", "compiled_release_date"])
            labels = [{"supplier_id": supplier, "opportunity_id": ocid, "label": 1, "award_date": award_date, "judgment_source": "historical_award"} for ocid, supplier, award_date in observed]
            write_csv(corpus / "historical_labels.csv", labels, list(labels[0]))
            (corpus / "corpus_report.json").write_text(json.dumps({"output_sha256": {p.name: sha256(p) for p in corpus.glob("*.csv")}, "snapshots": [{"snapshot": str(snapshot), "sha256": sha256(snapshot)}]}))
            args = SimpleNamespace(selection=root / "selection", preparation=preparation, cohort=cohort, cache=cache, corpus=corpus, output=root / "output")
            with patch.object(module, "candidate_queue", return_value=(plan, [{"ocid": ocid} for ocid, _, _ in observed])), contextlib.redirect_stdout(io.StringIO()):
                report = module.audit(args)
            self.assertEqual(report["verified_positive_pairs"], 1)
            self.assertEqual(report["profiles_without_known_positives"], 1)
            self.assertEqual(report["pair_statuses"], {"verified_positive": 1, "outcome_before_text_publication": 1, "candidate_not_retained": 1})
            self.assertEqual(report["native_award_statuses"], {"active": 3})
            self.assertTrue(report["inputs_and_embeddings_unchanged"])
            self.assertEqual(len(read_csv(cohort / "test_opportunities.csv")), 2)
            self.assertEqual(len(read_csv(args.output / "profile_label_coverage.csv")), 2)
            self.assertFalse(json.loads((args.output / "reveal_plan.json").read_text())["labels_opened_at_recording"])
            with self.assertRaisesRegex(ValueError, "No sobrescribir"):
                module.audit(args)
            runner_source = source.parent / "run_annual_contrast.py"
            runner_spec = importlib.util.spec_from_file_location("run_annual_contrast_test", runner_source)
            runner = importlib.util.module_from_spec(runner_spec)
            runner_spec.loader.exec_module(runner)
            run_args = SimpleNamespace(preparation=preparation, cohort=cohort, cache=cache, labels=args.output, output=root / "experiment")
            with contextlib.redirect_stdout(io.StringIO()):
                result = runner.run(run_args)
            self.assertEqual(result["population"]["profiles"], 2)
            self.assertEqual(result["population"]["candidates"], 2)
            self.assertEqual(result["population"]["profiles_without_known_positives"], 1)
            verifier_source = source.parent / "verify_annual_contrast.py"
            verifier_spec = importlib.util.spec_from_file_location("verify_annual_contrast_test", verifier_source)
            verifier = importlib.util.module_from_spec(verifier_spec)
            verifier_spec.loader.exec_module(verifier)
            verify_args = SimpleNamespace(experiment=run_args.output, cohort=cohort, cache=cache, labels=args.output, output=root / "verification.json")
            verified = verifier.verify(verify_args)
            self.assertTrue(verified["component_replay_exact"])
            self.assertEqual(verified["full_ranking_rows_verified"], 16)
            self.assertEqual(verified["top_k_explanations_verified"], 16)
            with self.assertRaisesRegex(ValueError, "No sobrescribir"):
                verifier.verify(verify_args)
            ranking = next((run_args.output / "rankings").glob("*.csv.gz"))
            ranking.write_bytes(b"tampered_test_only")
            verify_args.output = root / "tampered_verification.json"
            with self.assertRaisesRegex(ValueError, "Ranking modificado"):
                verifier.verify(verify_args)

    def test_streaming_json_matches_full_parser_with_small_unicode_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.json"
            records = [{"ocid": "ñá", "compiledRelease": {"date": "2025-02-03", "nested": [1.25, 12345, True, None], "text": 'braces }[ and " quotes'}}, {"ocid": "second"}]
            path.write_text(json.dumps({"uri": "source", "count": 12345, "records": records, "after": "ok"}, ensure_ascii=False))
            for size in (1, 7, 64):
                self.assertEqual(list(iter_snapshot_records(path, size)), records)
            for text in ('{"records": [', '{"records": [{}]} garbage', '{"records": [],"records": []}', '{"records": [3]}', '{"records": [],}', '{"records": [{},]}', '{"uri":"source"}'):
                path.write_text(text)
                with self.assertRaises(ValueError, msg=text):
                    list(iter_snapshot_records(path, 3))

    def test_positive_gate_preserves_same_day_and_does_not_filter_population(self):
        available = {"o": {"status": "verified", "text_published_at": "2025-03-02T02:00:00Z"},
                     "excluded": {"status": "outside_publication_window"}}
        row = {"supplier_id": "p", "opportunity_id": "o", "award_date": "2025-03-01"}
        self.assertEqual(classify_positive(row, {"p"}, available, date(2025, 1, 1)), ("verified_positive", "same_day_order_unresolved"))
        self.assertEqual(classify_positive({**row, "award_date": "2025-02-28"}, {"p"}, available, date(2025, 1, 1))[0], "outcome_before_text_publication")
        self.assertEqual(classify_positive({**row, "award_date": "2026-01-10"}, {"p"}, available, date(2025, 1, 1)), ("verified_positive", "later_day"))
        self.assertEqual(classify_positive({**row, "award_date": "invalid"}, {"p"}, available, date(2025, 1, 1))[0], "invalid_outcome_date")
        self.assertEqual(classify_positive({**row, "opportunity_id": "excluded"}, {"p"}, available, date(2025, 1, 1))[0], "candidate_not_retained")
        self.assertEqual(len(available), 2)

    def test_memmap_rankings_are_exactly_identical_to_precomputed_all_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            awards = [Award(f"{p}-{i}", p, date(2024, 1, i + 1), title, category="maintenance", region="Lima", amount=100 + i, currency="PEN")
                      for p, title in (("p1", "mantenimiento de equipos"), ("p2", "limpieza de oficinas")) for i in range(2)]
            opportunities = [Opportunity(f"o{i}", date(2025, 2, 1), title, category="maintenance", region="Lima", amount=100 + i, currency="PEN")
                for i, title in enumerate(["mantenimiento de equipos", "equipos", "limpieza de oficinas", "servicios médicos", "equipos", "auditoria financiera", "mantenimiento preventivo", "soporte técnico", "transporte", "limpieza"])]
            build_embedding_cache(awards, opportunities, root, DeterministicTestEmbedder())
            eager, lazy = PrecomputedEmbedder(root), MemmapEmbedder(root)
            texts = [o.text for o in opportunities]
            self.assertEqual(eager.encode(texts), list(lazy.encode(texts)))
            profiles = build_profiles(awards, date(2025, 1, 1))
            a, b = HybridRanker(opportunities, eager), HybridRanker(opportunities, lazy)
            for profile in profiles.values():
                for weights in ABLATIONS.values():
                    self.assertEqual(a.rank(profile, weights, None), b.rank(profile, weights, None))
                b._component_cache.clear()
            with self.assertRaises(KeyError):
                lazy.encode(["text not encoded"])
            index = json.loads((root / "index.json").read_text())
            index.append({**index[0], "row": (index[0]["row"] + 1) % len(lazy.matrix)})
            (root / "index.json").write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "ambiguo"):
                MemmapEmbedder(root)


if __name__ == "__main__":
    unittest.main()
