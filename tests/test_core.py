import sys
import unittest
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tesis_recommender.bm25 import BM25
from tesis_recommender.hybrid import HybridRanker
from tesis_recommender.metrics import evaluate_rankings, extraction_metrics, ndcg_at_k
from tesis_recommender.ocds import transform_release
from tesis_recommender.profiles import assert_no_temporal_leakage, build_profiles
from tesis_recommender.schemas import Award, Opportunity
from tesis_recommender.semantic import DeterministicTestEmbedder
from tesis_recommender.semantic import PrecomputedEmbedder
from tesis_recommender.embedding_cache import build_embedding_cache


class CoreTests(unittest.TestCase):
    def test_bm25_prefers_lexical_match(self):
        scores = BM25().fit(["limpieza hospitalaria", "desarrollo de software"]).score("software")
        self.assertGreater(scores[1], scores[0])

    def test_profiles_exclude_future_awards(self):
        awards = [
            Award("a1", "p1", date(2023, 1, 1), "software"),
            Award("a2", "p1", date(2024, 1, 1), "sistemas"),
            Award("future", "p1", date(2025, 2, 1), "futuro"),
        ]
        profiles = build_profiles(awards, date(2025, 1, 1))
        self.assertEqual([a.award_id for a in profiles["p1"].awards], ["a1", "a2"])
        assert_no_temporal_leakage(profiles)

    def test_metrics(self):
        metrics = evaluate_rankings({"p1": ["x", "good"], "p2": ["good2"]}, {"p1": {"good"}, "p2": {"good2"}}, (1, 2))
        self.assertEqual(metrics["hitrate@1"], 0.5)
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertEqual(metrics["mrr"], 0.75)

    def test_ordinal_and_extraction_metrics(self):
        self.assertEqual(ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 2), 0.796708)
        result = extraction_metrics([{"region": "Lima"}], [{"region": "Lima"}], ("region",))
        self.assertEqual(result["region"]["f1"], 1.0)

    def test_hybrid_returns_explanation_and_link(self):
        profile = build_profiles([
            Award("a1", "p1", date(2023, 1, 1), "desarrollo software", category="TI"),
            Award("a2", "p1", date(2024, 1, 1), "aplicaciones web", category="TI"),
        ], date(2025, 1, 1))["p1"]
        opportunities = [
            Opportunity("o1", date(2025, 2, 1), "servicio de software", category="TI", source_url="https://example.test/o1"),
            Opportunity("o2", date(2025, 2, 2), "limpieza de oficinas", category="Limpieza"),
        ]
        result = HybridRanker(opportunities, DeterministicTestEmbedder()).rank(profile, top_k=1)[0]
        self.assertEqual(result.opportunity_id, "o1")
        self.assertEqual(result.source_url, "https://example.test/o1")
        self.assertIn("dominant_signal", result.explanation)

    def test_ocds_service_release_transformation(self):
        release = {
            "ocid": "ocds-test-1", "date": "2025-01-10T00:00:00Z", "uri": "https://example.test/release",
            "tender": {"title": "Servicio de software", "description": "Desarrollo", "mainProcurementCategory": "services", "value": {"amount": 1000}},
            "awards": [{"id": "aw1", "date": "2025-02-01T00:00:00Z", "value": {"amount": 900}, "suppliers": [{"id": "supplier-1"}]}],
        }
        opportunity, awards = transform_release(release)
        self.assertEqual(opportunity.opportunity_id, "ocds-test-1")
        self.assertEqual(opportunity.publication_date, date(2025, 1, 10))
        self.assertEqual(awards[0].supplier_id, "supplier-1")

    def test_embedding_cache_aggregates_profiles_and_is_readable(self):
        awards = [
            Award("a1", "p1", date(2023, 1, 1), "desarrollo software"),
            Award("a2", "p1", date(2024, 1, 1), "aplicaciones web"),
        ]
        opportunities = [Opportunity("o1", date(2025, 2, 1), "servicio de software")]
        with tempfile.TemporaryDirectory() as directory:
            metadata = build_embedding_cache(
                awards,
                opportunities,
                Path(directory),
                DeterministicTestEmbedder(),
            )
            cached = PrecomputedEmbedder(directory)
            profile = build_profiles(awards, date(2025, 1, 1))["p1"]
            self.assertEqual(metadata["profiles"], 1)
            self.assertEqual(metadata["opportunities"], 1)
            self.assertEqual(len(cached.encode([profile.text, opportunities[0].text])), 2)


if __name__ == "__main__":
    unittest.main()
