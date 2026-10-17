import unittest
from datetime import date

from tesis_recommender.hybrid import HybridRanker, structured_signals
from tesis_recommender.metrics import evaluate_rankings
from tesis_recommender.schemas import Award, CompanyProfile, Opportunity
from tesis_recommender.semantic import DeterministicTestEmbedder


class AuditFixTests(unittest.TestCase):
    def test_macro_f1_and_short_ranking(self):
        result = evaluate_rankings({"a": ["x"], "b": ["y", "z"]}, {"a": {"x"}, "b": {"y", "z", "w"}}, (2,))
        self.assertEqual(result["precision@2"], 0.75)
        self.assertEqual(result["recall@2"], 0.833333)
        self.assertEqual(result["f1@2"], 0.733333)

    def test_empty_labels_and_invalid_rankings(self):
        self.assertEqual(evaluate_rankings({"a": ["x"]}, {})["f1@5"], 0)
        with self.assertRaises(ValueError):
            evaluate_rankings({"a": ["x", "x"]}, {"a": {"x"}})
        with self.assertRaises(ValueError):
            evaluate_rankings({}, {}, (0,))

    def test_currency_and_missing_explanation(self):
        profile = CompanyProfile("a", date(2025, 1, 1), [Award("h", "a", date(2024, 1, 1), "software", amount=100, currency="PEN")])
        for currency, expected in [("PEN", 1), ("USD", 0), ("", 0)]:
            opportunity = Opportunity("o", date(2025, 2, 1), "software", amount=100, currency=currency)
            self.assertEqual(structured_signals(profile, opportunity)["amount"], expected)
        ranker = HybridRanker([opportunity], DeterministicTestEmbedder())
        self.assertEqual(ranker.rank(profile)[0].explanation["amount_status"], "unavailable_or_not_comparable")
        for weights in ({"bm25": -1, "semantic": 2}, {"unknown": 1}, {}, {"bm25": float("nan")}):
            with self.assertRaises(ValueError):
                ranker.rank(profile, weights)

    def test_nonfinite_amount_excluded(self):
        profile = CompanyProfile("a", date(2025, 1, 1), [Award("h", "a", date(2024, 1, 1), "software", amount=float("inf"), currency="PEN")])
        self.assertIsNone(profile.median_amount_in("PEN"))

    def test_generic_service_category_is_not_matching_evidence(self):
        profile = CompanyProfile("a", date(2025, 1, 1), [Award("h", "a", date(2024, 1, 1), "software", category="services")])
        opportunity = Opportunity("o", date(2025, 2, 1), "limpieza", category="services")
        self.assertEqual(structured_signals(profile, opportunity)["category"], 0)
        result = HybridRanker([opportunity], DeterministicTestEmbedder()).rank(profile)[0]
        self.assertEqual(result.explanation["category_status"], "generic_not_informative")
