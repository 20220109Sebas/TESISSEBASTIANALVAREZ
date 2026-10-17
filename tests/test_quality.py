import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tesis_recommender.quality import audit_dataset
from tesis_recommender.schemas import Award, Opportunity


class QualityTests(unittest.TestCase):
    def test_audit_exposes_leakage_and_duplicates(self):
        awards = [Award("a", "p", date(2024, 1, 1), "uno"), Award("a", "p", date(2025, 1, 1), "dos")]
        opportunities = [Opportunity("o", date(2024, 12, 1), "prueba", object_type="goods", amount=-1)]
        audit = audit_dataset(awards, opportunities, date(2025, 1, 1))
        self.assertEqual(audit["uniqueness"]["duplicate_award_ids"], 1)
        self.assertEqual(audit["validity"]["historical_awards_on_or_after_cutoff"], 1)
        self.assertEqual(audit["validity"]["non_service_records"], 1)


if __name__ == "__main__":
    unittest.main()
