import unittest

from scripts.review_experience_formulations import evidence_span


class ExperienceFormulationReviewTests(unittest.TestCase):
    def test_literal_evidence_retains_source_offsets(self):
        text = "Antes. Experiencia no menor de tres (03) servicios. Después."
        r = evidence_span(text, "tres (03)")
        self.assertEqual(text[r["char_start"]:r["char_end"]], r["evidence"])

    def test_missing_or_repeated_literal_is_rejected(self):
        for text in ("otro texto", "tres (03) y tres (03)"):
            with self.assertRaises(ValueError):
                evidence_span(text, "tres (03)")
