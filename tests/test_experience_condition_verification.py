import copy
import hashlib
import unittest

from scripts.represent_experience_conditions import verify_variable
from tesis_recommender.experience_conditions import represent_experience_clause
from tesis_recommender.experience_segments import extract_experience_segments


class ExperienceConditionVerificationTests(unittest.TestCase):
    def setUp(self):
        text = "REQUISITOS DE CALIFICACIÓN\nEXPERIENCIA DEL POSTOR\nEl postor debe acreditar un monto facturado acumulado equivalente a S/ 100.00 durante los cinco (5) años anteriores.\nAcreditación:"
        self.pages = [{"page": 1, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}]
        self.clause = extract_experience_segments(self.pages)["qualification_clauses"][0]
        self.variable = represent_experience_clause(self.clause)

    def test_verifier_accepts_exact_source_slices(self):
        self.assertTrue(verify_variable(self.variable, self.clause, self.pages))

    def test_verifier_rejects_tampered_amount_span(self):
        v = copy.deepcopy(self.variable)
        v["branches"][0]["amount_evidence"]["source_spans"][0]["evidence"] = "S/ 999.00"
        with self.assertRaises(ValueError):
            verify_variable(v, self.clause, self.pages)

    def test_verifier_rejects_company_branch_selection(self):
        v = copy.deepcopy(self.variable)
        v["selected_branch"] = "primary_paragraph"
        with self.assertRaises(ValueError):
            verify_variable(v, self.clause, self.pages)

    def test_verifier_rejects_altered_logical_fragment(self):
        v = copy.deepcopy(self.variable)
        v["branches"][0]["window_evidence"]["evidence"] = "twenty years"
        with self.assertRaises(ValueError):
            verify_variable(v, self.clause, self.pages)

    def test_verifier_rejects_altered_normalized_amount_or_currency(self):
        for key, value in (("billed_amount", "999.00"), ("currency", "USD")):
            v = copy.deepcopy(self.variable)
            v["branches"][0][key] = value
            with self.assertRaises(ValueError):
                verify_variable(v, self.clause, self.pages)

    def test_verifier_rejects_altered_normalized_window(self):
        v = copy.deepcopy(self.variable)
        v["branches"][0]["lookback_years"] = 20
        with self.assertRaises(ValueError):
            verify_variable(v, self.clause, self.pages)
