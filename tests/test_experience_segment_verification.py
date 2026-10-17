import copy
import unittest

from scripts.compare_experience_segments import check_cases, validate_result
from tesis_recommender.experience_segments import extract_experience_segments


class ExperienceSegmentVerificationTests(unittest.TestCase):
    def setUp(self):
        self.pages = [{"page": 1, "text": "REQUISITOS DE CALIFICACIÓN\nEXPERIENCIA DEL POSTOR\nEl postor debe acreditar un monto facturado acumulado equivalente a S/ 100.00\nAcreditación: contratos."}]
        self.result = extract_experience_segments(self.pages)

    def test_independent_trace_verifier_accepts_source_slices(self):
        self.assertTrue(validate_result(self.result, self.pages))

    def test_trace_verifier_rejects_tampered_span_and_anchor(self):
        for kind in ("evidence_spans", "section_anchor"):
            result = copy.deepcopy(self.result)
            row = result["mentions"][0]
            target = row[kind][0] if kind == "evidence_spans" else row[kind]
            target["evidence"] = "changed"
            with self.assertRaises(ValueError):
                validate_result(result, self.pages)

    def test_reference_checker_detects_wrong_expected_value(self):
        expected = {"subject": "bidder", "section_role": "qualification", "status": "explicit_clause_extracted", "billed_amount": "999.00"}
        reference = {"cases": [{"case_id": "incorrect_amount", "opportunity_id": "op", "page": 1,
                                "source": {"sha256": "pdf"}, "expected": expected}]}
        docs = [{"opportunity_id": "op", "source": {"sha256": "pdf"}, "result": self.result}]
        self.assertFalse(check_cases(reference, docs)[0]["passed"])

    def test_reference_checker_rejects_another_pdf(self):
        ref = {"cases": [{"case_id": "wrong_source", "opportunity_id": "op", "page": 1,
                          "source": {"sha256": "different"}, "expected": {"subject": "bidder"}}]}
        with self.assertRaises(ValueError):
            check_cases(ref, [{"opportunity_id": "op", "source": {"sha256": "pdf"}, "result": self.result}])
