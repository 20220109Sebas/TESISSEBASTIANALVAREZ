import hashlib
import unittest

from tesis_recommender.experience import extract_experience, parse_money
from scripts.audit_experience_sources import before_initial, eligible_url


CLAUSE = """C EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD
Requisitos:
El postor debe acreditar un monto facturado acumulado equivalente a S/ 40,000.00
por servicios similares, durante los diez (10) años anteriores a la presentación de ofertas.
Acreditación: contratos y comprobantes de pago."""


class ExperienceTests(unittest.TestCase):
    def test_explicit_clause_extracts_value_window_and_exact_evidence(self):
        text = "3.2 REQUISITOS DE CALIFICACIÓN\n" + CLAUSE
        result = extract_experience([{"page": 31, "text": text}])
        row = result["qualification_clauses"][0]
        self.assertEqual((row["minimum_billed_amount"], row["currency"], row["lookback_years"]), ("40000.00", "PEN", 10))
        self.assertEqual(row["evidence"], text[row["char_start"]:row["char_end"]])
        self.assertEqual(row["evidence_sha256"], hashlib.sha256(row["evidence"].encode()).hexdigest())
        self.assertFalse(row["ranking_eligible"])
        self.assertFalse(result["semantic_extraction_validated"])

    def test_personnel_years_not_company_requirement(self):
        text = "REQUISITOS DE CALIFICACIÓN\nEXPERIENCIA DEL PERSONAL CLAVE\nNo menor a cinco (05) años."
        r = extract_experience([{"page": 1, "text": text}])
        self.assertEqual(r["qualification_clauses"], [])
        self.assertEqual(r["mentions"][0]["subject"], "key_personnel")

    def test_annex_form_not_requirement_even_with_clause_text(self):
        r = extract_experience([{"page": 1, "text": "ANEXO Nº 11\n" + CLAUSE}])
        self.assertEqual(r["qualification_clauses"], [])
        self.assertEqual(r["mentions"][0]["status"], "annex_form_not_requirement")

    def test_evaluation_factor_separate_from_qualification(self):
        r = extract_experience([{"page": 1, "text": "REQUISITOS DE CALIFICACIÓN\n" + CLAUSE},
                                {"page": 2, "text": "FACTORES DE EVALUACIÓN\n" + CLAUSE.replace("40,000.00", "60,000.00")}])
        self.assertEqual(r["qualification_clauses"][0]["minimum_billed_amount"], "40000.00")
        self.assertEqual(r["evaluation_factor_clauses"][0]["minimum_billed_amount"], "60000.00")

    def test_unknown_section_abstains(self):
        r = extract_experience([{"page": 1, "text": CLAUSE}])
        self.assertFalse(r["qualification_clauses"])
        self.assertEqual(r["mentions"][0]["status"], "explicit_clause_with_unknown_section")

    def test_placeholder_abstains(self):
        r = extract_experience([{"page": 1, "text": "REQUISITOS DE CALIFICACIÓN\n" + CLAUSE.replace("S/ 40,000.00", "[CONSIGNAR MONTO]")}])
        self.assertFalse(r["qualification_clauses"])
        self.assertEqual(r["mentions"][0]["status"], "unresolved_or_template_threshold")

    def test_generic_ux_word_not_supplier_requirement(self):
        r = extract_experience([{"page": 1, "text": "Servicio de estudio de experiencia de usuario. Experiencias turísticas."}])
        self.assertFalse(r["qualification_clauses"])
        self.assertFalse(r["mentions"])
        self.assertFalse(r["absence_of_experience_proven"])

    def test_empty_text_requires_ocr_not_absence(self):
        r = extract_experience([{"page": 1, "text": ""}, {"page": 2, "text": "\n"}])
        self.assertEqual(r["status"], "requires_ocr_or_other_text_source")
        self.assertFalse(r["absence_of_experience_proven"])

    def test_currency_is_explicit_and_ambiguous_numbers_abstain(self):
        text = "REQUISITOS DE CALIFICACIÓN\n" + CLAUSE.replace("S/ 40,000.00", "US$ 40,000.00")
        self.assertEqual(extract_experience([{"page": 1, "text": text}])["qualification_clauses"][0]["currency"], "USD")
        for number in ("40.000", "0", "NaN", "4,00,000", "-40"):
            self.assertIsNone(parse_money(number))
        self.assertEqual(parse_money("40.000,00"), "40000.00")

    def test_duplicate_pages_rejected(self):
        with self.assertRaises(ValueError):
            extract_experience([{"page": 1, "text": CLAUSE}, {"page": 1, "text": CLAUSE}])

    def test_declared_dates_are_not_assumed_when_missing_or_naive(self):
        initial = "2025-03-01T08:00:00-05:00"
        self.assertEqual(before_initial("2025-03-01T13:00:00+00:00", initial), "declared_at_or_before_initial")
        self.assertEqual(before_initial("2025-03-02T08:00:00-05:00", initial), "declared_after_initial")
        self.assertEqual(before_initial(None, initial), "missing_document_date")
        self.assertEqual(before_initial("2025-03-01", initial), "date_without_timezone")

    def test_download_url_does_not_allow_other_hosts_or_plain_http(self):
        valid = "https://prod1.seace.gob.pe/SeaceWeb-PRO/SdescargarArchivoAlfresco?fileCode=a"
        self.assertTrue(eligible_url(valid))
        self.assertFalse(eligible_url(valid.replace("https:", "http:")))
        self.assertFalse(eligible_url(valid.replace("prod1.seace.gob.pe", "example.com")))


if __name__ == "__main__":
    unittest.main()
