import copy
import unittest

from tesis_recommender.experience_conditions import digest
from tesis_recommender.experience_formulations import extract_formulations
from tesis_recommender.experience_segments import extract_experience_segments

SUPPLIER = """6. PERFIL MÍNIMO REQUERIDO DEL PROVEEDOR
Experiencia no menor de tres (03) servicios logísticos.
El proveedor deberá acreditar como mínimo un monto facturado acumulado
equivalente al 20% del valor ofertado en el servicio.
El proveedor debe contar con personal: coordinador.
Experiencia no menor de cinco (05) servicios.
"""
BRACKET = """REQUISITOS DE CALIFICACIÓN
B. EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD
El postor debe acreditar un monto facturado acumulado equivalente a [ S/ 221,857.81
(DOSCIENTOS MIL SOLES)], durante los veinte (20) años anteriores.
Acreditación: contratos.
"""


def extract(text):
    pages = [{"page": 1, "text": text, "text_sha256": digest(text), "extraction_method": "native"}]
    mentions = extract_experience_segments(pages)["mentions"]
    return extract_formulations(pages, mentions)


class FormulationTests(unittest.TestCase):
    def test_count_and_relative_requirement_remain_different_variables(self):
        rows = extract(SUPPLIER)["records"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["minimum_services"], 3)
        self.assertEqual(rows[1]["percentage"], "20")
        self.assertIsNone(rows[1]["absolute_amount"])
        self.assertIsNone(rows[1]["currency"])
        self.assertEqual(rows[1]["reference_basis"], "offered_value_not_procurement_budget")

    def test_personnel_and_accreditation_end_supplier_segment(self):
        for boundary in ["personal", "Acreditación:", "Dicha experiencia deberá ser acreditada"]:
            text = SUPPLIER.replace("El proveedor debe contar con personal: coordinador.", boundary)
            self.assertEqual(len(extract(text)["records"]), 2)

    def test_inline_heading_reference_is_not_provider_section(self):
        self.assertEqual(extract(SUPPLIER.replace("6. PERFIL MÍNIMO REQUERIDO DEL PROVEEDOR", "Ver el PERFIL MÍNIMO REQUERIDO DEL PROVEEDOR"))["records"], [])

    def test_next_numbered_heading_ends_provider_section(self):
        text = "6. PERFIL MÍNIMO REQUERIDO DEL PROVEEDOR\n7. PERSONAL REQUERIDO\n" + SUPPLIER.split("\n", 1)[1]
        self.assertEqual(extract(text)["records"], [])

    def test_annex_provider_heading_is_not_requirement(self):
        self.assertEqual(extract("ANEXO Nº 4\n" + SUPPLIER)["records"], [])

    def test_factor_parent_of_supplier_profile_is_preserved(self):
        rows = extract("FACTORES DE EVALUACIÓN\n" + SUPPLIER)["records"]
        self.assertTrue(all(r["parent_section_role"] == "evaluation_factor" for r in rows))

    def test_word_and_digit_disagreement_is_not_repaired(self):
        r = extract(SUPPLIER.replace("tres (03)", "tres (05)"))
        self.assertEqual(len(r["records"]), 1)
        self.assertEqual(r["warnings"][0]["reason"], "unresolved_service_count")

    def test_zero_count_and_unknown_word_are_rejected(self):
        for token in ["cero (00)", "varios (03)"]:
            self.assertEqual(len(extract(SUPPLIER.replace("tres (03)", token))["records"]), 1)

    def test_digits_only_count(self):
        self.assertEqual(extract(SUPPLIER.replace("tres (03)", "3"))["records"][0]["minimum_services"], 3)

    def test_decimal_percentage_preserves_source(self):
        r = extract(SUPPLIER.replace("20%", "12,50 %"))["records"][1]
        self.assertEqual(r["percentage"], "12.50")
        self.assertIn("12,50 %", r["evidence"]["evidence"])

    def test_out_of_scope_percentage_abstains(self):
        for value in ["0%", "120%"]:
            self.assertEqual(len(extract(SUPPLIER.replace("20%", value))["records"]), 1)

    def test_procurement_budget_is_not_offered_value(self):
        r = extract(SUPPLIER.replace("valor ofertado", "valor estimado"))
        self.assertEqual(len(r["records"]), 1)

    def test_bracketed_amount_keeps_delimiter_window_and_section(self):
        r = extract(BRACKET)["records"][0]
        self.assertEqual((r["billed_amount"], r["currency"], r["lookback_years"]), ("221857.81", "PEN", 20))
        self.assertIn("[", r["delimiter_evidence"]["evidence"])
        self.assertEqual(r["section_role"], "qualification")
        self.assertFalse(r["single_universal_threshold_supported"])

    def test_amount_requires_closing_bracket(self):
        self.assertEqual(extract(BRACKET.replace("]", ""))["records"], [])

    def test_ambiguous_zero_and_template_money_abstain(self):
        for value in ["221.857", "0.00", "CONSIGNAR MONTO"]:
            self.assertEqual(extract(BRACKET.replace("221,857.81", value))["records"], [])

    def test_annex_personnel_and_unknown_section_do_not_supply_amount(self):
        for text in [BRACKET.replace("REQUISITOS DE CALIFICACIÓN", "ANEXO Nº 1"),
                     BRACKET.replace("EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD", "EXPERIENCIA DEL PERSONAL CLAVE"),
                     BRACKET.replace("REQUISITOS DE CALIFICACIÓN", "OTRO")]:
            self.assertEqual(extract(text)["records"], [])

    def test_factor_is_not_promoted_to_qualification(self):
        self.assertEqual(extract(BRACKET.replace("REQUISITOS DE CALIFICACIÓN", "FACTORES DE EVALUACIÓN"))["records"][0]["section_role"], "evaluation_factor")

    def test_years_not_borrowed_from_micro_small_paragraph(self):
        text = BRACKET.replace("durante los veinte (20) años anteriores", "antes")
        text = text.replace("Acreditación:", "En el caso de postores que declaren en el Anexo Nº 1 tener la condición de micro y pequeña empresa, se acredita una experiencia de S/ 100.00 durante los ocho (8) años anteriores.\nAcreditación:")
        self.assertIsNone(extract(text)["records"][0]["lookback_years"])

    def test_empty_text_is_not_evidence_of_absence(self):
        r = extract("")
        self.assertEqual(r["records"], [])
        self.assertFalse(r["absence_of_experience_proven"])

    def test_no_company_eligibility_ranking_or_validation_claim(self):
        for r in extract(SUPPLIER + BRACKET)["records"]:
            for k in ["ranking_eligible", "independently_validated", "historical_content_proven", "company_capacity_determined", "legal_eligibility_determined"]:
                self.assertFalse(r[k])

    def test_page_hash_and_logical_clause_tampering_rejected(self):
        pages = [{"page": 1, "text": BRACKET, "text_sha256": digest(BRACKET)}]
        mentions = extract_experience_segments(pages)["mentions"]
        bad = copy.deepcopy(pages);bad[0]["text"] += "otro"
        with self.assertRaises(ValueError): extract_formulations(bad, mentions)
        bad = copy.deepcopy(mentions);bad[0]["evidence"] += "otro"
        with self.assertRaises(ValueError): extract_formulations(pages, bad)

    def test_preserves_inputs_and_source_offsets(self):
        pages = [{"page": 1, "text": SUPPLIER, "text_sha256": digest(SUPPLIER)}]
        original = copy.deepcopy(pages)
        for row in extract_formulations(pages, [])["records"]:
            for s in row["evidence"]["source_spans"]:
                self.assertEqual(SUPPLIER[s["char_start"]:s["char_end"]], s["evidence"])
        self.assertEqual(pages, original)
