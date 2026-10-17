import copy
import unittest

from tesis_recommender.experience_conditions import represent_experience_clause as represent
from tesis_recommender.experience_segments import extract_experience_segments

PRIMARY = """REQUISITOS DE CALIFICACIÓN
C EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD
El postor debe acreditar un monto facturado acumulado equivalente a S/ 244,875.00
por servicios similares durante los ocho (8) años anteriores a las ofertas.
"""
CONDITIONAL = """En el caso de postores que declaren en el Anexo Nº 1 tener la condición de micro y pequeña empresa,
se acredita una experiencia de S/ 20,406.25 por servicios similares durante los ocho (8) años anteriores.
En el caso de consorcios, todos los integrantes deben contar con la condición de micro y pequeña empresa.
"""


def clause(text=PRIMARY + CONDITIONAL):
    return extract_experience_segments([{"page": 1, "text": text + "\nAcreditación: contratos."}])["qualification_clauses"][0]


class ExperienceConditionTests(unittest.TestCase):
    def test_two_amounts_keep_explicit_condition_and_never_select_company_branch(self):
        r = represent(clause())
        self.assertEqual([b["billed_amount"] for b in r["branches"]], ["244875.00", "20406.25"])
        self.assertEqual([b["lookback_years"] for b in r["branches"]], [8, 8])
        self.assertEqual(r["status"], "conditional_branches_represented")
        self.assertEqual(r["branches"][1]["condition"]["type"], "explicit_declaration_micro_small_enterprise")
        self.assertIn("additional_condition", r["branches"][1])
        self.assertIsNone(r["selected_branch"])
        self.assertFalse(r["ranking_eligible"])
        self.assertFalse(r["single_universal_threshold_supported"])

    def test_single_paragraph_is_not_procurement_budget_or_company_capacity(self):
        r = represent(clause(PRIMARY))
        self.assertEqual(len(r["branches"]), 1)
        self.assertTrue(r["not_procurement_budget"])
        self.assertFalse(r["company_capacity_determined"])
        self.assertFalse(r["whole_document_conditions_complete"])

    def test_missing_window_is_not_borrowed_from_conditional_branch(self):
        c = clause(PRIMARY.replace("durante los ocho (8) años anteriores", "antes") + CONDITIONAL)
        r = represent(c)
        self.assertIsNone(r["branches"][0]["lookback_years"])
        self.assertEqual(r["branches"][1]["lookback_years"], 8)
        self.assertIn("branch_scoped_values_differ_from_v2_summary", r["warnings"])

    def test_different_windows_and_currencies_are_preserved(self):
        r = represent(clause(PRIMARY + CONDITIONAL.replace("S/ 20,406.25", "US$ 20,406.25").replace("ocho (8)", "diez (10)")))
        self.assertEqual([(b["currency"], b["lookback_years"]) for b in r["branches"]], [("PEN", 8), ("USD", 10)])

    def test_unknown_second_amount_requires_review(self):
        r = represent(clause(PRIMARY + "Otra condición requiere S/ 10,000.00."))
        self.assertEqual(len(r["branches"]), 1)
        self.assertEqual(r["status"], "requires_review_of_unresolved_segment")
        self.assertEqual(len(r["unassigned_monetary_evidence"]), 1)

    def test_ambiguous_conditional_amount_is_not_repaired(self):
        r = represent(clause(PRIMARY + CONDITIONAL.replace("20,406.25", "20.406")))
        self.assertEqual(len(r["branches"]), 1)
        self.assertIn("unresolved_or_template_amount_in_supported_branch", r["warnings"])

    def test_template_conditional_amount_is_not_invented(self):
        r = represent(clause(PRIMARY + CONDITIONAL.replace("S/ 20,406.25", "[CONSIGNAR MONTO]")))
        self.assertEqual(len(r["branches"]), 1)
        self.assertEqual(r["status"], "requires_review_of_unresolved_segment")

    def test_factor_role_is_not_changed_to_qualification(self):
        c = extract_experience_segments([{"page": 1, "text": PRIMARY.replace("REQUISITOS DE CALIFICACIÓN", "FACTORES DE EVALUACIÓN")}])["evaluation_factor_clauses"][0]
        self.assertEqual(represent(c)["section_role"], "evaluation_factor")

    def test_staff_and_annex_cannot_supply_bidder_variable(self):
        for key, value in (("subject", "key_personnel"), ("section_role", "annex_form")):
            c = clause()
            c[key] = value
            self.assertEqual(represent(c)["branches"], [])

    def test_evidence_tampering_is_rejected(self):
        c = clause()
        c["evidence"] += " changed"
        with self.assertRaises(ValueError):
            represent(c)

    def test_amount_spans_map_back_to_raw_pages(self):
        c = clause()
        r = represent(c)
        raw = c["evidence_spans"][0]
        for b in r["branches"]:
            span = b["amount_evidence"]["source_spans"][0]
            start = span["char_start"] - raw["char_start"]
            end = span["char_end"] - raw["char_start"]
            self.assertEqual(raw["evidence"][start:end], span["evidence"])

    def test_cross_page_condition_keeps_exact_evidence(self):
        text = PRIMARY + CONDITIONAL
        split = text.index("se acredita")
        c = extract_experience_segments([{"page": 1, "text": text[:split]},
                                       {"page": 2, "text": text[split:] + "\nAcreditación:"}])["qualification_clauses"][0]
        r = represent(c)
        self.assertEqual(len(r["branches"]), 2)
        self.assertEqual(len(r["branches"][1]["condition"]["evidence"]["source_spans"]), 2)

    def test_truncated_segment_has_warning(self):
        c = clause()
        c["segment_stop"] = "page_limit"
        self.assertIn("segment_boundary_may_omit_conditions", represent(c)["warnings"])

    def test_input_clause_is_not_modified(self):
        c = clause()
        before = copy.deepcopy(c)
        represent(c)
        self.assertEqual(c, before)
