import hashlib
import unittest

from tesis_recommender.experience_segments import extract_experience_segments as extract

CLAUSE = """C EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD
Requisitos:
El postor debe acreditar un monto facturado acumulado equivalente a S/ 40,000.00
por servicios similares, durante los diez (10) años anteriores a la presentación de ofertas.
Acreditación: contratos y comprobantes de pago."""


class ExperienceSegmentTests(unittest.TestCase):
    def test_whole_line_headings_preserve_value(self):
        result = extract([{"page": 1, "text": "3.2 REQUISITOS DE CALIFICACIÓN\n" + CLAUSE}])
        row = result["qualification_clauses"][0]
        self.assertEqual((row["billed_amount"], row["currency"], row["lookback_years"]), ("40000.00", "PEN", 10))
        self.assertEqual(row["segment_stop"], "accreditation")
        self.assertFalse(result["semantic_extraction_validated"])

    def test_inline_annex_reference_does_not_change_parent_section(self):
        text = "REQUISITOS DE CALIFICACIÓN\nAnexo N° 11 referido a la experiencia.\n" + CLAUSE
        self.assertEqual(len(extract([{"page": 1, "text": text}])["qualification_clauses"]), 1)

    def test_inline_mentions_are_not_headings(self):
        text = "Las disposiciones sobre Experiencia del postor son importantes.\nLos requisitos de calificación se describen después."
        self.assertEqual(extract([{"page": 1, "text": text}])["mentions"], [])

    def test_annex_form_is_not_requirement(self):
        r = extract([{"page": 1, "text": "ANEXO Nº 11\n" + CLAUSE}])
        self.assertEqual(r["qualification_clauses"], [])
        self.assertEqual(r["mentions"][0]["status"], "annex_form_not_requirement")

    def test_adjacent_pages_keep_exact_spans(self):
        pages = [{"page": 1, "text": "REQUISITOS DE CALIFICACIÓN\n" + CLAUSE.split("S/ ")[0]},
                 {"page": 2, "text": "S/ " + CLAUSE.split("S/ ")[1]}]
        row = extract(pages)["qualification_clauses"][0]
        self.assertEqual(len(row["evidence_spans"]), 2)
        for span in row["evidence_spans"]:
            p = next(p for p in pages if p["page"] == span["page"])
            self.assertEqual(span["evidence"], p["text"][span["char_start"]:span["char_end"]])
        self.assertEqual(row["evidence_sha256"], hashlib.sha256(row["evidence"].encode()).hexdigest())

    def test_missing_page_cannot_supply_threshold(self):
        r = extract([{"page": 1, "text": "REQUISITOS DE CALIFICACIÓN\n" + CLAUSE.split("S/ ")[0]},
                     {"page": 3, "text": "S/ " + CLAUSE.split("S/ ")[1]}])
        self.assertEqual(r["qualification_clauses"], [])
        self.assertEqual(r["mentions"][0]["segment_stop"], "non_adjacent_page")

    def test_new_section_stops_cross_page_clause(self):
        r = extract([{"page": 1, "text": "REQUISITOS DE CALIFICACIÓN\n" + CLAUSE.split("S/ ")[0]},
                     {"page": 2, "text": "FACTORES DE EVALUACIÓN\n" + CLAUSE}])
        self.assertEqual(r["qualification_clauses"], [])
        self.assertEqual(len(r["evaluation_factor_clauses"]), 1)

    def test_factor_table_heading_stays_separate(self):
        text = "FACTORES DE EVALUACIÓN\n" + CLAUSE.replace("C EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD", "A. EXPERIENCIA DEL POSTOR EN LA ESPECIALIDAD 100puntos")
        r = extract([{"page": 1, "text": text}])
        self.assertEqual(r["qualification_clauses"], [])
        self.assertEqual(len(r["evaluation_factor_clauses"]), 1)

    def test_personnel_not_bidder(self):
        r = extract([{"page": 1, "text": "REQUISITOS DE CALIFICACIÓN\nB.1 EXPERIENCIA DEL PERSONAL CLAVE\nCinco años."}])
        self.assertEqual(r["mentions"][0]["subject"], "key_personnel")
        self.assertEqual(r["qualification_clauses"], [])

    def test_template_and_corrupt_ocr_abstain(self):
        for clause in (CLAUSE.replace("S/ 40,000.00", "[CONSIGNAR MONTO]"), CLAUSE.replace("El postor", "-l postor")):
            self.assertEqual(extract([{"page": 1, "text": "REQUISITOS DE CALIFICACIÓN\n" + clause}])["qualification_clauses"], [])

    def test_mismatched_hash_duplicate_and_empty_inputs_rejected(self):
        for pages in ([], [{"page": 1, "text": CLAUSE, "text_sha256": "wrong"}], [{"page": 1, "text": ""}, {"page": 1, "text": ""}]):
            with self.assertRaises(ValueError):
                extract(pages)

    def test_unknown_chapter_resets_section_and_never_proves_absence(self):
        r = extract([{"page": 1, "text": "REQUISITOS DE CALIFICACIÓN\nCAPÍTULO V\n" + CLAUSE}])
        self.assertEqual(r["qualification_clauses"], [])
        self.assertFalse(r["absence_of_experience_proven"])

    def test_character_and_page_limits_are_explicit(self):
        r = extract([{"page": 1, "text": "REQUISITOS DE CALIFICACIÓN\n" + CLAUSE}], max_chars=100)
        self.assertEqual(r["qualification_clauses"], [])
        self.assertEqual(r["mentions"][0]["segment_stop"], "character_limit")
        with self.assertRaises(ValueError):
            extract([{"page": 1, "text": CLAUSE}], max_pages=3)

    def test_ocr_provenance_is_preserved_per_span(self):
        r = extract([{"page": 1, "text": "REQUISITOS DE CALIFICACIÓN\n" + CLAUSE,
                      "ocr_used": True, "extraction_method": "apple_vision_local_ocr",
                      "ocr_json_path": "saved/page.json", "ocr_json_sha256": "hash"}])
        span = r["qualification_clauses"][0]["evidence_spans"][0]
        self.assertEqual(span["ocr_json_path"], "saved/page.json")
        self.assertFalse(r["qualification_clauses"][0]["ranking_eligible"])
