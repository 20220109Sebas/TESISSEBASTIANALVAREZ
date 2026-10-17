import hashlib
import unittest
from tesis_recommender.annex_sources import document_role, safe_annex_url, select_annexes
from tesis_recommender.ocr_evidence import merge_page, bind_clause_method


URL = "https://prod1.seace.gob.pe/SeaceWeb-PRO/SdescargarArchivoAlfresco?fileCode=3355b281-e4c0-41ed-a3fe-ec23f3f76b80"


class AnnexOCRTests(unittest.TestCase):
    def test_reject_proposals_even_if_bidding_documents(self):
        self.assertEqual(document_role("Documentos de Presentación de Propuestas", "biddingDocuments"), "excluded_non_requirement_title")
        self.assertEqual(document_role("Resumen ejecutivo", "biddingDocuments"), "excluded_non_requirement_title")
        self.assertEqual(document_role("Documentos de Calificación y Evaluación", "evaluationReports"), "excluded_type")

    def test_titles_and_accents_recognized(self):
        self.assertEqual(document_role("TÉRMINOS DE REFERENCIA", "biddingDocuments"), "terms_of_reference")
        self.assertEqual(document_role(" Bases Administrativas ", "biddingDocuments"), "administrative_bases")
        self.assertEqual(document_role("Sin título", "biddingDocuments"), "unknown_title_not_selected")

    def test_url_rejects_private_ports_extra_query_and_invalid_guid(self):
        self.assertTrue(safe_annex_url(URL))
        for u in (URL.replace("https:", "http:"), URL.replace("prod1.seace.gob.pe", "example.com"),
                  URL + "&token=abc", URL + "#section", URL.replace(".pe/", ".pe:abc/"),
                  URL.replace("3355b281-e4c0-41ed-a3fe-ec23f3f76b80", "invalid")):
            self.assertFalse(safe_annex_url(u))

    def test_selection_prefers_initial_admin_and_one_per_opportunity(self):
        base = {"opportunity_id": "a", "sample_number": 1, "document_index": 1,
                "document_type": "biddingDocuments", "url": URL,
                "temporal_metadata_status": "declared_at_or_before_initial"}
        integrated = dict(base, title="Bases Integradas", document_index=0,
                          url=URL.replace("3355b281", "4355b281"))
        chosen, _, _ = select_annexes([integrated, dict(base, title="Bases Administrativas")])
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["document_role"], "administrative_bases")

    def test_temporal_unknown_not_silently_selected(self):
        doc = {"title": "Bases Administrativas", "document_type": "biddingDocuments",
               "url": URL, "temporal_metadata_status": "missing_document_date"}
        chosen, excluded, _ = select_annexes([doc])
        self.assertEqual(chosen, [])
        self.assertEqual(excluded[0]["selection_reason"], "after_or_unknown_initial_date")

    def native(self, text=""):
        return {"page": 1, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}

    def ocr(self):
        text = "Texto OCR"
        return {"page": 1, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "status": "ok", "ocr_json_path": "ocr.json", "ocr_json_sha256": "a" * 64, "low_confidence_lines": 1}

    def test_native_text_not_overwritten(self):
        page = merge_page(self.native("Original"), self.ocr())
        self.assertEqual(page["text"], "Original")
        self.assertFalse(page["ocr_used"])

    def test_ocr_missing_or_failed_does_not_prove_absence(self):
        for candidate in (None, {"status": "ocr_error"}):
            page = merge_page(self.native(), candidate)
            self.assertEqual(page["extraction_method"], "no_usable_text")
            self.assertFalse(page["absence_of_requirement_proven"])

    def test_ocr_hash_page_identity_and_method(self):
        page = merge_page(self.native(), self.ocr())
        self.assertTrue(page["ocr_used"])
        self.assertTrue(page["confidence_is_not_semantic_accuracy"])
        for changed in (dict(self.ocr(), page=2), dict(self.ocr(), text="Modified")):
            with self.assertRaises(ValueError):
                merge_page(self.native(), changed)

    def test_clause_bound_to_correct_text_and_ocr_source(self):
        page = merge_page(self.native(), self.ocr())
        clause = {"page": 1, "char_start": 0, "char_end": 5, "evidence": "Texto"}
        result = bind_clause_method(clause, [page])
        self.assertEqual(result["extraction_method"], "apple_vision_local_ocr")
        self.assertFalse(result["value_visually_verified"])
        self.assertFalse(result["ranking_eligible"])
        with self.assertRaises(ValueError):
            bind_clause_method(dict(clause, evidence="Wrong"), [page])


if __name__ == "__main__":
    unittest.main()
