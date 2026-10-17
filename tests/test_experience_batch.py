import unittest

from scripts.extend_experience_document_evidence import summarize


def document(oid, method="native_pdf_text", text="usable", truncated=False):
    return {"opportunity_id": oid, "pages": [{"text": text, "extraction_method": method,
                                             "ocr_used": method == "apple_vision_local_ocr"}],
            "truncated_by_page_limit": truncated,
            "result": {"qualification_clauses": [], "evaluation_factor_clauses": []}}


class ExperienceBatchTests(unittest.TestCase):
    def test_existing_ocr_is_separate_from_new_native_pages(self):
        r = summarize([document("old", "apple_vision_local_ocr"), document("new")], {"new"}, 50)
        self.assertEqual(r["previous_ocr_pages_retained"], 1)
        self.assertEqual(r["native_pages_with_usable_text"], 1)
        self.assertEqual(r["new_pages_processed"], 1)
        self.assertFalse(r["new_ocr_performed"])

    def test_no_text_is_not_absence_of_experience_or_an_f1_match(self):
        r = summarize([document("empty", "no_usable_text", "")], {"empty"}, 50)
        self.assertEqual(r["documents_with_no_usable_text"], 1)
        self.assertEqual(r["pages_without_usable_text"], 1)
        self.assertFalse(r["f1_estimable"])

    def test_coverage_is_by_document_not_clause_count(self):
        d = document("new")
        d["result"]["qualification_clauses"] = [{}, {}]
        r = summarize([d], {"new"}, 50)
        self.assertEqual(r["complete_pdf_opportunities"], 1)
        self.assertEqual(r["opportunities_without_complete_pdf"], 49)
        self.assertEqual(r["new_qualification_clauses"], 2)

    def test_page_limit_truncation_is_reported(self):
        r = summarize([document("new", truncated=True)], {"new"}, 50)
        self.assertEqual(r["page_limit_truncated_documents"], 1)
        self.assertFalse(r["historical_content_proven"])
        self.assertFalse(r["rankings_changed"])
