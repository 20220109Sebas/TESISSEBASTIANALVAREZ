import unittest

from tesis_recommender.document_coverage import coverage_rows, coverage_summary


def queue():
    return [{"opportunity_id": "op", "url": "url", "sample_number": 1, "title": "Bases", "format": "pdf"}]


def document(text="usable", method="native_pdf_text", qualified=False):
    return {"opportunity_id": "op", "source": {"url": "url", "file_path": "pdf", "sha256": "hash"},
            "pages": [{"text": text, "extraction_method": method, "ocr_used": method == "apple_vision_local_ocr"}],
            "total_pages": 1, "truncated_by_page_limit": False,
            "result": {"qualification_clauses": [{}] if qualified else [], "evaluation_factor_clauses": [], "mentions": []}}


class DocumentCoverageTests(unittest.TestCase):
    def test_unattempted_is_unknown_not_negative(self):
        row = coverage_rows(queue(), [], [])[0]
        self.assertEqual(row["download_state"], "unattempted")
        self.assertFalse(row["absence_of_requirement_proven"])
        self.assertFalse(coverage_summary([row])["negative_labels_created"])

    def test_partial_pdf_is_not_complete_or_inspected(self):
        row = coverage_rows(queue(), [{"url": "url", "status": "too_large_partial_file"}], [])[0]
        self.assertFalse(row["complete_pdf_available"])
        self.assertEqual(row["text_state"], "not_inspected")

    def test_successful_retry_does_not_inflate_opportunity_count(self):
        rows = coverage_rows(queue(), [{"url": "url", "status": "access_error"}, {"url": "url", "status": "downloaded_pdf"}], [document()])
        self.assertEqual(coverage_summary(rows)["complete_pdf_opportunities"], 1)
        self.assertEqual(rows[0]["access_attempts_recorded"], 2)

    def test_empty_text_is_not_absence_of_requirement(self):
        row = coverage_rows(queue(), [{"url": "url", "status": "downloaded_pdf"}], [document("", "no_usable_text")])[0]
        self.assertEqual(row["experience_state"], "not_evaluable_no_usable_text")
        self.assertFalse(row["absence_of_requirement_proven"])

    def test_saved_ocr_is_identified_and_not_semantically_validated(self):
        row = coverage_rows(queue(), [{"url": "url", "status": "downloaded_pdf"}], [document(method="apple_vision_local_ocr", qualified=True)])[0]
        self.assertEqual(row["text_state"], "saved_ocr_only")
        self.assertEqual(row["qualification_candidates"], 1)
        self.assertFalse(row["semantic_extraction_validated"])
        self.assertFalse(row["ranking_eligible"])

    def test_duplicate_queue_or_source_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            coverage_rows(queue() * 2, [], [])
        d = document()
        d["source"]["url"] = "wrong"
        with self.assertRaises(ValueError):
            coverage_rows(queue(), [{"url": "url", "status": "downloaded_pdf"}], [d])

    def test_inspected_partial_file_is_rejected(self):
        with self.assertRaises(ValueError):
            coverage_rows(queue(), [{"url": "url", "status": "too_large_partial_file"}], [document()])
