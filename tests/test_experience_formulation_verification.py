import copy
import unittest

from scripts.extend_experience_formulations import verify_record
from tesis_recommender.experience_conditions import digest
from tesis_recommender.experience_formulations import extract_formulations
from tesis_recommender.experience_segments import extract_experience_segments
from tests.test_experience_formulations import SUPPLIER, BRACKET


def prepare(text):
    pages = [{"page": 1, "text": text, "text_sha256": digest(text), "extraction_method": "native"}]
    records = extract_formulations(pages, extract_experience_segments(pages)["mentions"])["records"]
    return {p["page"]: p for p in pages}, records


class FormulationVerificationTests(unittest.TestCase):
    def test_verifies_three_literal_types(self):
        for text in [SUPPLIER, BRACKET]:
            pages, rows = prepare(text)
            for row in rows:
                verify_record(row, pages)

    def test_changed_count_is_rejected(self):
        pages, rows = prepare(SUPPLIER)
        row = copy.deepcopy(rows[0]);row["minimum_services"] = 5
        with self.assertRaises(ValueError): verify_record(row, pages)

    def test_changed_percentage_is_rejected(self):
        pages, rows = prepare(SUPPLIER)
        row = copy.deepcopy(rows[1]);row["percentage"] = "30"
        with self.assertRaises(ValueError): verify_record(row, pages)

    def test_money_conversion_and_wrong_reference_basis_are_rejected(self):
        pages, rows = prepare(SUPPLIER)
        for key, value in [("absolute_amount", "20000"), ("currency", "PEN"), ("reference_basis", "budget")]:
            row = copy.deepcopy(rows[1]);row[key] = value
            with self.assertRaises(ValueError): verify_record(row, pages)

    def test_changed_amount_currency_and_years_are_rejected(self):
        pages, rows = prepare(BRACKET)
        for key, value in [("billed_amount", "100"), ("currency", "USD"), ("lookback_years", 8)]:
            row = copy.deepcopy(rows[0]);row[key] = value
            with self.assertRaises(ValueError): verify_record(row, pages)

    def test_missing_source_spans_and_changed_fragment_are_rejected(self):
        pages, rows = prepare(SUPPLIER)
        row = copy.deepcopy(rows[0]);row["evidence"]["source_spans"] = []
        with self.assertRaises(ValueError): verify_record(row, pages)
        row = copy.deepcopy(rows[0]);row["evidence"]["source_spans"][0]["evidence"] += "otro"
        with self.assertRaises(ValueError): verify_record(row, pages)

    def test_changed_logical_context_offsets_rejected(self):
        pages, rows = prepare(SUPPLIER)
        row = copy.deepcopy(rows[0]);row["evidence"]["segment_char_start"] += 1
        with self.assertRaises(ValueError): verify_record(row, pages)

    def test_inferred_eligibility_or_validation_is_rejected(self):
        pages, rows = prepare(BRACKET)
        for key in ["ranking_eligible", "independently_validated", "historical_content_proven", "company_capacity_determined", "legal_eligibility_determined", "whole_document_conditions_complete", "single_universal_threshold_supported"]:
            row = copy.deepcopy(rows[0]);row[key] = True
            with self.assertRaises(ValueError): verify_record(row, pages)

    def test_wrong_subject_section_method_and_applicability_are_rejected(self):
        pages, rows = prepare(BRACKET)
        for key, value in [("subject", "key_personnel"), ("section_role", "annex_form"), ("company_applicability", "approved")]:
            row = copy.deepcopy(rows[0]);row[key] = value
            with self.assertRaises(ValueError): verify_record(row, pages)
        row = copy.deepcopy(rows[0]);row["evidence"]["source_spans"][0]["extraction_method"] = "ocr"
        with self.assertRaises(ValueError): verify_record(row, pages)

    def test_cross_page_money_and_window_keep_raw_spans(self):
        split = BRACKET.index("221,857.81")
        pages = [{"page": 1, "text": BRACKET[:split], "extraction_method": "native"},
                 {"page": 2, "text": BRACKET[split:], "extraction_method": "native"}]
        for p in pages: p["text_sha256"] = digest(p["text"])
        rows = extract_formulations(pages, extract_experience_segments(pages)["mentions"])["records"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["evidence"]["source_spans"]), 2)
        verify_record(rows[0], {p["page"]: p for p in pages})
