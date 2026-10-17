import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_extraction_commitment.py"
spec = importlib.util.spec_from_file_location("extraction_commitment_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ExtractionCommitmentTests(unittest.TestCase):
    def test_object_is_description_not_procedure_identifier(self):
        release = {"tender": {"title": "AS-001-2025", "description": "Servicio de limpieza"}}
        self.assertEqual(audit.source_reference(release, "object"), ("Servicio de limpieza", "tender.description"))

    def test_category_first_item_and_fallback(self):
        release = {"tender": {"mainProcurementCategory": "services", "items": [{"classification": {"id": "123"}}]}}
        self.assertEqual(audit.source_reference(release, "category"), ("123", "tender.items[0].classification.id"))
        self.assertEqual(audit.source_reference({"tender": {"mainProcurementCategory": "services"}}, "category")[0], "services")

    def test_zero_is_observed_not_missing(self):
        row = {"field": "amount", "prediction": "0.0", "reference_value": 0, "reference_state": "observed", "status": "match"}
        result = audit.score_field([row])
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["f1"], 1)

    def test_undefined_metric_is_not_perfect_when_all_absent(self):
        row = {"field": "object", "prediction": "", "reference_value": "", "reference_state": "absent", "status": "match"}
        result = audit.score_field([row])
        self.assertEqual(result["tn"], 1)
        self.assertIsNone(result["f1"])

    def test_wrong_value_counts_fp_and_fn(self):
        row = {"field": "object", "prediction": "Limpieza", "reference_value": "Vigilancia", "reference_state": "observed", "status": "mismatch"}
        result = audit.score_field([row])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))
        self.assertEqual(result["f1"], 0)

    def test_region_conflict_not_silently_selected(self):
        release = {"buyer": {"id": "A"}, "tender": {"procuringEntity": {"id": "B"}},
                   "parties": [{"id": "A", "address": {"department": "Lima"}}, {"id": "B", "address": {"department": "Cusco"}}]}
        with self.assertRaises(ValueError):
            audit.source_reference(release, "region")

    def test_experience_mentions_are_not_inferred_from_documents_or_outcomes(self):
        release = {"tender": {"description": "Servicio de mantenimiento", "documents": [{"title": "Experiencia requerida"}]},
                   "awards": [{"description": "Proveedor con experiencia"}]}
        self.assertEqual(audit.source_reference(release, "experience")[0], [])
        release["tender"]["description"] = "Experiencia del postor: dos años"
        self.assertEqual(audit.source_reference(release, "experience")[0][0]["path"], "tender.description")

    def test_non_evaluable_experience_not_true_negative_or_false_metric(self):
        row = {"field": "experience", "prediction": "", "reference_value": [], "reference_state": "not_identified_in_available_text", "status": "not_evaluable"}
        result = audit.score_field([row])
        self.assertEqual((result["evaluated"], result["non_evaluable"], result["tn"]), (0, 1, 0))
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["f1"])

    def test_invalid_amount_rejected(self):
        for value in ("NaN", "Infinity", "no disponible"):
            with self.assertRaises(ValueError):
                audit.norm(value, "amount")


if __name__ == "__main__":
    unittest.main()
