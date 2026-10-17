import csv
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from tesis_recommender.validation import (publication, publication_day, version_hint, normalized_url,
                                         materialize_validated_cohort, write_csv, sha256, evaluate_annotation, verify_experiment_inputs, prepare_annotation)
from tesis_recommender.corpus import _initial_tender_reference


class VersionValidationTests(unittest.TestCase):
    def test_event_date_cannot_replace_publication(self):
        self.assertIsNone(publication({"date": "2022-01-01T00:00:00Z"}))
        self.assertEqual(publication({"date": "2022-01-01T00:00:00Z", "publishedDate": "2025-06-25T16:00:00-05:00"}).year, 2025)
        self.assertIsNone(publication({"publishedDate": "2024-01-01"}))
        self.assertEqual(publication_day({"publishedDate": "2024-12-31T23:30:00-05:00"}), date(2024, 12, 31))

    def test_reference_order_uses_version_hint_not_backdated_event(self):
        base = "https://contratacionesabiertas.oece.gob.pe/api/v1/release/x-"
        refs = [{"url": base + "2025-06-01T00:00:00Z", "date": "2022-01-01T00:00:00Z", "tag": ["tender"]},
                {"url": base + "2023-12-06T00:00:00Z", "date": "2022-01-01T00:00:00Z", "tag": ["tender"]}]
        self.assertEqual(_initial_tender_reference({"releases": refs})[0], refs[1]["url"])
        self.assertEqual(version_hint(refs[1]).year, 2023)
        with self.assertRaises(ValueError):
            normalized_url("https://other.example/api/v1/release/x")

    def test_materialization_requires_pre_cutoff_award_evidence_and_preserves_currency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort, audit, output = root / "cohort", root / "audit", root / "output"
            cohort.mkdir(); audit.mkdir()
            historical, queue, results = [], [], []
            for number, published in [(1, "2024-02-01T00:00:00Z"), (2, "2024-03-01T00:00:00Z"), (3, "2025-06-01T00:00:00Z")]:
                ocid = f"h{number}"
                award_id = f"{ocid}:a:s"
                release = {"ocid": ocid, "id": ocid, "publishedDate": published, "tag": ["tender", "award"],
                           "tender": {"mainProcurementCategory": "services", "datePublished": "2024-01-01T00:00:00Z", "title": "Servicio", "value": {"amount": 100, "currency": "PEN"}},
                           "awards": [{"id": "a", "date": "2024-01-20T00:00:00Z", "value": {"amount": 100, "currency": "PEN"}, "suppliers": [{"id": "s"}]}]}
                path = audit / f"{ocid}.json"
                path.write_text(json.dumps({"releases": [release]}))
                queue.append({"url": ocid})
                results.append({"url": ocid, "ocid": ocid, "status": "ok", "cache_path": str(path), "sha256": sha256(path)})
                historical.append({"award_id": award_id, "supplier_id": "s", "award_date": "2024-01-20", "source_url": f"https://example/record/{ocid}"})
            release = {"ocid": "o", "publishedDate": "2025-02-01T00:00:00Z", "tag": ["tender"],
                       "tender": {"mainProcurementCategory": "services", "datePublished": "2025-02-01T00:00:00Z", "title": "Servicio", "value": {"amount": 90, "currency": "PEN"}}}
            path = audit / "o.json"
            path.write_text(json.dumps({"releases": [release]}))
            queue.append({"url": "o"})
            results.append({"url": "o", "ocid": "o", "status": "ok", "cache_path": str(path), "sha256": sha256(path)})
            write_csv(audit / "version_queue.csv", queue, ["url"])
            write_csv(audit / "version_results.csv", results, list(results[0]))
            write_csv(cohort / "cohort_historical_awards.csv", historical, list(historical[0]))
            write_csv(cohort / "cohort_test_opportunities_compiled.csv", [{"opportunity_id": "o"}], ["opportunity_id"])
            write_csv(cohort / "cohort_labels.csv", [{"supplier_id": "s", "opportunity_id": "o", "label": 1, "award_date": "2025-03-01"}], ["supplier_id", "opportunity_id", "label", "award_date"])
            report = materialize_validated_cohort(cohort, audit, output)
            self.assertEqual(report["retained_profiles_ge_2"], 1)
            self.assertEqual(report["retained_historical_awards"], 2)
            self.assertEqual(report["award_statuses"]["no_pre_cutoff_text_version"], 1)
            verify_experiment_inputs(output / "validation_report.json", output / "historical_awards.csv", output / "test_opportunities.csv", output / "labels.csv")
            with (output / "historical_awards.csv").open() as handle:
                self.assertTrue(all(row["currency"] == "PEN" for row in csv.DictReader(handle)))
            # Alterar evidencia invalida la reproducción antes de usar los datos.
            path.write_text("{}")
            with self.assertRaises(ValueError):
                materialize_validated_cohort(cohort, audit, output)
            (output / "labels.csv").write_text("supplier_id,opportunity_id,label\n")
            with self.assertRaises(ValueError):
                verify_experiment_inputs(output / "validation_report.json", output / "historical_awards.csv", output / "test_opportunities.csv", output / "labels.csv")

    def test_annotation_does_not_invent_truth_and_counts_wrong_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "annotation.csv"
            rows = [{"opportunity_id": str(i), "field": "amount", "prediction": prediction,
                     "gold_value": gold, "gold_state": "observed", "annotator_id": "reviewer",
                     "reviewed_at": "2026-10-08", "evidence_fragment": "JSON comprobado"}
                    for i, (prediction, gold) in enumerate([("100.0", "100"), ("50", "80")])]
            write_csv(path, rows, list(rows[0]))
            metrics = evaluate_annotation(path, root / "metrics")["metrics_by_field"]["amount"]
            self.assertEqual(metrics["precision"], 0.5)
            self.assertEqual(metrics["recall"], 0.5)
            rows[0]["gold_state"] = ""
            write_csv(path, rows, list(rows[0]))
            with self.assertRaises(ValueError):
                evaluate_annotation(path, root / "metrics")

    def test_annotation_generator_preserves_human_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "sample"
            output.mkdir()
            path = output / "field_annotations.csv"
            write_csv(path, [{"gold_value": "Valor revisado"}], ["gold_value"])
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                prepare_annotation(root / "cohort", root / "audit", output)
            self.assertEqual(path.read_bytes(), original)
