import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tesis_recommender.contrast_history import verify_history_queue, fetch_contrast_history, materialize_contrast_history
from tesis_recommender.validation import sha256, read_csv, write_csv


class ContrastHistoryTests(unittest.TestCase):
    def fixture(self, root):
        selection, preparation, recovered = [root / p for p in ("selection", "preparation", "recovered")]
        for p in (selection, preparation, recovered):
            p.mkdir()
        history, queue, results = [], [], []
        for n in range(1, 5):
            ocid, supplier = f"h{n}", "p1" if n <= 2 else "p2"
            history.append({"supplier_id": supplier, "award_id": f"{ocid}:a:{supplier}",
                            "award_date": "2024-02-10", "source_url": f"https://example/record/{ocid}"})
            for tag, native in (("tender", "2025-02-01T10:00:00-05:00" if n == 4 else "2024-02-01T10:00:00-05:00"),
                                ("award", "2025-02-01T10:00:00-05:00" if n == 3 else "2024-02-15T10:00:00-05:00")):
                identity = f"{ocid}-{tag}-2024-02-15T10:00:00-05:00"
                url = f"https://contratacionesabiertas.oece.gob.pe/api/v1/release/{identity}"
                release = {"ocid": ocid, "id": identity, "tag": [tag], "date": "2024-01-01T00:00:00Z", "publishedDate": native}
                if tag == "tender":
                    release["tender"] = {"mainProcurementCategory": "services", "datePublished": "2024-01-01T00:00:00Z",
                                         "title": "Servicio histórico", "description": "Mantenimiento de equipos"}
                else:
                    # Evidencia parcial: no depende de un tender añadido por el código.
                    release["awards"] = [{"id": "a", "status": "active", "date": "2024-02-10",
                                          "value": {"amount": 100, "currency": "USD"}, "suppliers": [{"id": supplier}]}]
                path = recovered / f"{ocid}_{tag}.json"
                path.write_text(json.dumps({"releases": [release]}))
                row = {"ocid": ocid, "role": "historical_profile", "url": url, "version_hint": "2024-02-15T15:00:00Z",
                       "reference_date": "2024-01-01", "cache_key": path.stem}
                queue.append(row)
                results.append({**row, "status": "ok", "cache_path": str(path), "sha256": sha256(path)})
        write_csv(selection / "profile_history_ids.csv", history, list(history[0]))
        write_csv(selection / "selected_profiles.csv", [{"supplier_id": p, "history_count_provisional": 2} for p in ("p1", "p2")],
                  ["supplier_id", "history_count_provisional"])
        write_csv(selection / "candidate_ids.csv", [{"opportunity_id": "not_loaded"}], ["opportunity_id"])
        selection_hashes = {p.name: sha256(p) for p in selection.glob("*.csv")}
        (selection / "selection_manifest.json").write_text(json.dumps({"selection_sha256": selection_hashes}))
        original = preparation / "frozen_evaluation_plan.json"
        original.write_text(json.dumps({"cutoff": "2025-01-01"}))
        write_csv(preparation / "version_queue.csv", queue, list(queue[0]))
        write_csv(preparation / "historical_version_queue.csv", queue, list(queue[0]))
        (preparation / "preparation_manifest.json").write_text(json.dumps({"plan_sha256": sha256(original), "queue_sha256": sha256(preparation / "version_queue.csv")}))
        plan = preparation / "frozen_plan_with_code.json"
        plan.write_text(json.dumps({"original_plan_sha256": sha256(original), "selection_manifest_sha256": sha256(selection / "selection_manifest.json"),
                                   "selection_sha256": selection_hashes, "cutoff": "2025-01-01", "code_archive_sha256": {}, "code_archive": str(root)}))
        write_csv(recovered / "historical_version_results.csv", results, list(results[0]))
        (recovered / "fetch_manifest.json").write_text(json.dumps({"results_sha256": sha256(recovered / "historical_version_results.csv"),
                                                                 "queue_sha256": sha256(preparation / "historical_version_queue.csv"), "plan_sha256": sha256(plan)}))
        return selection, preparation, recovered

    def test_partial_award_evidence_and_native_publication_keep_only_eligible_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection, preparation, recovered = self.fixture(root)
            # No future label file or candidate text corpus is provided.
            report = materialize_contrast_history(selection, preparation, recovered, root / "validated")
            self.assertEqual(report["eligible_profiles_ge_2"], 1)
            self.assertEqual(report["excluded_profiles"], 1)
            self.assertEqual(report["retained_historical_awards"], 2)
            self.assertEqual(report["award_statuses"]["no_pre_cutoff_text_version"], 1)
            self.assertEqual(report["award_statuses"]["no_pre_cutoff_award_evidence"], 1)
            self.assertEqual(report["replaced_profiles"], 0)
            self.assertFalse(report["labels_read"])
            self.assertEqual({r["currency"] for r in read_csv(root / "validated/historical_awards.csv")}, {"USD"})
            self.assertEqual(len(read_csv(root / "validated/profile_eligibility.csv")), 2)
            with self.assertRaises(ValueError):
                materialize_contrast_history(selection, preparation, recovered, root / "validated")
            (recovered / "h1_tender.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Hash de evidencia"):
                materialize_contrast_history(selection, preparation, recovered, root / "validated2")

    def test_queue_modification_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection, preparation, _ = self.fixture(root)
            verify_history_queue(selection, preparation)
            rows = read_csv(preparation / "historical_version_queue.csv")
            write_csv(preparation / "historical_version_queue.csv", rows[:-1], list(rows[0]))
            with self.assertRaisesRegex(ValueError, "subconjunto congelado"):
                verify_history_queue(selection, preparation)

    def test_recovery_reuses_pinned_sources_without_network_and_keeps_hash_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection, preparation, recovered = self.fixture(root)
            with patch("tesis_recommender.validation.subprocess.run", side_effect=AssertionError("No network expected")):
                report = fetch_contrast_history(selection, preparation, root / "cache", root / "fetch", [recovered / "historical_version_results.csv"], workers=2)
            self.assertEqual(report["ok"], 8)
            self.assertEqual(report["reusable_versions"], 8)
            self.assertFalse(report["labels_read"])
            (recovered / "h1_tender.json").write_text("{}")
            with patch("tesis_recommender.validation.subprocess.run", side_effect=AssertionError("No network expected")):
                report = fetch_contrast_history(selection, preparation, root / "cache", root / "fetch", [recovered / "historical_version_results.csv"], workers=2)
            self.assertEqual(report["failed"], 1)
            with self.assertRaises(ValueError):
                fetch_contrast_history(selection, preparation, root / "cache", root / "fetch", [], workers=0)


if __name__ == "__main__":
    unittest.main()
