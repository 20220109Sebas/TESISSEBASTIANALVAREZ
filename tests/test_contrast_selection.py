import json
import tempfile
import unittest
from pathlib import Path

from tesis_recommender.contrast_selection import select_contrast_cohort, prepare_contrast_sources, freeze_contrast_code
from tesis_recommender.error_dossier import build_error_dossier, REVIEW_FIELDS
from tesis_recommender.validation import read_csv, write_csv


class ContrastSelectionTests(unittest.TestCase):
    def test_selection_ignores_future_labels_texts_and_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            fields = ["supplier_id", "award_id", "award_date", "source_url", "object_type", "description", "region"]
            awards = [{"supplier_id": s, "award_id": f"{s}-{i}", "award_date": "2024-01-02",
                       "source_url": "source", "object_type": "services", "description": "old", "region": "region"}
                      for s in ("p1", "p2", "p3") for i in range(2)]
            awards.extend([dict(awards[0]), {**awards[0], "award_id": "future", "award_date": "2025-01-02"}])
            write_csv(corpus / "historical_awards.csv", awards, fields)
            opp_fields = ["opportunity_id", "publication_date", "source_url", "object_type", "awarded_supplier_id", "title"]
            candidates = [{"opportunity_id": f"o{i}", "publication_date": "2025-02-01", "source_url": "source",
                           "object_type": "services", "awarded_supplier_id": "p1", "title": "old"} for i in range(3)]
            write_csv(corpus / "test_opportunities.csv", candidates, opp_fields)
            exclusion = root / "exclude.csv"
            write_csv(exclusion, [{"supplier_id": "p3"}], ["supplier_id"])
            first = select_contrast_cohort(corpus, exclusion, root / "one", 2, 42)
            # No se necesita ni se abre historical_labels.csv. Alterar otros
            # campos consolidados no puede cambiar la identidad seleccionada.
            for award in awards:
                award["description"], award["region"] = "future text", "future region"
            for candidate in candidates:
                candidate["awarded_supplier_id"], candidate["title"] = "p3", "new title"
            write_csv(corpus / "historical_awards.csv", list(reversed(awards)), fields)
            write_csv(corpus / "test_opportunities.csv", list(reversed(candidates)), opp_fields)
            second = select_contrast_cohort(corpus, exclusion, root / "two", 2, 42)
            self.assertEqual(first["selection_sha256"], second["selection_sha256"])
            self.assertEqual(first["historical_awards_provisional"], 4)
            self.assertFalse(first["labels_read"])
            self.assertTrue(first["native_publication_audit_pending"])
            self.assertEqual({r["supplier_id"] for r in read_csv(root / "one/selected_profiles.csv")}, {"p1", "p2"})
            self.assertNotIn("awarded_supplier_id", read_csv(root / "one/candidate_ids.csv")[0])
            with self.assertRaises(ValueError):
                select_contrast_cohort(corpus, exclusion, root / "one", 2, 42)

    def test_invalid_sizes_and_ambiguous_duplicates_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                select_contrast_cohort(root, root / "missing", root / "out", 0)
            write_csv(root / "exclude.csv", [], ["supplier_id"])
            write_csv(root / "historical_awards.csv", [
                {"supplier_id": "p", "award_id": "a", "award_date": "2024-01-01", "source_url": "one", "object_type": "services"},
                {"supplier_id": "p", "award_id": "a", "award_date": "2024-02-01", "source_url": "one", "object_type": "services"}],
                ["supplier_id", "award_id", "award_date", "source_url", "object_type"])
            with self.assertRaisesRegex(ValueError, "duplicada"):
                select_contrast_cohort(root, root / "exclude.csv", root / "out", 1)

    def test_dossier_preserves_human_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / "human_reviews.csv", [{"case_id": "case", "reason": "Human observation"}],
                      ["case_id", *REVIEW_FIELDS])
            with self.assertRaisesRegex(ValueError, "revisiones humanas"):
                build_error_dossier(root / "missing", root, root, root)
            self.assertEqual(read_csv(root / "human_reviews.csv")[0]["reason"], "Human observation")

    def test_source_preparation_uses_native_identifier_hint_and_checks_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = root / "selection"
            selection.mkdir()
            write_csv(selection / "profile_history_ids.csv", [{"source_url": "https://example/record/history"}], ["source_url"])
            write_csv(selection / "candidate_ids.csv", [{"opportunity_id": "opportunity"}], ["opportunity_id"])
            from tesis_recommender.validation import sha256
            hashes = {p.name: sha256(p) for p in selection.glob("*.csv")}
            (selection / "selection_manifest.json").write_text(json.dumps({"selection_sha256": hashes,
                "cutoff": "2025-01-01", "candidate_end": "2025-12-31", "seed": 42, "evaluation_status": "test_provisional"}))
            write_csv(root / "lineage.csv", [{"ocid": ocid, "selected_snapshot_id": "2025-01"}
                                             for ocid in ("history", "opportunity")], ["ocid", "selected_snapshot_id"])
            snapshot = root / "raw/2025/01"
            snapshot.mkdir(parents=True)
            def ref(ocid, native, event, tag):
                identifier = f"{ocid}-{native}"
                return {"id": identifier, "url": f"https://contratacionesabiertas.oece.gob.pe/api/v1/release/{identifier}", "date": event, "tag": [tag]}
            (snapshot / "snapshot.json").write_text(json.dumps({"records": [
                {"ocid": "history", "releases": [ref("history", "2024-03-01T00:00:00-05:00", "2024-03-01", "tender"),
                    ref("history", "2025-06-01T00:00:00-05:00", "2020-01-01", "tender"),
                    ref("history", "2024-04-01T00:00:00-05:00", "2024-04-01", "award"),
                    ref("history", "2025-03-01T00:00:00-05:00", "2024-01-01", "award")]},
                {"ocid": "opportunity", "releases": [ref("opportunity", "2025-01-02T00:00:00-05:00", "2025-01-02", "tender")]}]}))
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"model_name": "test", "model_revision": "test-revision", "profile_strategy": "test", "opportunity_strategy": "test"}))
            report = prepare_contrast_sources(selection, root, root / "raw", metadata, root / "prepared")
            self.assertEqual(report["versions_to_inspect"], 3)
            self.assertEqual(report["historical_versions_to_inspect"], 2)
            self.assertFalse(report["labels_read"])
            self.assertFalse(report["publication_verified"])
            urls = [r["url"] for r in read_csv(root / "prepared/version_queue.csv")]
            self.assertFalse(any("2025-06" in u or "2025-03" in u for u in urls))
            frozen = freeze_contrast_code(root / "prepared")
            self.assertTrue(frozen["ranking_modules_unchanged"])
            self.assertGreater(frozen["archived_files"], 20)
            with self.assertRaisesRegex(ValueError, "no se sobrescribe"):
                freeze_contrast_code(root / "prepared")
            write_csv(selection / "candidate_ids.csv", [{"opportunity_id": "changed"}], ["opportunity_id"])
            with self.assertRaisesRegex(ValueError, "modificada"):
                prepare_contrast_sources(selection, root, root / "raw", metadata, root / "prepared2")


if __name__ == "__main__":
    unittest.main()
