"""Pruebas sintéticas de categorías, sin usar positivos para definir reglas."""
import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tesis_recommender.category_review import code_keys, extract_categories, normalize_label, primary_labels

_path = Path(__file__).resolve().parents[1] / "scripts/audit_annual_categories.py"
_spec = importlib.util.spec_from_file_location("annual_category_review", _path)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def tender(label="SERVICIO A", cubso="7612200300391331", unspsc="76122003"):
    return {"mainProcurementCategory": "services", "items": [
        {"id": "item-1", "classification": {"scheme": "CUBSO", "id": cubso, "description": label},
         "additionalClassifications": [{"scheme": "UNSPSC", "id": unspsc, "description": "Servicio clasificado"}]}]}


class CategoryReviewTests(unittest.TestCase):
    def test_normalization_changes_format_not_meaning(self):
        self.assertEqual(normalize_label("  SERVICIO  DE\tLIMPIEZA\n"), "servicio de limpieza")
        self.assertEqual(normalize_label("remediacio\u0301n"), "remediación")
        self.assertNotEqual(normalize_label("peligrosos"), normalize_label("no peligrosos"))
        self.assertNotEqual(normalize_label("remediación"), normalize_label("remediacion"))

    def test_keeps_first_category_and_all_item_codes(self):
        value = tender()
        value["items"] += tender("SERVICIO B", "7711160200232632", "77111602")["items"]
        result = extract_categories(value)
        self.assertEqual(result["legacy_category"], "SERVICIO A")
        self.assertEqual(result["item_count"], 2)
        self.assertEqual(len(code_keys(result, "UNSPSC")), 2)
        self.assertEqual(len(code_keys(result, "UNSPSC", first_item_only=True)), 1)
        self.assertEqual(primary_labels(result), {"servicio a", "servicio b"})

    def test_missing_and_malformed_codes_are_not_inferred(self):
        result = extract_categories(tender(cubso="7612200300391331", unspsc=""))
        self.assertEqual(code_keys(result, "UNSPSC"), set())
        for malformed in ("7612200", "761220030", "７６１２２００３", "7612 003", "nan"):
            self.assertFalse(code_keys(extract_categories(tender(unspsc=malformed)), "UNSPSC"))
        self.assertEqual(extract_categories({"mainProcurementCategory": "services"})["classifications"], [])

    def test_codes_remain_scheme_scoped(self):
        value = tender()
        value["items"][0]["additionalClassifications"].append({"scheme": "OTHER", "id": "76122003"})
        result = extract_categories(value)
        self.assertEqual(code_keys(result, "UNSPSC"), {("UNSPSC", "76122003")})
        self.assertEqual(code_keys(result, "OTHER"), set())

    def test_shared_unspsc_does_not_require_same_cubso_or_label(self):
        history = extract_categories(tender("HOSPITALARIOS", "7612200300232630"))
        opportunity = extract_categories(tender("PELIGROSOS"))
        self.assertFalse(code_keys(history, "CUBSO") & code_keys(opportunity, "CUBSO"))
        self.assertTrue(code_keys(history, "UNSPSC") & code_keys(opportunity, "UNSPSC"))

    def test_inventory_reports_both_kinds_of_ambiguity(self):
        data = [extract_categories(tender("A")), extract_categories(tender("B")),
                extract_categories(tender("A", "7612200300232630"))]
        for i, record in enumerate(data):
            record.update({"ocid": str(i), "source_path": f"source-{i}"})
        catalog, aliases = audit.inventory(data)
        self.assertEqual(next(r for r in catalog if r["code"] == "7612200300391331")["normalized_description_count"], 2)
        self.assertEqual(next(r for r in aliases if r["scheme"] == "CUBSO")["codes"], ["7612200300232630", "7612200300391331"])

    def build_fixture(self, root):
        experiment, cohort, history = (root / name for name in ("experiment", "cohort", "history"))
        for directory in (experiment, cohort, history):
            directory.mkdir()

        def csv_write(path, records):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)

        sources = {}
        for ocid, label, cubso in (("o1", "PELIGROSOS", "7612200300391331"), ("h1", "HOSPITALARIOS", "7612200300232630")):
            path = root / f"{ocid}.json"
            path.write_text(json.dumps({"releases": [{"ocid": ocid, "id": ocid + "-version", "tender": tender(label, cubso)}]}))
            sources[ocid] = path
        csv_write(cohort / "test_opportunities.csv", [{"opportunity_id": "o1", "category": "PELIGROSOS"}])
        csv_write(cohort / "historical_awards.csv", [{"award_id": "a1", "category": "HOSPITALARIOS", "supplier_id": "p1"}])
        csv_write(cohort / "opportunity_availability.csv", [{"ocid": "o1", "status": "verified", "source_path": str(sources["o1"]), "sha256": audit.digest(sources["o1"])}])
        csv_write(history / "award_availability.csv", [{"ocid": "h1", "award_id": "a1", "supplier_id": "p1", "status": "verified", "text_source": str(sources["h1"]), "text_sha256": audit.digest(sources["h1"])}])
        report_path = history / "history_validation_report.json"
        report_path.write_text(json.dumps({"output_sha256": {"award_availability.csv": audit.digest(history / "award_availability.csv")},
                                          "evidence_sha256": {str(sources["h1"]): audit.digest(sources["h1"])}}))
        candidate_report = cohort / "candidate_validation_report.json"
        candidate_report.write_text(json.dumps({"input_sha256": {str(report_path): audit.digest(report_path)},
                                               "output_sha256": {"opportunity_availability.csv": audit.digest(cohort / "opportunity_availability.csv")},
                                               "evidence_sha256": {str(sources["o1"]): audit.digest(sources["o1"])}}))
        (experiment / "results.json").write_text("{}")
        (experiment / "manifest.json").write_text(json.dumps({"status": "complete", "results_sha256": audit.digest(experiment / "results.json"),
            "input_sha256": {str(path): audit.digest(path) for path in (candidate_report, cohort / "test_opportunities.csv", cohort / "historical_awards.csv")}}))
        return SimpleNamespace(experiment=experiment, cohort=cohort, history=history, output=root / "audit"), sources

    def test_end_to_end_preserves_sources_and_never_runs_rankings(self):
        with tempfile.TemporaryDirectory() as directory:
            args, sources = self.build_fixture(Path(directory))
            before = {str(p): audit.digest(p) for p in sources.values()}
            with contextlib.redirect_stdout(io.StringIO()):
                summary = audit.audit(args)
            self.assertFalse(summary["labels_read"])
            self.assertFalse(summary["rankings_run"])
            self.assertEqual(summary["profiles"], 1)
            self.assertEqual(before, {str(p): audit.digest(p) for p in sources.values()})
            profile = audit.json_read(args.output / "profile_match_counts.json")[0]
            self.assertEqual(profile["exact_unspsc_any_item"], 1)
            self.assertEqual(profile["legacy_first_label"], 0)
            with self.assertRaises(ValueError):
                audit.audit(args)

    def test_tampered_source_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args, sources = self.build_fixture(Path(directory))
            sources["o1"].write_text("{}")
            with self.assertRaises(ValueError):
                audit.audit(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
