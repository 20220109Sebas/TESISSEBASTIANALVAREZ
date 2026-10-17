import csv
import gzip
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.demonstrate_annual_recommendations import (
    SIGNALS, build, checked, default_supplier, replay_scores, source_details, verify_ranking,
)


class AnnualDemonstrationTests(unittest.TestCase):
    def components(self):
        return {name: [0.5, 0.5] for name in SIGNALS}

    def weights(self):
        return dict(zip(SIGNALS, (0.35, 0.50, 0.08, 0.04, 0.03)))

    def test_profile_choice_independent_of_input_order(self):
        ids = ["PE-RUC-001", "PE-RUC-002", "PE-RUC-003"]
        expected = min(ids, key=lambda s: (hashlib.sha256(("42:" + s).encode()).hexdigest(), s))
        self.assertEqual(default_supplier(ids), expected)
        self.assertEqual(default_supplier(list(reversed(ids))), expected)

    def test_exact_python_score_and_ocid_tie_break(self):
        c, w = self.components(), self.weights()
        scores, ranked = replay_scores(["ocid-b", "ocid-a"], c, w)
        self.assertEqual(scores, [sum(w[name] * c[name][0] for name in SIGNALS)] * 2)
        self.assertEqual(ranked, [1, 0])

    def test_reject_invalid_components_or_population(self):
        cases = [("duplicates", ["a", "a"], self.components())]
        c = self.components(); c["semantic"][1] = math.nan
        cases.append(("nan", ["a", "b"], c))
        c = self.components(); c["amount"] = [0.5]
        cases.append(("length", ["a", "b"], c))
        for label, order, components in cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                replay_scores(order, components, self.weights())

    def test_reject_invalid_weights(self):
        for replacement in (-0.1, math.inf, 2.0):
            weights = self.weights(); weights["amount"] = replacement
            with self.assertRaises(ValueError):
                replay_scores(["a", "b"], self.components(), weights)

    def test_checksum_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"; path.write_text("{}")
            evidence = {}; expected = hashlib.sha256(b"{}").hexdigest()
            checked(path, expected, evidence)
            path.write_text("{ }")
            with self.assertRaises(ValueError):
                checked(path, expected, evidence)

    def test_ranking_verifies_all_rows_and_rejects_missing_or_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank.csv.gz"
            def write(rows):
                with gzip.open(path, "wt", newline="") as handle:
                    w = csv.DictWriter(handle, fieldnames=["supplier_id", "model", "rank", "opportunity_id", "score"])
                    w.writeheader(); w.writerows(rows)
            rows = [{"supplier_id": "p", "model": "m", "rank": n,
                     "opportunity_id": oid, "score": score}
                    for n, oid, score in [(1, "b", .75), (2, "a", .25)]]
            write(rows)
            self.assertEqual(verify_ranking(path, "p", "m", ["a", "b"], [.25, .75], [1, 0]), 2)
            write(rows[:1])
            with self.assertRaises(ValueError):
                verify_ranking(path, "p", "m", ["a", "b"], [.25, .75], [1, 0])
            rows[1]["score"] = .26; write(rows)
            with self.assertRaises(ValueError):
                verify_ranking(path, "p", "m", ["a", "b"], [.25, .75], [1, 0])

    def test_source_identity_and_exact_version_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            data = {"uri": "https://contratacionesabiertas.oece.gob.pe/api/v1/release/a-1",
                    "releases": [{"ocid": "a", "id": "a-1", "publishedDate": "2024-12-01T00:00:00-05:00"}]}
            path.write_text(json.dumps(data))
            source = source_details(path, hashlib.sha256(path.read_bytes()).hexdigest(), "a", {})
            self.assertEqual(source["release_id"], "a-1")
            with self.assertRaises(ValueError):
                source_details(path, hashlib.sha256(path.read_bytes()).hexdigest(), "different", {})
            data["uri"] = "https://contratacionesabiertas.oece.gob.pe/api/v1/record/a"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                source_details(path, hashlib.sha256(path.read_bytes()).hexdigest(), "a", {})

    def test_protect_existing_output_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing"; output.mkdir()
            note = output / "note.txt"; note.write_text("keep")
            with self.assertRaises(ValueError):
                build(Path(tmp) / "missing", None, 10, output)
            self.assertEqual(note.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
