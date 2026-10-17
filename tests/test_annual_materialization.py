import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/materialize_annual_pool.py"
spec = importlib.util.spec_from_file_location("annual_materialization", SCRIPT)
annual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annual)


class AnnualMaterializationTests(unittest.TestCase):
    def reference(self, number):
        return {"url": f"https://example.test/{number}", "ocid": str(number), "role": "test_candidate",
                "cache_key": str(number), "version_hint": "2025-02-01", "reference_date": "2025-02-01"}

    def test_checkpoint_merge_keeps_the_annual_pool_and_rejects_changed_sources(self):
        refs = [self.reference(i) for i in range(3)]
        expected = {row["url"]: row for row in refs}
        good = {**refs[0], "status": "ok", "sha256": "a" * 64, "cache_path": "a.json"}
        failed = {**refs[1], "status": "failed", "sha256": "", "cache_path": "b.json"}
        rows = annual.aggregate_checkpoint([good], [failed], expected)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(expected), 3)
        retried = {**refs[1], "status": "ok", "sha256": "b" * 64, "cache_path": "b.json"}
        self.assertEqual(annual.aggregate_checkpoint(rows, [retried], expected)[1]["status"], "ok")
        with self.assertRaisesRegex(ValueError, "Cambió"):
            annual.aggregate_checkpoint([good], [{**good, "sha256": "c" * 64}], expected)
        with self.assertRaisesRegex(ValueError, "Cambió"):
            annual.aggregate_checkpoint([good], [{**good, "status": "failed"}], expected)
        with self.assertRaisesRegex(ValueError, "cola"):
            annual.aggregate_checkpoint([good], [{**retried, "ocid": "other"}], expected)
        with self.assertRaisesRegex(ValueError, "repetidas"):
            annual.aggregate_checkpoint([good, good], [], expected)

    def test_partial_csv_writes_are_not_used_as_a_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.csv"
            self.assertIsNone(annual.stable_rows(path))
            path.write_text("url,status\nhttps://example.test/1,ok")
            self.assertIsNone(annual.stable_rows(path))
            path.write_text("url,status\nhttps://example.test/1\n")
            self.assertIsNone(annual.stable_rows(path))
            path.write_text("url,status\nhttps://example.test/1,ok\n")
            self.assertEqual(annual.stable_rows(path), [{"url": "https://example.test/1", "status": "ok"}])


if __name__ == "__main__":
    unittest.main()
