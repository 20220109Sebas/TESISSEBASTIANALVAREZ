import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tesis_recommender.validation import read_csv, sha256, write_csv


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/recover_annual_versions.py"
spec = importlib.util.spec_from_file_location("annual_recovery", SCRIPT)
annual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annual)


class AnnualRecoveryTests(unittest.TestCase):
    def test_recovery_reuses_pinned_checkpoint_and_commits_the_full_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("fetch", "preparation", "cache"):
                (root / name).mkdir()
            refs = [{"url": f"https://example.test/version{i}", "ocid": str(i), "role": "test_candidate",
                     "cache_key": str(i), "version_hint": "2025-02-01", "reference_date": "2025-02-01"}
                    for i in range(3)]
            result_fields = [*refs[0], "status", "cache_path", "sha256", "release_published_at", "error"]
            cached = []
            for i in (0, 1):
                source = root / "cache" / f"{i}.json"
                source.write_text(json.dumps({"releases": [{"ocid": str(i), "id": f"version{i}", "publishedDate": "2025-02-01"}]}))
                cached.append({**refs[i], "status": "ok", "cache_path": str(source), "sha256": sha256(source),
                               "release_published_at": "2025-02-01", "error": ""})
            write_csv(root / "fetch/candidate_version_results.csv", cached[:1], result_fields)
            (root / "fetch/fetch_manifest.json").write_text(json.dumps({"results_sha256": sha256(root / "fetch/candidate_version_results.csv")}))
            write_csv(root / "checkpoint.csv", cached[1:], result_fields)
            write_csv(root / "queue.csv", refs[1:], list(refs[0]))
            (root / "preparation/frozen_plan_with_code.json").write_text("{}")
            (root / "preparation/version_queue.csv").write_text("pinned queue")
            args = annual.argparse.Namespace(selection=root, preparation=root / "preparation", queue=root / "queue.csv",
                results=root / "checkpoint.csv", cache=root / "cache", fetch=root / "fetch", audit=root / "transport", workers=2)
            response = MagicMock()
            response.content = json.dumps({"releases": [{"ocid": "2", "id": "version2", "publishedDate": "2025-02-01"}]}).encode()
            client = MagicMock()
            client.get.return_value = response
            manager = MagicMock()
            manager.__enter__.return_value = client
            with patch.object(annual, "candidate_queue", return_value=({}, refs)), \
                    patch.object(annual.httpx, "Client", return_value=manager):
                annual.recover(args)
            client.get.assert_called_once_with(refs[2]["url"])
            manifest = json.loads((root / "fetch/fetch_manifest.json").read_text())
            self.assertEqual((manifest["ok"], manifest["pending_versions"]), (3, 0))
            self.assertTrue(manifest["retrieval_complete"])
            self.assertFalse(manifest["labels_read"])
            rows = read_csv(root / "fetch/candidate_version_results.csv")
            self.assertEqual(rows[:2], cached)
            self.assertEqual(manifest["results_sha256"], sha256(root / "fetch/candidate_version_results.csv"))
            self.assertTrue((root / "transport/initial_checkpoint.csv").exists())


if __name__ == "__main__":
    unittest.main()
