import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tesis_recommender.contrast_candidates import audit_candidate_batch, fetch_candidate_batch
from tesis_recommender.contrast_embeddings import DocumentVectorStore, CheckedLaBSE
from tesis_recommender.semantic import safe_token_chunks
from tesis_recommender.validation import write_csv, read_csv, sha256


class FakeEncoder:
    model_name = "test-only"
    model_revision = "test-only-revision"
    model_max_seq_length = 10

    def __init__(self):
        self.encoded = []

    def token_lengths(self, texts):
        return [len(t.split()) + 2 for t in texts]

    def encode_documents(self, texts):
        self.encoded.extend(texts)
        return [[1.0, 0.0] for _ in texts]


class CandidateAndEmbeddingTests(unittest.TestCase):
    def test_encoder_rejects_oversized_chunks_before_model_execution(self):
        encoder = object.__new__(CheckedLaBSE)
        with patch.object(CheckedLaBSE, "model_max_seq_length", 10), \
                patch.object(encoder, "token_lengths", return_value=[11]), \
                patch("tesis_recommender.semantic.LaBSEEmbedder.encode") as underlying:
            with self.assertRaisesRegex(ValueError, "supera el límite"):
                encoder.encode(["oversized"])
            underlying.assert_not_called()

    def test_fetch_batches_reuse_sources_and_resume_without_reducing_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preparation = root / "preparation"
            preparation.mkdir()
            (preparation / "version_queue.csv").write_text("pinned queue")
            (preparation / "frozen_plan_with_code.json").write_text("{}")
            queue = [{"ocid": f"o{i}", "role": "test_candidate", "url": f"https://contratacionesabiertas.oece.gob.pe/api/v1/release/{i}",
                      "version_hint": "2025-01-02T00:00:00Z", "reference_date": "2025-01-02", "cache_key": f"o{i}"}
                     for i in range(4)]
            paths = []
            for i in range(4):
                source = root / f"source_{i}.json"
                source.write_text(json.dumps({"source": i}))
                paths.append(source)
            cached_index = root / "reused.csv"
            write_csv(cached_index, [{"initial_release_url": queue[0]["url"],
                "cache_path": str(paths[0]), "sha256": sha256(paths[0])}],
                ["initial_release_url", "cache_path", "sha256"])
            requests = []

            def fake_fetch(queue_path, cache_dir, results_path, index_path, workers):
                requested = read_csv(queue_path)
                requests.append([r["url"] for r in requested])
                rows = [{**r, "status": "ok", "cache_path": str(paths[int(r["ocid"][1:])]),
                         "sha256": sha256(paths[int(r["ocid"][1:])]), "release_published_at": "2025-01-02", "error": ""}
                        for r in requested]
                write_csv(results_path, rows, [*list(queue[0]), "status", "cache_path", "sha256", "release_published_at", "error"])
                return {"ok": len(rows)}

            with patch("tesis_recommender.contrast_candidates.candidate_queue", return_value=({}, queue)), \
                    patch("tesis_recommender.contrast_candidates.fetch_versions", side_effect=fake_fetch):
                first = fetch_candidate_batch(root, preparation, root / "raw", root / "fetch", [cached_index], max_new=2)
                self.assertEqual((first["ok"], first["pending_versions"], first["batch_reused"]), (3, 1, 1))
                self.assertFalse(first["retrieval_complete"])
                second = fetch_candidate_batch(root, preparation, root / "raw", root / "fetch", [], max_new=2)
                self.assertEqual((second["ok"], second["pending_versions"]), (4, 0))
                self.assertTrue(second["retrieval_complete"])
                self.assertFalse(second["labels_read"])
                self.assertEqual(requests[1], [queue[3]["url"]])
                self.assertEqual(len(read_csv(root / "fetch/candidate_version_results.csv")), 4)
                (root / "fetch/candidate_version_results.csv").write_text("changed")
                with self.assertRaisesRegex(ValueError, "manifiesto"):
                    fetch_candidate_batch(root, preparation, root / "raw", root / "fetch", [], max_new=2)

    def test_chunk_partition_covers_every_token_without_retokenization_overflow(self):
        tokens = list(range(31))
        chunks, counts, adjustments = safe_token_chunks(tokens, 10, lambda ids: ",".join(str(i) for i in ids),
            lambda text: len(text.split(",")) + 3)
        recovered = [int(i) for text in chunks for i in text.split(",")]
        self.assertEqual(recovered, tokens)
        self.assertEqual(sum(counts), len(tokens))
        self.assertGreater(adjustments, 0)
        self.assertTrue(all(len(t.split(",")) + 3 <= 10 for t in chunks))
        self.assertEqual(safe_token_chunks([], 10, str, len)[0], [""])
        with self.assertRaises(ValueError):
            safe_token_chunks([1], 2, str, len)
        with self.assertRaises(ValueError):
            safe_token_chunks([1], 10, str, lambda _: 11)

    def test_document_store_reuses_vectors_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encoder = FakeEncoder()
            store = DocumentVectorStore(root, encoder, {"revision": "test"}, documents_per_block=1)
            first = store.encode_documents(["one", "two", "one"])
            self.assertEqual(encoder.encoded, ["one", "two"])
            resumed_encoder = FakeEncoder()
            resumed = DocumentVectorStore(root, resumed_encoder, {"revision": "test"})
            self.assertEqual(resumed.encode_documents(["one", "two", "three"]), [[1.0, 0.0]] * 3)
            self.assertEqual(resumed_encoder.encoded, ["three"])
            self.assertEqual(resumed.last_encoding_audit["source_documents_reused"], 2)
            self.assertEqual(first[0], first[2])
            with self.assertRaises(ValueError):
                DocumentVectorStore(root, encoder, {"revision": "changed"})
            (root / "block_00001.npy").write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "modificado"):
                DocumentVectorStore(root, encoder, {"revision": "test"})

    def test_candidates_are_filtered_by_dates_not_winner_and_preserve_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history, preparation, recovered = [root / n for n in ("history", "preparation", "fetch")]
            for p in (history, preparation, recovered):
                p.mkdir()
            plan_path = preparation / "frozen_plan_with_code.json"
            plan_path.write_text("{}")
            write_csv(history / "historical_awards.csv", [{"amount": 100, "currency": "PEN", "category": "service"}], ["amount", "currency", "category"])
            (history / "history_validation_report.json").write_text(json.dumps({"output_sha256": {"historical_awards.csv": sha256(history / "historical_awards.csv")},
                "input_sha256": {str(plan_path): sha256(plan_path)}, "eligible_profiles_ge_2": 1, "retained_historical_awards": 2}))
            queue, results = [], []
            for ocid, published in (("good", "2025-03-01T10:00:00-05:00"), ("late", "2026-01-01T10:00:00-05:00"), ("pending", "")):
                identity = f"{ocid}-2025-03-01T10:00:00-05:00"
                url = f"https://contratacionesabiertas.oece.gob.pe/api/v1/release/{identity}"
                reference = {"ocid": ocid, "role": "test_candidate", "url": url, "cache_key": ocid, "version_hint": "2025-03-01T15:00:00Z"}
                queue.append(reference)
                if ocid == "pending":
                    continue
                release = {"ocid": ocid, "id": identity, "publishedDate": published, "tag": ["tender"],
                    "tender": {"mainProcurementCategory": "services", "datePublished": "2025-03-01", "title": "Servicio", "description": "Limpieza"},
                    "awards": [{"id": "a", "suppliers": [{"id": "future-winner"}]}]}
                source = root / f"{ocid}.json"
                source.write_text(json.dumps({"releases": [release]}))
                results.append({**reference, "status": "ok", "cache_path": str(source), "sha256": sha256(source)})
            result_path = recovered / "candidate_version_results.csv"
            write_csv(result_path, results, list(results[0]))
            def manifest():
                (recovered / "fetch_manifest.json").write_text(json.dumps({"results_sha256": sha256(result_path), "plan_sha256": sha256(plan_path), "retrieval_complete": False}))
            manifest()
            plan = {"cutoff": "2025-01-01", "candidate_end": "2025-12-31"}
            with patch("tesis_recommender.contrast_candidates.candidate_queue", return_value=(plan, queue)):
                report = audit_candidate_batch(root, preparation, history, recovered, root / "one")
            self.assertEqual(report["candidate_statuses"], {"verified": 1, "outside_publication_window": 1, "pending_download": 1})
            self.assertFalse(report["evaluation_ready"])
            self.assertFalse(report["labels_read"])
            self.assertEqual(read_csv(root / "one/test_opportunities.csv")[0]["awarded_supplier_id"], "")
            # Cambiar ganadores en la fuente no modifica los campos modelables.
            source = root / "good.json"
            payload = json.loads(source.read_text())
            payload["releases"][0]["awards"][0]["suppliers"] = [{"id": "different-winner"}]
            source.write_text(json.dumps(payload))
            results[0]["sha256"] = sha256(source)
            write_csv(result_path, results, list(results[0]))
            manifest()
            with patch("tesis_recommender.contrast_candidates.candidate_queue", return_value=(plan, queue)):
                audit_candidate_batch(root, preparation, history, recovered, root / "two")
            self.assertEqual(sha256(root / "one/test_opportunities.csv"), sha256(root / "two/test_opportunities.csv"))


if __name__ == "__main__":
    unittest.main()
