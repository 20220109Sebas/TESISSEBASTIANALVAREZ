import unittest

from tesis_recommender.annex_queue import next_annex_batch


class AnnexQueueTests(unittest.TestCase):
    def setUp(self):
        self.queue = [{"opportunity_id": str(i), "url": f"url:{i}"} for i in range(10)]

    def test_reuses_complete_pdf_and_preserves_fixed_order(self):
        r = next_annex_batch(self.queue, [{"url": "url:0", "status": "downloaded_pdf"}])
        self.assertEqual([d["opportunity_id"] for d in r["selected"]], ["1", "2", "3", "4", "5"])
        self.assertEqual(r["completed_opportunities"], 1)

    def test_does_not_retry_oversized_or_failed_response(self):
        r = next_annex_batch(self.queue, [{"url": "url:0", "status": "too_large_partial_file"}, {"url": "url:1", "status": "access_error"}])
        self.assertEqual(r["selected"][0]["opportunity_id"], "2")
        self.assertEqual(len(r["pending_previously_attempted"]), 2)
        self.assertFalse(r["automatic_retries"])

    def test_other_urls_and_duplicate_attempts_do_not_inflate_coverage(self):
        access = [{"url": "url:0", "status": "downloaded_pdf"}] * 2 + [{"url": "not_in_queue", "status": "downloaded_pdf"}]
        self.assertEqual(next_annex_batch(self.queue, access)["completed_opportunities"], 1)

    def test_empty_queue_and_fully_attempted_queue_are_safe(self):
        self.assertEqual(next_annex_batch([], [])["selected"], [])
        r = next_annex_batch(self.queue, [{"url": d["url"], "status": "not_pdf"} for d in self.queue])
        self.assertEqual(r["selected"], [])
        self.assertEqual(r["pending_opportunities"], 10)

    def test_duplicate_or_unbounded_queue_rejected(self):
        with self.assertRaises(ValueError):
            next_annex_batch(self.queue + [self.queue[0]], [])
        with self.assertRaises(ValueError):
            next_annex_batch(self.queue, [], limit=6)
