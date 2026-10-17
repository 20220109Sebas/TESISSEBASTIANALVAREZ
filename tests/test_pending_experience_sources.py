import unittest

from scripts.prepare_pending_experience_sources import partition_pending


class PendingSourceTests(unittest.TestCase):
    def test_fixes_all_26_in_original_order_and_bounded_batches(self):
        queue = [{"url": str(i), "opportunity_id": str(i)} for i in range(50)]
        batches = partition_pending(queue, [{"url": str(i)} for i in range(24)])
        self.assertEqual([len(b) for b in batches], [5, 5, 5, 5, 5, 1])
        self.assertEqual([r["url"] for b in batches for r in b], [str(i) for i in range(24, 50)])

    def test_does_not_retry_previously_failed_or_complete_sources(self):
        queue = [{"url": str(i), "opportunity_id": str(i)} for i in range(4)]
        accesses = [{"url": "0", "status": "access_error"}, {"url": "1", "status": "too_large_partial_file"}, {"url": "2", "status": "downloaded_pdf"}]
        self.assertEqual(partition_pending(queue, accesses), [[queue[3]]])

    def test_rejects_duplicate_queue_identity(self):
        for queue in [[{"url": "a", "opportunity_id": "x"}, {"url": "a", "opportunity_id": "y"}],
                      [{"url": "a", "opportunity_id": "x"}, {"url": "b", "opportunity_id": "x"}]]:
            with self.assertRaises(ValueError): partition_pending(queue, [])
