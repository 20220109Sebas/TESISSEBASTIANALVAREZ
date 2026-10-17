import unittest

from scripts.register_remaining_browser_experience_sources import select_batch


class RemainingBatchTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"batches": [{"sources": 5}, {"sources": 5}, {"sources": 3}],
                     "pending_sources": [{"opportunity_id": str(i)} for i in range(13)]}
        self.observations = [{"opportunity_id": str(i)} for i in range(5, 13)]

    def test_second_batch_keeps_five_fixed_sources(self):
        _, sources, observations = select_batch(self.plan, self.observations, 2)
        self.assertEqual(sources, self.plan["pending_sources"][5:10])
        self.assertEqual(observations, self.observations[:5])

    def test_third_batch_keeps_three_fixed_sources(self):
        _, sources, observations = select_batch(self.plan, self.observations, 3)
        self.assertEqual(sources, self.plan["pending_sources"][10:])
        self.assertEqual(observations, self.observations[5:])

    def test_rejects_reordered_or_incomplete_queue(self):
        for rows in (self.observations[::-1], self.observations[:-1]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                select_batch(self.plan, rows, 2)

    def test_rejects_old_or_unplanned_batch(self):
        for batch_number in (0, 1, 4):
            with self.subTest(batch_number=batch_number), self.assertRaises(ValueError):
                select_batch(self.plan, self.observations, batch_number)

    def test_rejects_duplicate_queue(self):
        rows = self.observations[:-1] + [self.observations[0]]
        with self.assertRaises(ValueError):
            select_batch(self.plan, rows, 3)


if __name__ == "__main__":
    unittest.main()
