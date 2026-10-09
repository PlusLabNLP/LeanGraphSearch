from __future__ import annotations

import math
import unittest

from leansearchv2.eval.search_metrics import ndcg_at_k, recall_at_k


class SearchMetricsTests(unittest.TestCase):
    def test_ndcg_at_1_perfect_rank1_hit(self) -> None:
        self.assertEqual(ndcg_at_k(["GroundTruth", "Other"], "GroundTruth", 1), 1.0)

    def test_ndcg_at_1_miss(self) -> None:
        self.assertEqual(ndcg_at_k(["Other", "GroundTruth"], "GroundTruth", 1), 0.0)

    def test_ndcg_at_5_rank_positions_and_outside_cutoff(self) -> None:
        self.assertEqual(ndcg_at_k(["gt", "x", "y", "z", "w"], "gt", 5), 1.0)
        self.assertAlmostEqual(
            ndcg_at_k(["x", "y", "gt", "z", "w"], "gt", 5),
            1.0 / math.log2(4),
        )
        self.assertAlmostEqual(
            ndcg_at_k(["x", "y", "z", "w", "gt"], "gt", 5),
            1.0 / math.log2(6),
        )
        self.assertEqual(ndcg_at_k(["x", "y", "z", "w", "v", "gt"], "gt", 5), 0.0)

    def test_ndcg_at_10_matches_single_relevant_dcg_formula(self) -> None:
        retrieved = ["x0", "x1", "x2", "gt", "x4", "x5", "x6", "x7", "x8", "x9"]
        self.assertAlmostEqual(ndcg_at_k(retrieved, "gt", 10), 1.0 / math.log2(5))

    def test_ndcg_at_50_rank_positions_and_cutoff(self) -> None:
        rank_1 = ["gt"] + [f"x{i}" for i in range(1, 60)]
        rank_5 = [f"x{i}" for i in range(4)] + ["gt"] + [f"x{i}" for i in range(5, 60)]
        rank_10 = [f"x{i}" for i in range(9)] + ["gt"] + [f"x{i}" for i in range(10, 60)]
        rank_50 = [f"x{i}" for i in range(49)] + ["gt"] + [f"x{i}" for i in range(50, 60)]
        rank_51 = [f"x{i}" for i in range(50)] + ["gt"]

        self.assertEqual(ndcg_at_k(rank_1, "gt", 50), 1.0)
        self.assertAlmostEqual(ndcg_at_k(rank_5, "gt", 50), 1.0 / math.log2(6))
        self.assertAlmostEqual(ndcg_at_k(rank_10, "gt", 50), 1.0 / math.log2(11))
        self.assertAlmostEqual(ndcg_at_k(rank_50, "gt", 50), 1.0 / math.log2(51))
        self.assertEqual(ndcg_at_k(rank_51, "gt", 50), 0.0)

    def test_recall_at_k_accepts_result_lists_shorter_than_k(self) -> None:
        self.assertEqual(recall_at_k(["a", "gt"], "gt", 100), 1.0)
        self.assertEqual(recall_at_k(["a", "b"], "gt", 100), 0.0)

    def test_ndcg_at_50_accepts_shorter_lists(self) -> None:
        self.assertAlmostEqual(ndcg_at_k(["a", "gt"], "gt", 50), 1.0 / math.log2(3))
        self.assertEqual(ndcg_at_k(["a", "b"], "gt", 50), 0.0)

    def test_single_ground_truth_duplicate_handling_uses_first_hit(self) -> None:
        retrieved = ["x", "gt", "y", "gt"]
        self.assertEqual(recall_at_k(retrieved, "gt", 5), 1.0)
        self.assertAlmostEqual(ndcg_at_k(retrieved, "gt", 5), 1.0 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(retrieved, "gt", 50), 1.0 / math.log2(3))


if __name__ == "__main__":
    unittest.main()
