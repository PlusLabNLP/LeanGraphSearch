"""Search-task metrics (single-relevant-document, binary relevance).

Used by `scripts/reproduce_mathlibqr.py`.
"""

from __future__ import annotations

import math
from typing import Iterable


def hit_position(retrieved: list[str], ground_truth: str) -> int | None:
    """Return the 1-indexed position of `ground_truth` in `retrieved`, or None."""
    for i, name in enumerate(retrieved, 1):
        if name == ground_truth:
            return i
    return None


def ndcg_at_k(retrieved: list[str], ground_truth: str, k: int) -> float:
    """DCG@k with one relevant document. IDCG@k = 1, so nDCG = DCG."""
    pos = hit_position(retrieved[:k], ground_truth)
    if pos is None:
        return 0.0
    return 1.0 / math.log2(pos + 1)


def recall_at_k(retrieved: list[str], ground_truth: str, k: int) -> float:
    return 1.0 if hit_position(retrieved[:k], ground_truth) is not None else 0.0


def summarize(per_query: Iterable[dict], k_values: list[int]) -> dict:
    """Average nDCG@k and Recall@k across the provided per-query records."""
    rows = list(per_query)
    n = len(rows)
    out = {"n": n}
    if n == 0:
        for k in k_values:
            out[f"ndcg@{k}"] = 0.0
            out[f"recall@{k}"] = 0.0
        return out
    for k in k_values:
        out[f"ndcg@{k}"] = sum(r["ndcg"][str(k)] for r in rows) / n
        out[f"recall@{k}"] = sum(r["recall"][str(k)] for r in rows) / n
    return out
