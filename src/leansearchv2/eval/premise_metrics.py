"""Premise-retrieval metrics: group-level recall and Covered@k.

Aligned with `task2_others/score_task2bench.py` from the paper run. Each
benchmark row carries a list of premise groups, each `{"kind": "original"|"alternative", "docs": [name, ...]}`.
A group is "hit" at k iff any of its `docs` appears in the retrieved
top-k after short-name normalization.

Reported metrics (all in percent over the query set):
- recall_at_k (macro_recall_group): mean per-query (hits/N).
- covered_at_k (solved_strict): per-query 1/0; the original PR routing's
  groups are *all* hit, OR (alternative groups exist AND are all hit).
  Matches the paper's "Covered" wording: some complete proof routing has
  every premise group hit.
"""

from __future__ import annotations


KS = [5, 10, 20, 30, 50]


def short_name(doc_id: str) -> str:
    """Strip a trailing `::<int>` segment (cuvs index suffix) and return the
    last `::`-separated component, which is the bare Mathlib name."""
    nd = doc_id
    parts = nd.rsplit("::", 1)
    if len(parts) == 2 and parts[1].isdigit():
        nd = parts[0]
    return nd.rsplit("::", 1)[-1] if "::" in nd else nd


def score_one(
    retrieved: list[str],
    premise_groups: list[dict],
    ks: list[int] = KS,
) -> dict[str, dict[int, float]]:
    """Score a single query. `retrieved` is the ranked list of doc_ids
    (e.g. `'Mathlib.Foo::Foo.bar'`); `premise_groups` is the row's
    `premise_group` list. Returns a dict with `recall_group` (per-k float)
    and `covered` (per-k 0/1)."""
    retrieved_short = [short_name(d) for d in retrieved]
    orig_groups = [g for g in premise_groups if g.get("kind") == "original"]
    alt_groups = [g for g in premise_groups if g.get("kind") == "alternative"]
    N = len(premise_groups)
    out_recall: dict[int, float] = {}
    out_covered: dict[int, int] = {}
    for k in ks:
        seen = set(retrieved_short[:k])

        def hit(g: dict) -> bool:
            return any(d in seen for d in g.get("docs", []))

        hit_orig = [hit(g) for g in orig_groups]
        hit_alt = [hit(g) for g in alt_groups]
        n_hit = sum(hit_orig) + sum(hit_alt)
        out_recall[k] = n_hit / max(N, 1)
        main_done = all(hit_orig) if orig_groups else True
        alt_all = len(alt_groups) > 0 and all(hit_alt)
        out_covered[k] = int(main_done or alt_all)
    return {"recall_group": out_recall, "covered": out_covered}


def aggregate(per_query: list[dict], ks: list[int] = KS) -> dict[str, dict[int, float]]:
    """Macro-average across queries. Returns percent values."""
    n = len(per_query)
    out: dict[str, dict[int, float]] = {"recall_group": {}, "covered": {}}
    if n == 0:
        for k in ks:
            out["recall_group"][k] = 0.0
            out["covered"][k] = 0.0
        return out
    for k in ks:
        out["recall_group"][k] = 100.0 * sum(r["recall_group"][k] for r in per_query) / n
        out["covered"][k] = 100.0 * sum(r["covered"][k] for r in per_query) / n
    return out
