# SPDX-License-Identifier: MIT
"""Deterministic query-personalized PageRank."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _seed_vector(num_nodes: int, seeds: Mapping[int, float]) -> np.ndarray:
    vec = np.zeros(num_nodes, dtype=np.float64)
    for idx, value in seeds.items():
        if 0 <= idx < num_nodes and value > 0:
            vec[idx] += float(value)
    total = float(vec.sum())
    if total <= 0:
        raise ValueError("personalized PageRank requires at least one positive seed")
    return vec / total


def personalized_pagerank(
    adjacency: list[dict[int, float]],
    seeds: Mapping[int, float],
    *,
    restart_probability: float = 0.15,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Run power-iteration PPR over an outgoing weighted adjacency list."""
    if not 0.0 < restart_probability < 1.0:
        raise ValueError("restart_probability must be between 0 and 1")
    n = len(adjacency)
    seed = _seed_vector(n, seeds)
    scores = seed.copy()
    row_totals = np.asarray([sum(neighbors.values()) for neighbors in adjacency], dtype=np.float64)

    for _ in range(max_iterations):
        nxt = restart_probability * seed
        dangling_mass = 0.0
        for source, neighbors in enumerate(adjacency):
            mass = scores[source]
            if mass == 0:
                continue
            total = row_totals[source]
            if total <= 0:
                dangling_mass += mass
                continue
            share = (1.0 - restart_probability) * mass / total
            for target, weight in neighbors.items():
                nxt[target] += share * weight
        if dangling_mass:
            nxt += (1.0 - restart_probability) * dangling_mass * seed
        total_score = float(nxt.sum())
        if total_score > 0:
            nxt /= total_score
        if float(np.abs(nxt - scores).sum()) <= tolerance:
            return nxt
        scores = nxt
    return scores


def transition_matrix_from_adjacency(adjacency: list[dict[int, float]]):
    """Build a row-stochastic sparse transition matrix from adjacency lists."""
    from scipy import sparse

    n = len(adjacency)
    src: list[int] = []
    dst: list[int] = []
    data: list[float] = []
    dangling = np.ones(n, dtype=bool)
    for source, neighbors in enumerate(adjacency):
        total = float(sum(neighbors.values()))
        if total <= 0:
            continue
        dangling[source] = False
        for target, weight in neighbors.items():
            if weight > 0:
                src.append(source)
                dst.append(int(target))
                data.append(float(weight) / total)
    transition = sparse.csr_matrix((data, (src, dst)), shape=(n, n), dtype=np.float64)
    return transition, dangling


def transition_matrix_from_edges(
    num_nodes: int,
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
):
    """Build a row-stochastic sparse transition matrix from serialized edges."""
    from scipy import sparse

    sources = np.asarray(sources, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    row_totals = np.bincount(sources, weights=weights, minlength=num_nodes)
    keep = (weights > 0) & (row_totals[sources] > 0)
    data = weights[keep] / row_totals[sources[keep]]
    transition = sparse.csr_matrix(
        (data, (sources[keep], targets[keep])),
        shape=(num_nodes, num_nodes),
        dtype=np.float64,
    )
    dangling = row_totals <= 0
    return transition, dangling


def personalized_pagerank_sparse(
    transition,
    dangling: np.ndarray,
    seeds: Mapping[int, float],
    *,
    restart_probability: float = 0.15,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Run power-iteration PPR over a row-stochastic sparse transition matrix."""
    if not 0.0 < restart_probability < 1.0:
        raise ValueError("restart_probability must be between 0 and 1")
    n = int(transition.shape[0])
    seed = _seed_vector(n, seeds)
    scores = seed.copy()
    dangling = np.asarray(dangling, dtype=bool)

    for _ in range(max_iterations):
        propagated = np.asarray(scores @ transition).ravel()
        dangling_mass = float(scores[dangling].sum()) if dangling.any() else 0.0
        nxt = restart_probability * seed + (1.0 - restart_probability) * propagated
        if dangling_mass:
            nxt += (1.0 - restart_probability) * dangling_mass * seed
        total_score = float(nxt.sum())
        if total_score > 0:
            nxt /= total_score
        if float(np.abs(nxt - scores).sum()) <= tolerance:
            return nxt
        scores = nxt
    return scores


def top_ppr_nodes(
    scores: np.ndarray,
    *,
    limit: int,
    exclude: set[int] | None = None,
) -> list[tuple[int, float]]:
    excluded = exclude or set()
    ranked = [
        (int(idx), float(score))
        for idx, score in enumerate(scores)
        if idx not in excluded and score > 0
    ]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:limit]
