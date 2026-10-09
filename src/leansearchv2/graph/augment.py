# SPDX-License-Identifier: MIT
"""Merge standard-mode candidates with graph-expanded candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .artifact import GraphArtifact, GraphArtifactError
from .config import GraphConfig
from .ppr import personalized_pagerank, personalized_pagerank_sparse, top_ppr_nodes


@dataclass
class CandidateScore:
    index: int
    standard_score: float
    distance: float
    graph_score: float | None = None
    combined_score: float | None = None
    seed_source: str = "standard"
    graph_evidence: list[dict] = field(default_factory=list)


def _normalize(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if math.isclose(lo, hi):
        return {k: 1.0 if v > 0 else 0.0 for k, v in values.items()}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _seed_distribution(candidates: list[CandidateScore], limit: int) -> dict[int, float]:
    seeds: dict[int, float] = {}
    top = candidates[: max(1, limit)]
    use_scores = any(c.standard_score > 0 for c in top)
    for rank, candidate in enumerate(top, 1):
        seeds[candidate.index] = candidate.standard_score if use_scores and candidate.standard_score > 0 else 1.0 / rank
    return seeds


def _best_evidence(artifact: GraphArtifact, seeds: set[int], target: int) -> list[dict]:
    evidence = []
    for seed in sorted(seeds):
        direct = artifact.evidence_for(seed, target)
        if direct:
            evidence.append(direct)
        reverse = artifact.evidence_for(target, seed)
        if reverse:
            evidence.append(reverse)
        if evidence:
            break
    return evidence


def augment_candidates(
    standard_candidates: list[CandidateScore],
    *,
    artifact: GraphArtifact | None,
    config: GraphConfig,
    final_top_k: int,
) -> list[CandidateScore]:
    if not config.graph_augment:
        return list(standard_candidates[:final_top_k])
    if artifact is None:
        raise GraphArtifactError(
            "graph_augment=true requires a graph artifact. Build it with "
            "`python -m leansearchv2.graph.build --metadata-pkl <cuvs_db>/metadata.pkl "
            "--output-dir <graph_dir>` and set graph.artifact_dir."
        )
    if not standard_candidates:
        return []

    seeds = _seed_distribution(standard_candidates, config.initial_top_n)
    if artifact.transition is not None and artifact.dangling is not None:
        ppr_scores = personalized_pagerank_sparse(
            artifact.transition,
            artifact.dangling,
            seeds,
            restart_probability=config.restart_probability,
            max_iterations=config.max_iterations,
            tolerance=config.tolerance,
        )
    else:
        ppr_scores = personalized_pagerank(
            artifact.adjacency,
            seeds,
            restart_probability=config.restart_probability,
            max_iterations=config.max_iterations,
            tolerance=config.tolerance,
        )

    candidate_map: dict[int, CandidateScore] = {}
    for candidate in standard_candidates:
        graph_score = float(ppr_scores[candidate.index]) if candidate.index < len(ppr_scores) else None
        candidate_map[candidate.index] = CandidateScore(
            index=candidate.index,
            standard_score=candidate.standard_score,
            distance=candidate.distance,
            graph_score=graph_score,
            seed_source="standard",
            graph_evidence=list(candidate.graph_evidence),
        )

    seed_indices = set(seeds)
    fallback_distance = max((c.distance for c in standard_candidates), default=0.0)
    for idx, graph_score in top_ppr_nodes(ppr_scores, limit=config.graph_expand_m, exclude=set()):
        if idx in candidate_map:
            candidate_map[idx].graph_score = graph_score
            if idx not in seed_indices:
                candidate_map[idx].seed_source = "standard_graph_neighbor"
            continue
        candidate_map[idx] = CandidateScore(
            index=idx,
            standard_score=0.0,
            distance=fallback_distance,
            graph_score=graph_score,
            seed_source="graph_only",
            graph_evidence=_best_evidence(artifact, seed_indices, idx),
        )

    standard_norm = _normalize({idx: c.standard_score for idx, c in candidate_map.items()})
    graph_norm = _normalize({idx: c.graph_score or 0.0 for idx, c in candidate_map.items()})
    for idx, candidate in candidate_map.items():
        degree = 0
        if idx < len(artifact.in_degree):
            degree += artifact.in_degree[idx]
        if idx < len(artifact.out_degree):
            degree += artifact.out_degree[idx]
        candidate.combined_score = (
            config.alpha * standard_norm.get(idx, 0.0)
            + config.beta * graph_norm.get(idx, 0.0)
            - config.gamma * math.log1p(degree)
        )

    ranked = sorted(
        candidate_map.values(),
        key=lambda c: (-(c.combined_score or 0.0), c.seed_source == "graph_only", c.index),
    )

    preserve = max(0, min(config.preserve_standard_top_k, final_top_k))
    slots = max(0, min(config.graph_only_slots_after_preserved, max(0, final_top_k - preserve)))
    if preserve or slots:
        preserved: list[CandidateScore] = []
        preserved_indices: set[int] = set()
        for candidate in standard_candidates[:preserve]:
            kept = candidate_map.get(candidate.index, candidate)
            preserved.append(kept)
            preserved_indices.add(candidate.index)

        remainder = [candidate for candidate in ranked if candidate.index not in preserved_indices]
        if slots:
            graph_only = [candidate for candidate in remainder if candidate.seed_source == "graph_only"]
            slotted = graph_only[:slots]
            slotted_indices = {candidate.index for candidate in slotted}
            remainder = slotted + [candidate for candidate in remainder if candidate.index not in slotted_indices]
        ranked = preserved + remainder
    return ranked[:final_top_k]
