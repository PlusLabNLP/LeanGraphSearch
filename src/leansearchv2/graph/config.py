# SPDX-License-Identifier: MIT
"""Configuration objects for graph-augmented standard search."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class GraphEdgeWeights:
    dep: float = 1.0
    rev_dep: float = 0.7
    same_module: float = 0.05
    semantic_knn: float = 0.0
    co_premise: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GraphEdgeWeights":
        if not data:
            return cls()
        allowed = {k: float(v) for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**allowed)

    def as_dict(self) -> dict[str, float]:
        return {
            "dep": self.dep,
            "rev_dep": self.rev_dep,
            "same_module": self.same_module,
            "semantic_knn": self.semantic_knn,
            "co_premise": self.co_premise,
        }


@dataclass(frozen=True)
class GraphConfig:
    graph_augment: bool = False
    artifact_dir: Path | None = None
    initial_top_n: int = 200
    graph_expand_m: int = 300
    final_top_k: int | None = None
    preserve_standard_top_k: int = 0
    graph_only_slots_after_preserved: int = 0
    restart_probability: float = 0.15
    max_iterations: int = 50
    tolerance: float = 1e-6
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.05
    infer_text_references: bool = False
    dependency_source: str = "structured_reference_fields"
    text_reference_fields: tuple[str, ...] = ("signature", "type", "value")
    max_text_refs_per_record: int = 128
    same_module_max_neighbors_per_node: int = 16
    same_module_max_module_size: int = 512
    enable_co_premise_edges: bool = False
    co_premise_max_refs_per_source: int = 64
    co_premise_max_ref_frequency: int = 512
    co_premise_theorem_like_only: bool = True
    dependency_fields: tuple[str, ...] = (
        "dependencies",
        "deps",
        "typeReferences",
        "valueReferences",
        "references",
    )
    dependency_field_weights: dict[str, float] = field(default_factory=dict)
    dependency_idf_fields: tuple[str, ...] = ()
    dependency_hub_max_frequency: int | None = None
    dependency_additive_fields: bool = False
    include_theorem_value_references: bool = False
    theorem_value_reference_weight: float = 1.0
    excluded_proof_dependency_source_names: tuple[str, ...] = ()
    generated_declaration_weight: float = 1.0
    reverse_dependency_hub_penalty: str = "none"
    reverse_dependency_theorem_like_weight: float = 1.0
    reverse_dependency_non_theorem_weight: float = 1.0
    graph_edge_profile: str = "dep_only"
    graph_seed_top_n: int | None = None
    ppr_restart_prob: float | None = None
    ppr_max_iter: int | None = None
    ppr_tol: float | None = None
    rerank_graph_union: bool = False
    reranker_required: bool = False
    rank_fusion: str = "none"
    ppr_zscore_weight: float = 0.005
    edge_weights: GraphEdgeWeights = field(default_factory=GraphEdgeWeights)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GraphConfig":
        if not data:
            return cls()
        payload = dict(data)
        aliases = {
            "graph_artifact_dir": "artifact_dir",
            "graph_seed_top_n": "initial_top_n",
            "preserve_top_k": "preserve_standard_top_k",
            "ppr_restart_prob": "restart_probability",
            "ppr_max_iter": "max_iterations",
            "ppr_tol": "tolerance",
        }
        for source, target in aliases.items():
            if source in payload and target not in payload:
                payload[target] = payload[source]
        if payload.get("artifact_dir") is not None:
            artifact_dir = Path(payload["artifact_dir"])
            payload["artifact_dir"] = artifact_dir if artifact_dir.is_absolute() else _REPO_ROOT / artifact_dir
        payload["edge_weights"] = GraphEdgeWeights.from_dict(payload.get("edge_weights"))
        for key in (
            "dependency_fields",
            "dependency_idf_fields",
            "text_reference_fields",
            "excluded_proof_dependency_source_names",
        ):
            if key in payload and payload[key] is not None and not isinstance(payload[key], tuple):
                payload[key] = tuple(payload[key])
        if payload.get("dependency_field_weights") is not None:
            payload["dependency_field_weights"] = {
                str(k): float(v) for k, v in dict(payload["dependency_field_weights"]).items()
            }
        allowed = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**allowed)
