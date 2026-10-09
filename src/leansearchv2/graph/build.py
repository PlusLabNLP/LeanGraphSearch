# SPDX-License-Identifier: MIT
"""CLI entry point for building standard-mode graph artifacts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import json

from ..config import get_path
from .artifact import build_graph_artifact
from .config import GraphConfig, GraphEdgeWeights


log = logging.getLogger("leansearchv2.graph.build")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build graph artifacts from LeanSearch corpus metadata")
    parser.add_argument(
        "--metadata-pkl",
        type=str,
        default=None,
        help="Path to metadata.pkl or a supported JSON/JSONL/GZ/Parquet record source (default: <paths.cuvs_db>/metadata.pkl)",
    )
    parser.add_argument(
        "--records-source",
        type=str,
        default=None,
        help="Alias for --metadata-pkl that accepts JSON, JSONL, JSONL.GZ, Parquet, or a directory of those files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Graph artifact directory (default: <paths.cuvs_db>/graph)",
    )
    parser.add_argument("--dep-weight", type=float, default=1.0)
    parser.add_argument("--rev-dep-weight", type=float, default=0.7)
    parser.add_argument("--same-module-weight", type=float, default=0.05)
    parser.add_argument("--same-module-max-neighbors-per-node", type=int, default=16)
    parser.add_argument("--same-module-max-module-size", type=int, default=512)
    parser.add_argument(
        "--reverse-dependency-hub-penalty",
        choices=("none", "log2"),
        default="none",
        help="Discount reverse edges from declarations referenced by many sources.",
    )
    parser.add_argument(
        "--reverse-dependency-theorem-like-weight",
        type=float,
        default=1.0,
        help="Multiplier for reverse edges whose destination is theorem/lemma-like.",
    )
    parser.add_argument(
        "--reverse-dependency-non-theorem-weight",
        type=float,
        default=1.0,
        help="Multiplier for reverse edges whose destination is not theorem/lemma-like.",
    )
    parser.add_argument("--co-premise-weight", type=float, default=0.0)
    parser.add_argument("--enable-co-premise-edges", action="store_true")
    parser.add_argument("--co-premise-max-refs-per-source", type=int, default=64)
    parser.add_argument("--co-premise-max-ref-frequency", type=int, default=512)
    parser.add_argument(
        "--co-premise-include-non-theorems",
        action="store_true",
        help="Allow co-premise source records that are not theorem/lemma-like",
    )
    parser.add_argument(
        "--infer-text-references",
        action="store_true",
        help="Infer dependency edges by exact declaration-name matches in signature/type/value fields",
    )
    parser.add_argument("--max-text-refs-per-record", type=int, default=128)
    parser.add_argument(
        "--dependency-source",
        default="structured_reference_fields",
        help="Label describing the provenance of dependency/reference edges in edge_stats.json",
    )
    parser.add_argument(
        "--include-theorem-value-references",
        action="store_true",
        help="Add directed theorem/lemma/Prop -> valueReferences proof-dependency edges.",
    )
    parser.add_argument(
        "--theorem-value-reference-weight",
        type=float,
        default=1.0,
        help="Weight multiplier for theorem/lemma/Prop valueReferences edges.",
    )
    parser.add_argument(
        "--exclude-proof-dependency-sources-json",
        type=str,
        default=None,
        help=(
            "JSON benchmark/list whose formal_main_result/name values are excluded "
            "as proof-dependency edge sources to prevent target-proof leakage."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cuvs_db = Path(get_path("VECTORDB_DIR", "paths", "cuvs_db"))
    metadata_path = Path(args.records_source or args.metadata_pkl) if (args.records_source or args.metadata_pkl) else cuvs_db / "metadata.pkl"
    output_dir = Path(args.output_dir) if args.output_dir else cuvs_db / "graph"
    excluded_proof_sources: tuple[str, ...] = ()
    if args.exclude_proof_dependency_sources_json:
        exclusion_path = Path(args.exclude_proof_dependency_sources_json)
        payload = json.loads(exclusion_path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
        if not isinstance(rows, list):
            parser.error("proof-dependency exclusion JSON must be a list or contain a rows list")
        values: list[str] = []
        for row in rows:
            if isinstance(row, str):
                values.append(row)
            elif isinstance(row, dict):
                value = row.get("formal_main_result") or row.get("name")
                if value:
                    values.append(str(value))
        excluded_proof_sources = tuple(dict.fromkeys(values))
        if not excluded_proof_sources:
            parser.error("proof-dependency exclusion JSON did not contain any declaration names")

    cfg = GraphConfig(
        infer_text_references=args.infer_text_references,
        dependency_source=args.dependency_source,
        include_theorem_value_references=args.include_theorem_value_references,
        theorem_value_reference_weight=args.theorem_value_reference_weight,
        excluded_proof_dependency_source_names=excluded_proof_sources,
        max_text_refs_per_record=args.max_text_refs_per_record,
        same_module_max_neighbors_per_node=args.same_module_max_neighbors_per_node,
        same_module_max_module_size=args.same_module_max_module_size,
        enable_co_premise_edges=args.enable_co_premise_edges,
        co_premise_max_refs_per_source=args.co_premise_max_refs_per_source,
        co_premise_max_ref_frequency=args.co_premise_max_ref_frequency,
        co_premise_theorem_like_only=not args.co_premise_include_non_theorems,
        reverse_dependency_hub_penalty=args.reverse_dependency_hub_penalty,
        reverse_dependency_theorem_like_weight=args.reverse_dependency_theorem_like_weight,
        reverse_dependency_non_theorem_weight=args.reverse_dependency_non_theorem_weight,
        edge_weights=GraphEdgeWeights(
            dep=args.dep_weight,
            rev_dep=args.rev_dep_weight,
            same_module=args.same_module_weight,
            co_premise=args.co_premise_weight,
        ),
    )
    artifact = build_graph_artifact(metadata_path, output_dir, cfg)
    log.info(
        "wrote graph artifact to %s (%s nodes, %s edges)",
        output_dir,
        artifact.edge_stats["num_nodes"],
        artifact.edge_stats["num_edges"],
    )
    if artifact.edge_stats.get("records_with_reference_fields", 0) == 0:
        if artifact.edge_stats.get("text_reference_edges", 0) > 0:
            log.warning(
                "metadata did not contain structured dependency fields; graph uses text-inferred references"
            )
        elif artifact.edge_stats.get("same_module_edge_generation") == "disabled_by_zero_weight":
            log.warning(
                "metadata did not contain structured dependency fields and same_module edge "
                "generation was disabled; graph contains no dependency edges"
            )
        else:
            log.warning(
                "metadata did not contain structured dependency fields; graph contains same_module edges only"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
