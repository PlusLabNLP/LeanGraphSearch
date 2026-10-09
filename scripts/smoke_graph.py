#!/usr/bin/env python3
"""Exercise dependency graph construction, PageRank, and baseline preservation on synthetic data."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from leansearchv2.graph.artifact import build_graph_artifact, load_graph_artifact
from leansearchv2.graph.config import GraphConfig, GraphEdgeWeights
from leansearchv2.graph.ppr import personalized_pagerank
from leansearchv2.graph.augment import CandidateScore, augment_candidates


def main() -> None:
    records = [
        {"name": ["Example", "A"], "kind": "theorem", "isProp": True, "typeReferences": [["Example", "B"]]},
        {"name": ["Example", "B"], "kind": "definition", "isProp": False, "typeReferences": []},
    ]
    with tempfile.TemporaryDirectory(prefix="lgs-smoke-") as temp:
        root = Path(temp)
        source = root / "records.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in records))
        cfg = GraphConfig(edge_weights=GraphEdgeWeights(dep=1.0, rev_dep=0.0, same_module=0.0))
        graph = build_graph_artifact(source, root / "graph", cfg)
        restored = load_graph_artifact(root / "graph")
        assert graph.node_names == restored.node_names == ["Example.A", "Example.B"]
        assert graph.edge_stats["edge_counts"]["dep"] == 1
        scores = personalized_pagerank(graph.adjacency, {0: 1.0})
        assert scores[1] > 0 and abs(float(scores.sum()) - 1.0) < 1e-10
        baseline = [CandidateScore(index=0, standard_score=1.0, distance=0.0)]
        assert augment_candidates(baseline, artifact=None, config=GraphConfig(graph_augment=False), final_top_k=1) == baseline
    print("PASS: synthetic graph, PageRank propagation, and graph-off preservation.")


if __name__ == "__main__":
    main()
