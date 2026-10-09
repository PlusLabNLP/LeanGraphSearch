from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from leansearchv2.graph.artifact import (
    GraphArtifact,
    build_graph_artifact,
    load_graph_artifact,
)
from leansearchv2.graph.config import GraphConfig, GraphEdgeWeights
from leansearchv2.graph.ppr import (
    personalized_pagerank,
    personalized_pagerank_sparse,
    top_ppr_nodes,
    transition_matrix_from_adjacency,
)


def _tiny_records() -> list[dict]:
    return [
        {
            "name": ["Alpha"],
            "module_name": ["Mathlib", "Tiny"],
            "kind": "theorem",
            "typeReferences": [["Beta"]],
            "valueReferences": [["Gamma"]],
            "isProp": True,
        },
        {
            "name": ["Beta"],
            "module_name": ["Mathlib", "Tiny"],
            "kind": "definition",
            "typeReferences": [],
            "valueReferences": [],
            "isProp": False,
        },
        {
            "name": ["Gamma"],
            "module_name": ["Mathlib", "Other"],
            "kind": "definition",
            "typeReferences": [],
            "valueReferences": [["Beta"]],
            "isProp": False,
        },
    ]


class GraphCoreTests(unittest.TestCase):
    def test_build_graph_artifact_from_metadata_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "metadata.pkl"
            output_dir = root / "graph"
            with metadata_path.open("wb") as f:
                pickle.dump({"data": _tiny_records()}, f)

            artifact = build_graph_artifact(metadata_path, output_dir, GraphConfig())

            self.assertEqual(artifact.node_names, ["Alpha", "Beta", "Gamma"])
            self.assertIn(1, artifact.adjacency[0])
            self.assertGreater(artifact.adjacency[0][1], artifact.adjacency[1][0])
            self.assertIn(1, artifact.adjacency[2])
            self.assertEqual(artifact.edge_stats["edge_counts"]["dep"], 2)
            self.assertEqual(artifact.edge_stats["edge_counts"]["rev_dep"], 2)
            self.assertEqual(artifact.edge_stats["edge_counts"]["same_module"], 2)
            self.assertTrue((output_dir / "graph_adjacency.npz").exists())
            self.assertTrue((output_dir / "nodes.json").exists())
            self.assertTrue((output_dir / "edge_stats.json").exists())

            saved_stats = json.loads((output_dir / "edge_stats.json").read_text())
            self.assertEqual(saved_stats["num_nodes"], 3)
            self.assertEqual(saved_stats["edge_weights"]["dep"], 1.0)

    def test_load_graph_artifact_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "metadata.pkl"
            output_dir = root / "graph"
            with metadata_path.open("wb") as f:
                pickle.dump({"data": _tiny_records()}, f)

            build_graph_artifact(metadata_path, output_dir, GraphConfig())
            loaded = load_graph_artifact(output_dir)

            self.assertIsInstance(loaded, GraphArtifact)
            self.assertEqual(loaded.name_to_node["Beta"], 1)
            self.assertEqual(loaded.node_records[2]["kind"], "definition")
            self.assertEqual(loaded.out_degree[0], 1)

    def test_load_graph_artifact_uses_utf8_under_ascii_locale(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "metadata.pkl"
            output_dir = root / "graph"
            records = [
                {"name": ["Join_⊔"], "module_name": ["M"], "kind": "definition"},
                {"name": ["Beta"], "module_name": ["M"], "kind": "definition"},
            ]
            with metadata_path.open("wb") as f:
                pickle.dump({"data": records}, f)

            build_graph_artifact(metadata_path, output_dir, GraphConfig())

            env = os.environ.copy()
            env.update(
                {
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                    "LC_ALL": "C",
                    "PYTHONUTF8": "0",
                    "PYTHONCOERCECLOCALE": "0",
                    "PYTHONIOENCODING": "utf-8",
                }
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from leansearchv2.graph.artifact import load_graph_artifact; "
                        f"a=load_graph_artifact({str(output_dir)!r}); "
                        "print(a.node_names[0])"
                    ),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("Join_⊔", proc.stdout)

    def test_zero_same_module_weight_disables_same_module_generation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "metadata.pkl"
            output_dir = root / "graph"
            records = [
                {"name": ["A"], "module_name": ["M"], "kind": "definition"},
                {"name": ["B"], "module_name": ["M"], "kind": "definition"},
                {"name": ["C"], "module_name": ["M"], "kind": "definition"},
            ]
            with metadata_path.open("wb") as f:
                pickle.dump({"data": records}, f)

            cfg = GraphConfig(
                edge_weights=GraphEdgeWeights(dep=1.0, rev_dep=0.7, same_module=0.0)
            )
            artifact = build_graph_artifact(metadata_path, output_dir, cfg)

            self.assertEqual(artifact.edge_stats["num_edges"], 0)
            self.assertEqual(artifact.edge_stats["edge_counts"]["same_module"], 0)
            self.assertEqual(
                artifact.edge_stats["same_module_edge_generation"],
                "disabled_by_zero_weight",
            )

    def test_reverse_dependencies_penalize_hubs_and_prefer_theorems(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "metadata.pkl"
            output_dir = root / "graph"
            records = [
                {
                    "name": ["UseTheorem"],
                    "kind": "theorem",
                    "isProp": True,
                    "typeReferences": [["Hub"]],
                },
                {
                    "name": ["UseDefinition"],
                    "kind": "definition",
                    "isProp": False,
                    "typeReferences": [["Hub"]],
                },
                {
                    "name": ["SoloTheorem"],
                    "kind": "theorem",
                    "isProp": True,
                    "typeReferences": [["Solo"]],
                },
                {"name": ["Hub"], "kind": "definition", "isProp": False},
                {"name": ["Solo"], "kind": "definition", "isProp": False},
            ]
            with metadata_path.open("wb") as f:
                pickle.dump({"data": records}, f)

            artifact = build_graph_artifact(
                metadata_path,
                output_dir,
                GraphConfig(
                    reverse_dependency_hub_penalty="log2",
                    reverse_dependency_theorem_like_weight=1.0,
                    reverse_dependency_non_theorem_weight=0.25,
                    edge_weights=GraphEdgeWeights(
                        dep=1.0,
                        rev_dep=0.3,
                        same_module=0.0,
                    ),
                ),
            )

            hub = artifact.name_to_node["Hub"]
            solo = artifact.name_to_node["Solo"]
            theorem = artifact.name_to_node["UseTheorem"]
            definition = artifact.name_to_node["UseDefinition"]
            solo_theorem = artifact.name_to_node["SoloTheorem"]
            self.assertAlmostEqual(artifact.adjacency[solo][solo_theorem], 0.3)
            self.assertAlmostEqual(
                artifact.adjacency[hub][theorem],
                0.3 / np.log2(3.0),
            )
            self.assertAlmostEqual(
                artifact.adjacency[hub][definition],
                0.25 * artifact.adjacency[hub][theorem],
            )
            policy = artifact.edge_stats["reverse_dependency_policy"]
            self.assertEqual(policy["hub_penalty"], "log2")
            self.assertEqual(policy["theorem_like_edge_count"], 2)
            self.assertEqual(policy["non_theorem_edge_count"], 1)

    def test_build_graph_artifact_uses_additional_dependency_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "metadata.pkl"
            output_dir = root / "graph"
            records = [
                {"name": ["Alpha"], "module_name": ["M"], "dependencies": [["Beta"]]},
                {"name": ["Beta"], "module_name": ["M"]},
                {"name": ["Gamma"], "module_name": ["M"], "references": [["Beta"]]},
            ]
            with metadata_path.open("wb") as f:
                pickle.dump({"data": records}, f)

            cfg = GraphConfig(
                edge_weights=GraphEdgeWeights(dep=1.0, rev_dep=0.0, same_module=0.0)
            )
            artifact = build_graph_artifact(metadata_path, output_dir, cfg)

            self.assertIn(1, artifact.adjacency[0])
            self.assertIn(1, artifact.adjacency[2])
            self.assertEqual(artifact.edge_stats["edge_counts"]["dep"], 2)
            self.assertEqual(artifact.edge_stats["dependency_field_counts"]["dependencies"], 1)
            self.assertEqual(artifact.edge_stats["dependency_field_counts"]["references"], 1)

    def test_infer_text_references_adds_dependency_edges_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "metadata.pkl"
            output_dir = root / "graph"
            records = [
                {
                    "name": ["Alpha"],
                    "module_name": ["M"],
                    "kind": "theorem",
                    "type": "Beta -> Ns.Gamma -> Prop",
                },
                {"name": ["Beta"], "module_name": ["M"], "kind": "definition"},
                {"name": ["Ns", "Gamma"], "module_name": ["M"], "kind": "definition"},
            ]
            with metadata_path.open("wb") as f:
                pickle.dump({"data": records}, f)

            disabled = build_graph_artifact(
                metadata_path,
                output_dir / "disabled",
                GraphConfig(edge_weights=GraphEdgeWeights(same_module=0.0)),
            )
            enabled = build_graph_artifact(
                metadata_path,
                output_dir / "enabled",
                GraphConfig(
                    infer_text_references=True,
                    edge_weights=GraphEdgeWeights(dep=1.0, rev_dep=0.0, same_module=0.0),
                ),
            )

            self.assertEqual(disabled.edge_stats["edge_counts"]["dep"], 0)
            self.assertIn(1, enabled.adjacency[0])
            self.assertIn(2, enabled.adjacency[0])
            self.assertEqual(enabled.edge_stats["edge_counts"]["dep"], 2)
            self.assertEqual(enabled.edge_stats["text_reference_edges"], 2)
            self.assertEqual(enabled.edge_stats["records_with_text_references"], 1)
            self.assertEqual(enabled.edge_stats["structured_dependency_edge_count"], 0)
            self.assertEqual(enabled.edge_stats["text_inferred_edge_count"], 2)

    def test_build_graph_artifact_from_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records_path = root / "records.jsonl"
            output_dir = root / "graph"
            records = [
                {"name": "Alpha", "kind": "theorem", "typeReferences": ["Beta"], "isProp": True},
                {"name": "Beta", "kind": "def", "valueReferences": []},
            ]
            records_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

            artifact = build_graph_artifact(
                records_path,
                output_dir,
                GraphConfig(edge_weights=GraphEdgeWeights(dep=1.0, rev_dep=0.7, same_module=0.0)),
            )

            self.assertEqual(artifact.node_names, ["Alpha", "Beta"])
            self.assertIn(1, artifact.adjacency[0])
            self.assertEqual(artifact.edge_stats["structured_dependency_edge_count"], 1)
            self.assertEqual(artifact.edge_stats["text_inferred_edge_count"], 0)
            self.assertEqual(artifact.edge_stats["reverse_edge_count"], 1)

    def test_build_graph_artifact_records_dependency_source_label(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records_path = root / "records.json"
            output_dir = root / "graph"
            records_path.write_text(
                json.dumps(
                    [
                        {
                            "name": "Alpha",
                            "kind": "theorem",
                            "typeReferences": ["Beta"],
                            "isProp": True,
                        },
                        {"name": "Beta", "kind": "def"},
                    ]
                )
            )

            artifact = build_graph_artifact(
                records_path,
                output_dir,
                GraphConfig(
                    dependency_source="structured_sym_side_artifact",
                    edge_weights=GraphEdgeWeights(dep=1.0, rev_dep=0.7, same_module=0.0),
                ),
            )

            self.assertEqual(artifact.edge_stats["dependency_source"], "structured_sym_side_artifact")
            saved_stats = json.loads((output_dir / "edge_stats.json").read_text())
            self.assertEqual(saved_stats["dependency_source"], "structured_sym_side_artifact")

    def test_structured_dependency_edge_count_tracks_actual_forward_edges(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records_path = root / "records.jsonl"
            output_dir = root / "graph"
            records = [
                {"name": "Alpha", "kind": "def", "typeReferences": ["Alpha", "Beta"]},
                {"name": "Beta", "kind": "def"},
            ]
            records_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

            artifact = build_graph_artifact(
                records_path,
                output_dir,
                GraphConfig(edge_weights=GraphEdgeWeights(dep=1.0, rev_dep=0.0, same_module=0.0)),
            )

            self.assertEqual(artifact.edge_stats["edge_counts"]["dep"], 1)
            self.assertEqual(artifact.edge_stats["structured_dependency_edge_count"], 1)

    def test_structured_edge_stats_are_split_by_reference_field(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records_path = root / "records.jsonl"
            output_dir = root / "graph"
            records = [
                {
                    "name": "Alpha",
                    "kind": "def",
                    "typeReferences": ["Beta"],
                    "valueReferences": ["Beta", "Gamma"],
                },
                {"name": "Beta", "kind": "def"},
                {"name": "Gamma", "kind": "def"},
            ]
            records_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

            artifact = build_graph_artifact(
                records_path,
                output_dir,
                GraphConfig(edge_weights=GraphEdgeWeights(dep=1.0, rev_dep=0.0, same_module=0.0)),
            )

            self.assertEqual(artifact.edge_stats["structured_dependency_edge_count"], 2)
            self.assertEqual(artifact.edge_stats["type_reference_edge_count"], 1)
            self.assertEqual(artifact.edge_stats["value_reference_edge_count"], 1)
            self.assertEqual(artifact.edge_stats["dependency_field_edge_counts"]["typeReferences"], 1)
            self.assertEqual(artifact.edge_stats["dependency_field_edge_counts"]["valueReferences"], 1)


    def test_capped_same_module_edges_do_not_build_full_clique(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records_path = root / "records.jsonl"
            output_dir = root / "graph"
            records = [
                {"name": f"Decl{i}", "kind": "def", "module_name": "M"}
                for i in range(5)
            ]
            records_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

            artifact = build_graph_artifact(
                records_path,
                output_dir,
                GraphConfig(
                    same_module_max_neighbors_per_node=2,
                    edge_weights=GraphEdgeWeights(dep=0.0, rev_dep=0.0, same_module=0.05),
                ),
            )

            self.assertEqual(artifact.edge_stats["same_module_edge_generation"], "capped")
            self.assertEqual(artifact.edge_stats["same_module_max_neighbors_per_node"], 2)
            self.assertEqual(artifact.edge_stats["same_module_edge_count"], 10)
            self.assertLess(artifact.edge_stats["same_module_edge_count"], 20)
            self.assertTrue(all(len(neighbors) <= 2 for neighbors in artifact.adjacency))

    def test_co_premise_edges_use_value_references_and_filter_hubs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records_path = root / "records.jsonl"
            output_dir = root / "graph"
            records = [
                {
                    "name": "Thm1",
                    "kind": "theorem",
                    "isProp": True,
                    "valueReferences": ["A", "B", "Hub"],
                },
                {
                    "name": "Thm2",
                    "kind": "theorem",
                    "isProp": True,
                    "valueReferences": ["C", "D", "Hub"],
                },
                {"name": "A", "kind": "def"},
                {"name": "B", "kind": "def"},
                {"name": "C", "kind": "def"},
                {"name": "D", "kind": "def"},
                {"name": "Hub", "kind": "def"},
            ]
            records_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

            artifact = build_graph_artifact(
                records_path,
                output_dir,
                GraphConfig(
                    enable_co_premise_edges=True,
                    co_premise_max_ref_frequency=1,
                    co_premise_max_refs_per_source=64,
                    edge_weights=GraphEdgeWeights(
                        dep=0.0, rev_dep=0.0, same_module=0.0, co_premise=0.05
                    ),
                ),
            )

            self.assertEqual(artifact.edge_stats["co_premise_edge_generation"], "enabled")
            self.assertEqual(artifact.edge_stats["co_premise_edge_count"], 4)
            self.assertEqual(artifact.edge_stats["co_premise_hub_reference_count"], 1)
            self.assertIn(artifact.name_to_node["B"], artifact.adjacency[artifact.name_to_node["A"]])
            self.assertIn(artifact.name_to_node["A"], artifact.adjacency[artifact.name_to_node["B"]])
            hub = artifact.name_to_node["Hub"]
            self.assertFalse(any(hub in neighbors for neighbors in artifact.adjacency))

    def test_theorem_value_references_are_opt_in_direct_edges_with_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records_path = root / "records.jsonl"
            records = [
                {
                    "name": "AllowedTheorem",
                    "kind": "theorem",
                    "isProp": True,
                    "typeReferences": ["TypeOnly"],
                    "valueReferences": ["ProofOnly", "TypeOnly"],
                },
                {
                    "name": "HeldOutTarget",
                    "kind": "lemma",
                    "isProp": True,
                    "valueReferences": ["ProofOnly"],
                },
                {"name": "TypeOnly", "kind": "definition"},
                {"name": "ProofOnly", "kind": "theorem", "isProp": True},
            ]
            records_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            baseline = build_graph_artifact(
                records_path,
                root / "baseline",
                GraphConfig(
                    edge_weights=GraphEdgeWeights(
                        dep=1.0, rev_dep=0.0, same_module=0.0
                    )
                ),
            )
            variant = build_graph_artifact(
                records_path,
                root / "variant",
                GraphConfig(
                    include_theorem_value_references=True,
                    theorem_value_reference_weight=0.5,
                    excluded_proof_dependency_source_names=("HeldOutTarget", "Absent"),
                    edge_weights=GraphEdgeWeights(
                        dep=1.0, rev_dep=0.0, same_module=0.0
                    ),
                ),
            )

            allowed = variant.name_to_node["AllowedTheorem"]
            held_out = variant.name_to_node["HeldOutTarget"]
            proof_only = variant.name_to_node["ProofOnly"]
            self.assertNotIn(proof_only, baseline.adjacency[allowed])
            self.assertAlmostEqual(variant.adjacency[allowed][proof_only], 0.5)
            self.assertNotIn(proof_only, variant.adjacency[held_out])
            self.assertEqual(variant.edge_stats["edge_counts"]["proof_dep"], 1)
            policy = variant.edge_stats["proof_dependency_policy"]
            self.assertEqual(policy["candidate_source_count"], 2)
            self.assertEqual(policy["excluded_source_name_count_requested"], 2)
            self.assertEqual(policy["excluded_source_name_count_matched"], 1)
            self.assertEqual(policy["excluded_reference_item_count"], 1)
            self.assertEqual(policy["excluded_source_names_absent"], ["Absent"])

    def test_sparse_pagerank_matches_list_pagerank(self) -> None:
        adjacency = [{1: 2.0, 2: 1.0}, {2: 1.0}, {}]
        transition, dangling = transition_matrix_from_adjacency(adjacency)

        dense_scores = personalized_pagerank(
            adjacency,
            {0: 1.0},
            restart_probability=0.2,
            max_iterations=100,
            tolerance=1e-12,
        )
        sparse_scores = personalized_pagerank_sparse(
            transition,
            dangling,
            {0: 1.0},
            restart_probability=0.2,
            max_iterations=100,
            tolerance=1e-12,
        )

        np.testing.assert_allclose(sparse_scores, dense_scores, rtol=1e-12, atol=1e-12)

    def test_personalized_pagerank_is_deterministic_and_prefers_reachable_nodes(self) -> None:
        adjacency = [{1: 1.0}, {2: 1.0}, {}]

        scores_a = personalized_pagerank(
            adjacency,
            {0: 1.0},
            restart_probability=0.2,
            max_iterations=100,
            tolerance=1e-12,
        )
        scores_b = personalized_pagerank(
            adjacency,
            {0: 1.0},
            restart_probability=0.2,
            max_iterations=100,
            tolerance=1e-12,
        )

        np.testing.assert_allclose(scores_a, scores_b)
        self.assertAlmostEqual(float(scores_a.sum()), 1.0, places=9)
        self.assertGreater(scores_a[1], scores_a[2])
        self.assertEqual(top_ppr_nodes(scores_a, limit=2, exclude={0})[0][0], 1)

    def test_config_edge_weights_are_used(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "metadata.pkl"
            output_dir = root / "graph"
            with metadata_path.open("wb") as f:
                pickle.dump({"data": _tiny_records()}, f)

            cfg = GraphConfig(
                edge_weights=GraphEdgeWeights(dep=2.0, rev_dep=0.25, same_module=0.01)
            )
            artifact = build_graph_artifact(metadata_path, output_dir, cfg)

            self.assertAlmostEqual(artifact.adjacency[0][1], 2.01)
            self.assertAlmostEqual(artifact.adjacency[1][0], 0.26)


if __name__ == "__main__":
    unittest.main()
