from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from leansearchv2.graph.artifact import (
    GraphArtifactError,
    build_graph_artifact,
    load_graph_artifact,
)
from leansearchv2.graph.augment import CandidateScore, augment_candidates
from leansearchv2.graph.config import GraphConfig
from leansearchv2.standard_client import ResultData, SearchResult
from leansearchv2 import server


def _result_data(name: str) -> ResultData:
    return ResultData(
        module_name=["Mathlib", "Tiny"],
        kind="theorem",
        name=[name],
        signature="",
        type="",
    )


class StandardGraphIntegrationTests(unittest.TestCase):
    def test_server_omitted_graph_flag_uses_config_default(self) -> None:
        pipeline = SimpleNamespace(graph_config=GraphConfig(graph_augment=True))
        with patch.object(server.app, "pipeline", pipeline, create=True):
            self.assertTrue(server._effective_graph_augment(None))

    def test_server_explicit_false_overrides_graph_config_default(self) -> None:
        pipeline = SimpleNamespace(graph_config=GraphConfig(graph_augment=True))
        with patch.object(server.app, "pipeline", pipeline, create=True):
            self.assertFalse(server._effective_graph_augment(False))

    def test_server_standard_config_remains_off_when_flag_is_omitted(self) -> None:
        pipeline = SimpleNamespace(graph_config=GraphConfig(graph_augment=False))
        with patch.object(server.app, "pipeline", pipeline, create=True):
            self.assertFalse(server._effective_graph_augment(None))

    def test_search_result_keeps_graph_off_schema_unchanged(self) -> None:
        plain = SearchResult(result=_result_data("Alpha"), distance=0.1)

        self.assertEqual(set(plain.model_dump().keys()), {"result", "distance"})

    def test_search_result_allows_graph_metadata_when_present(self) -> None:
        graph = SearchResult(
            result=_result_data("Beta"),
            distance=0.2,
            graph_score=0.72,
            seed_source="graph_only",
            graph_evidence=[{"source": "Alpha", "target": "Beta", "edge_types": ["dep"]}],
        )

        dumped = graph.model_dump()
        self.assertEqual(dumped["graph_score"], 0.72)
        self.assertEqual(dumped["seed_source"], "graph_only")
        self.assertEqual(dumped["graph_evidence"][0]["edge_types"], ["dep"])

    def test_missing_graph_artifact_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing_graph"

            with self.assertRaisesRegex(GraphArtifactError, "build.*graph"):
                load_graph_artifact(missing)

    def test_graph_off_candidate_postprocess_is_unchanged(self) -> None:
        original = [
            CandidateScore(index=0, standard_score=0.9, distance=0.1),
            CandidateScore(index=1, standard_score=0.4, distance=0.2),
        ]

        returned = augment_candidates(
            original,
            artifact=None,
            config=GraphConfig(graph_augment=False),
            final_top_k=2,
        )

        self.assertEqual(returned, original)

    def test_graph_expansion_can_introduce_graph_only_candidates(self) -> None:
        records = [
            {
                "name": ["Alpha"],
                "module_name": ["Mathlib", "Tiny"],
                "kind": "theorem",
                "typeReferences": [["Beta"]],
                "isProp": True,
            },
            {
                "name": ["Beta"],
                "module_name": ["Mathlib", "Tiny"],
                "kind": "theorem",
                "typeReferences": [],
                "isProp": True,
            },
            {
                "name": ["Gamma"],
                "module_name": ["Mathlib", "Other"],
                "kind": "theorem",
                "typeReferences": [],
                "isProp": True,
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "metadata.pkl"
            with metadata_path.open("wb") as f:
                pickle.dump({"data": records}, f)
            artifact = build_graph_artifact(metadata_path, root / "graph", GraphConfig())

            augmented = augment_candidates(
                [CandidateScore(index=0, standard_score=1.0, distance=0.01)],
                artifact=artifact,
                config=GraphConfig(graph_augment=True, graph_expand_m=2, beta=5.0, gamma=0.0),
                final_top_k=2,
            )

            self.assertIn(1, [c.index for c in augmented])
            graph_only = next(c for c in augmented if c.index == 1)
            self.assertEqual(graph_only.seed_source, "graph_only")
            self.assertGreater(graph_only.graph_score or 0.0, 0.0)

    def test_preserve_standard_top_k_keeps_reranker_head_before_graph_only(self) -> None:
        records = [
            {
                "name": ["Alpha"],
                "module_name": ["Mathlib", "Tiny"],
                "kind": "theorem",
                "typeReferences": [["Beta"]],
                "isProp": True,
            },
            {
                "name": ["Beta"],
                "module_name": ["Mathlib", "Tiny"],
                "kind": "theorem",
                "typeReferences": [],
                "isProp": True,
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "metadata.pkl"
            with metadata_path.open("wb") as f:
                pickle.dump({"data": records}, f)
            artifact = build_graph_artifact(metadata_path, root / "graph", GraphConfig())

            augmented = augment_candidates(
                [CandidateScore(index=0, standard_score=1.0, distance=0.01)],
                artifact=artifact,
                config=GraphConfig(
                    graph_augment=True,
                    graph_expand_m=2,
                    beta=100.0,
                    gamma=0.0,
                    preserve_standard_top_k=1,
                ),
                final_top_k=2,
            )

            self.assertEqual(augmented[0].index, 0)
            self.assertEqual(augmented[0].seed_source, "standard")
            self.assertEqual(augmented[1].index, 1)
            self.assertEqual(augmented[1].seed_source, "graph_only")

    def test_graph_only_slots_after_preserved_head_move_graph_candidates_up(self) -> None:
        records = [
            {
                "name": ["Alpha"],
                "module_name": ["Mathlib", "Tiny"],
                "kind": "theorem",
                "typeReferences": [["Beta"]],
                "isProp": True,
            },
            {"name": ["Beta"], "module_name": ["Mathlib", "Tiny"], "kind": "theorem"},
            {"name": ["Gamma"], "module_name": ["Mathlib", "Other"], "kind": "theorem"},
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "metadata.pkl"
            with metadata_path.open("wb") as f:
                pickle.dump({"data": records}, f)
            artifact = build_graph_artifact(metadata_path, root / "graph", GraphConfig())

            augmented = augment_candidates(
                [
                    CandidateScore(index=0, standard_score=1.0, distance=0.01),
                    CandidateScore(index=2, standard_score=0.9, distance=0.02),
                ],
                artifact=artifact,
                config=GraphConfig(
                    graph_augment=True,
                    graph_expand_m=2,
                    beta=10.0,
                    gamma=0.0,
                    preserve_standard_top_k=1,
                    graph_only_slots_after_preserved=1,
                ),
                final_top_k=3,
            )

            self.assertEqual([candidate.index for candidate in augmented], [0, 1, 2])
            self.assertEqual(augmented[1].seed_source, "graph_only")


if __name__ == "__main__":
    unittest.main()
