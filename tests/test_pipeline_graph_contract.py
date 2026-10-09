from __future__ import annotations

import inspect
import unittest

from fastapi.params import Body

from leansearchv2.pipeline import RetrievalPipeline
from leansearchv2.graph.config import GraphConfig
from leansearchv2 import server


class PipelineGraphContractTests(unittest.TestCase):
    def test_pipeline_search_has_graph_opt_in_parameters(self) -> None:
        params = inspect.signature(RetrievalPipeline.search).parameters

        self.assertIn("graph_augment", params)
        self.assertEqual(params["graph_augment"].default, False)
        self.assertIn("graph_initial_top_n", params)
        self.assertIn("graph_expand_m", params)
        self.assertIn("graph_final_top_k", params)


    def test_graph_config_accepts_documented_alias_fields(self) -> None:
        cfg = GraphConfig.from_dict(
            {
                "graph_artifact_dir": "data/graph/dep_only",
                "graph_seed_top_n": 123,
                "preserve_top_k": 7,
                "ppr_restart_prob": 0.2,
                "ppr_max_iter": 77,
                "ppr_tol": 1.0e-5,
                "rerank_graph_union": True,
                "reranker_required": False,
            }
        )

        self.assertEqual(cfg.initial_top_n, 123)
        self.assertEqual(cfg.preserve_standard_top_k, 7)
        self.assertEqual(cfg.restart_probability, 0.2)
        self.assertEqual(cfg.max_iterations, 77)
        self.assertEqual(cfg.tolerance, 1.0e-5)
        self.assertTrue(cfg.rerank_graph_union)
        self.assertFalse(cfg.reranker_required)
        self.assertTrue(str(cfg.artifact_dir).endswith("data/graph/dep_only"))

    def test_server_search_defaults_defer_graph_mode_to_config(self) -> None:
        params = inspect.signature(server.search).parameters

        self.assertIn("graph_augment", params)
        default = params["graph_augment"].default
        self.assertIsInstance(default, Body)
        self.assertIsNone(default.default)

    def test_build_result_adds_graph_metadata_only_when_supplied(self) -> None:
        pipeline = object.__new__(RetrievalPipeline)
        pipeline.data = [
            {
                "module_name": ["Mathlib", "Tiny"],
                "kind": "theorem",
                "name": ["Alpha"],
                "signature": "",
                "type": "",
            }
        ]

        plain = pipeline._build_result(0, 0.1)
        graph = pipeline._build_result(
            0,
            0.1,
            graph_metadata={
                "graph_score": 0.5,
                "seed_source": "graph_only",
                "graph_evidence": [{"edge_types": ["dep"]}],
            },
        )

        self.assertEqual(set(plain.model_dump().keys()), {"result", "distance"})
        self.assertEqual(graph.model_dump()["graph_score"], 0.5)
        self.assertEqual(graph.model_dump()["seed_source"], "graph_only")


if __name__ == "__main__":
    unittest.main()
