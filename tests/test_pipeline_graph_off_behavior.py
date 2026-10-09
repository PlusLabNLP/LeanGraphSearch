from __future__ import annotations

import unittest

import numpy as np

from leansearchv2.graph.config import GraphConfig
from leansearchv2.pipeline import RetrievalPipeline


class _FakeEmbedder:
    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class PipelineGraphOffBehaviorTests(unittest.TestCase):
    def test_graph_off_search_keeps_standard_rerank_order_and_schema(self) -> None:
        pipeline = object.__new__(RetrievalPipeline)
        pipeline.graph_config = GraphConfig()
        pipeline._graph_artifact = None
        pipeline._graph_artifact_path = None
        pipeline.embedding_model = _FakeEmbedder()
        pipeline.texts = ["doc alpha", "doc beta", "doc gamma"]
        pipeline.data = [
            {"module_name": ["M"], "kind": "theorem", "name": ["Alpha"], "signature": "", "type": ""},
            {"module_name": ["M"], "kind": "theorem", "name": ["Beta"], "signature": "", "type": ""},
            {"module_name": ["M"], "kind": "theorem", "name": ["Gamma"], "signature": "", "type": ""},
        ]

        def fake_search_cuvs(query_embedding: np.ndarray, k: int):
            self.assertEqual(k, 50)
            return (
                np.asarray([[0.3, 0.1, 0.2]], dtype=np.float32),
                np.asarray([[0, 1, 2]], dtype=np.int64),
            )

        def fake_rerank(pairs: list[str], fallback_scores: list[float]) -> list[float]:
            return [0.2, 0.9, 0.1]

        pipeline._search_cuvs = fake_search_cuvs
        pipeline._rerank = fake_rerank

        results = pipeline.search("query", top_k=2, rerank=True, graph_augment=False)

        self.assertEqual([".".join(r.result.name) for r in results], ["Beta", "Alpha"])
        for result in results:
            self.assertEqual(set(result.model_dump().keys()), {"result", "distance"})


if __name__ == "__main__":
    unittest.main()
