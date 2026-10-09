from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from leansearchv2.graph.artifact import GraphArtifactError
from leansearchv2.graph.config import GraphConfig
from leansearchv2.pipeline import RetrievalPipeline


class _FakeEmbedder:
    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class PipelineMissingGraphTests(unittest.TestCase):
    def test_graph_enabled_search_missing_artifact_has_build_hint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipeline = object.__new__(RetrievalPipeline)
            pipeline.vectordb_dir = Path(td)
            pipeline.graph_config = GraphConfig()
            pipeline._graph_artifact = None
            pipeline._graph_artifact_path = None
            pipeline.embedding_model = _FakeEmbedder()
            pipeline.texts = ["doc alpha"]
            pipeline.data = [
                {"module_name": ["M"], "kind": "theorem", "name": ["Alpha"], "signature": "", "type": ""}
            ]

            def fake_search_cuvs(query_embedding: np.ndarray, k: int):
                return (
                    np.asarray([[0.1]], dtype=np.float32),
                    np.asarray([[0]], dtype=np.int64),
                )

            pipeline._search_cuvs = fake_search_cuvs

            with self.assertRaisesRegex(GraphArtifactError, "python -m leansearchv2.graph.build"):
                pipeline.search("query", top_k=1, rerank=False, graph_augment=True)


if __name__ == "__main__":
    unittest.main()
