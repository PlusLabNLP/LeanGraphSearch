from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from leansearchv2.graph.build import main as graph_build_main


class GraphBuildCliTests(unittest.TestCase):
    def test_build_cli_writes_graph_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata_path = root / "metadata.pkl"
            output_dir = root / "graph"
            with metadata_path.open("wb") as f:
                pickle.dump(
                    {
                        "data": [
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
                    },
                    f,
                )

            code = graph_build_main(
                [
                    "--metadata-pkl",
                    str(metadata_path),
                    "--output-dir",
                    str(output_dir),
                    "--dep-weight",
                    "1.5",
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((output_dir / "graph_adjacency.npz").exists())
            self.assertTrue((output_dir / "nodes.json").exists())
            self.assertTrue((output_dir / "edge_stats.json").exists())


if __name__ == "__main__":
    unittest.main()
