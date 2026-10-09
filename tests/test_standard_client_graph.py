from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from leansearchv2.standard_client import StandardClient


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[list[dict]]:
        return [
            [
                {
                    "result": {
                        "module_name": ["Mathlib", "Tiny"],
                        "kind": "theorem",
                        "name": ["Alpha"],
                        "signature": "",
                        "type": "",
                    },
                    "distance": 0.1,
                }
            ]
        ]


class _FakeAsyncClient:
    last_json: dict | None = None

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, json: dict) -> _FakeResponse:
        self.__class__.last_json = json
        return _FakeResponse()


class StandardClientGraphTests(unittest.TestCase):
    def test_default_request_explicitly_disables_graph_augmentation(self) -> None:
        async def run() -> None:
            client = StandardClient(url="http://example.test", timeout=1)
            await client.search_batch(["query"], top_k=5)

        with patch("leansearchv2.standard_client.httpx.AsyncClient", _FakeAsyncClient):
            asyncio.run(run())

        self.assertEqual(
            _FakeAsyncClient.last_json,
            {
                "query": ["query"],
                "num_results": 5,
                "rerank": True,
                "graph_augment": False,
            },
        )

    def test_graph_request_body_includes_opt_in_fields(self) -> None:
        async def run() -> None:
            client = StandardClient(url="http://example.test", timeout=1)
            await client.search_batch(
                ["query"],
                top_k=5,
                graph_augment=True,
                graph_initial_top_n=7,
                graph_expand_m=11,
                graph_final_top_k=3,
            )

        with patch("leansearchv2.standard_client.httpx.AsyncClient", _FakeAsyncClient):
            asyncio.run(run())

        self.assertEqual(_FakeAsyncClient.last_json["graph_augment"], True)
        self.assertEqual(_FakeAsyncClient.last_json["graph_initial_top_n"], 7)
        self.assertEqual(_FakeAsyncClient.last_json["graph_expand_m"], 11)
        self.assertEqual(_FakeAsyncClient.last_json["graph_final_top_k"], 3)


if __name__ == "__main__":
    unittest.main()
