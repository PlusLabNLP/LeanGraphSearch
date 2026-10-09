"""HTTP client for the standard-mode retriever endpoint.

Reasoning mode and the prove task consume the retriever exclusively through
this client; nothing here imports `pipeline.py` (which pulls in cuVS/torch),
so the client is CPU-only and can run anywhere with network access.
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from .config import get


class ResultData(BaseModel):
    module_name: list[str]
    kind: str
    name: list[str]
    signature: str
    type: str
    value: str | None = None
    docstring: str | None = None
    informal_name: str | None = None
    informal_description: str | None = None


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    result: ResultData
    distance: float


class StandardClient:
    def __init__(self, url: str | None = None, timeout: float | None = None) -> None:
        url = url or get("RETRIEVER_URL", "retriever", "url", default="http://localhost:8000")
        self.url = url.rstrip("/")
        self.timeout = float(
            timeout if timeout is not None else get("RETRIEVER_TIMEOUT_S", "retriever", "timeout_s", default=60)
        )

    async def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        rerank: bool = True,
        retrieve_k: int | None = None,
        graph_augment: bool = False,
        graph_initial_top_n: int | None = None,
        graph_expand_m: int | None = None,
        graph_final_top_k: int | None = None,
        rank_fusion: str | None = None,
        return_metadata: bool | None = None,
    ) -> list[SearchResult]:
        results = await self.search_batch(
            [query],
            top_k,
            rerank=rerank,
            retrieve_k=retrieve_k,
            graph_augment=graph_augment,
            graph_initial_top_n=graph_initial_top_n,
            graph_expand_m=graph_expand_m,
            graph_final_top_k=graph_final_top_k,
            rank_fusion=rank_fusion,
            return_metadata=return_metadata,
        )
        return results[0]

    async def search_batch(
        self,
        queries: list[str],
        top_k: int = 10,
        *,
        rerank: bool = True,
        retrieve_k: int | None = None,
        graph_augment: bool = False,
        graph_initial_top_n: int | None = None,
        graph_expand_m: int | None = None,
        graph_final_top_k: int | None = None,
        rank_fusion: str | None = None,
        return_metadata: bool | None = None,
    ) -> list[list[SearchResult]]:
        # Always send the treatment bit.  Omitting false would let a server
        # started from a Graph-enabled config silently apply its own default,
        # contaminating Standard-vs-Graph experiments that share one service.
        body: dict[str, Any] = {
            "query": queries,
            "num_results": top_k,
            "rerank": rerank,
            "graph_augment": bool(graph_augment),
        }
        if retrieve_k is not None:
            body["retrieve_k"] = retrieve_k
        if return_metadata is not None:
            body["return_metadata"] = return_metadata
        if graph_augment:
            if graph_initial_top_n is not None:
                body["graph_initial_top_n"] = graph_initial_top_n
            if graph_expand_m is not None:
                body["graph_expand_m"] = graph_expand_m
            if graph_final_top_k is not None:
                body["graph_final_top_k"] = graph_final_top_k
            if rank_fusion is not None:
                body["rank_fusion"] = rank_fusion
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.url}/search", json=body)
            r.raise_for_status()
            data = r.json()
        return [[SearchResult.model_validate(item) for item in batch] for batch in data]

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(f"{self.url}/health")
            r.raise_for_status()
            return r.json()

    async def graph_expand(
        self, *, seeds: list[str], intent: str, max_hops: int, top_k: int
    ) -> dict[str, Any]:
        body = {
            "seeds": seeds,
            "intent": intent,
            "max_hops": max_hops,
            "top_k": top_k,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.url}/graph_expand", json=body)
            response.raise_for_status()
            return dict(response.json())
