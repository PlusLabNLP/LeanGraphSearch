"""Proof-stage call budgets for modified Table-3 prove runs.

The A2-optional-40 protocol counts model, retrieval-action, and compiler
calls independently.  This module deliberately contains no policy about
which action to choose; it only provides concurrency-safe accounting and a
thin LLM proxy for the proof-stage prover and standard query generator.
Reasoning-mode sketch/filter/judge calls are a separate retrieval-preparation
budget and must not be wrapped by this module.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


class BudgetExhausted(RuntimeError):
    """Raised when an attempted action would exceed its frozen limit."""


@dataclass
class A2CallBudget:
    max_model_calls: int = 40
    max_query_calls: int = 8
    max_compiler_calls: int = 32
    model_calls: int = 0
    query_calls: int = 0
    compiler_calls: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def _reserve(self, field_name: str, limit: int, label: str) -> int:
        async with self._lock:
            current = int(getattr(self, field_name))
            if current >= limit:
                raise BudgetExhausted(f"{label} budget exhausted ({current}/{limit})")
            current += 1
            setattr(self, field_name, current)
            return current

    async def reserve_model(self) -> int:
        return await self._reserve("model_calls", self.max_model_calls, "model-call")

    async def reserve_query(self) -> int:
        return await self._reserve("query_calls", self.max_query_calls, "query-action")

    async def reserve_compiler(self) -> int:
        return await self._reserve(
            "compiler_calls", self.max_compiler_calls, "compiler-call"
        )

    def remaining_model(self) -> int:
        return max(0, self.max_model_calls - self.model_calls)

    def remaining_query(self) -> int:
        return max(0, self.max_query_calls - self.query_calls)

    def remaining_compiler(self) -> int:
        return max(0, self.max_compiler_calls - self.compiler_calls)

    def snapshot(self) -> dict[str, int]:
        return {
            "max_model_calls": self.max_model_calls,
            "max_query_calls": self.max_query_calls,
            "max_compiler_calls": self.max_compiler_calls,
            "model_calls": self.model_calls,
            "query_calls": self.query_calls,
            "compiler_calls": self.compiler_calls,
        }


class BudgetedLLMClient:
    """Duck-typed LLMClient proxy sharing one :class:`A2CallBudget`."""

    def __init__(self, client: Any, budget: A2CallBudget) -> None:
        self._client = client
        self._budget = budget
        for name in ("profile", "provider", "model", "timeout"):
            if hasattr(client, name):
                setattr(self, name, getattr(client, name))

    async def chat(self, *args: Any, **kwargs: Any) -> str:
        await self._budget.reserve_model()
        return await self._client.chat(*args, **kwargs)

    def usage_snapshot(self) -> dict[str, int]:
        return self._client.usage_snapshot()

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["A2CallBudget", "BudgetedLLMClient", "BudgetExhausted"]
