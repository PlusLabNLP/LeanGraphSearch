"""Durable per-call API usage telemetry for prove experiments."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable


USAGE_KEYS = (
    "requests",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def usage_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in USAGE_KEYS}


def usage_cost_usd(usage: dict[str, int]) -> float:
    """Gemini 3.1 Pro reference cost used by the frozen experiments."""

    return (
        int(usage.get("input_tokens", 0)) * 2.0
        + int(usage.get("output_tokens", 0)) * 12.0
        + int(usage.get("cache_creation_input_tokens", 0)) * 2.5
        + int(usage.get("cache_read_input_tokens", 0)) * 0.2
    ) / 1_000_000


class DurableUsageRecorder:
    """Append every completed call to JSONL and retain current-process events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._events: dict[str, list[dict[str, Any]]] = {}

    async def append(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("schema_version", 1)
        payload.setdefault("event_id", str(uuid.uuid4()))
        trajectory_id = str(payload["trajectory_id"])
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            self._events.setdefault(trajectory_id, []).append(payload)

    def events_for(self, trajectory_id: str) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events.get(trajectory_id, ())]


class UsageTracingLLMClient:
    """Duck-typed client proxy recording exact response-reported usage deltas."""

    def __init__(
        self,
        client: Any,
        recorder: DurableUsageRecorder,
        *,
        trajectory_id: str,
        problem_id: str,
        mode: str,
        phase: str,
        role: str,
        branch_index: int | None = None,
        round_from_call_index: Callable[[int], int | None] | None = None,
    ) -> None:
        self._client = client
        self._recorder = recorder
        self._context = {
            "trajectory_id": trajectory_id,
            "problem_id": problem_id,
            "mode": mode,
            "phase": phase,
            "role": role,
            "branch_index": branch_index,
        }
        self._round_from_call_index = round_from_call_index
        self._call_index = 0
        for name in ("profile", "provider", "model", "timeout"):
            if hasattr(client, name):
                setattr(self, name, getattr(client, name))

    async def chat(self, *args: Any, **kwargs: Any) -> str:
        self._call_index += 1
        call_index = self._call_index
        before = self._client.usage_snapshot()
        captured_usage: list[dict[str, int]] = []
        observer_token: Any = None
        if hasattr(self._client, "set_usage_observer"):
            observer_token = self._client.set_usage_observer(captured_usage.append)
        started = time.time()
        status = "ok"
        error: str | None = None
        try:
            response = await self._client.chat(*args, **kwargs)
        except asyncio.CancelledError as exc:
            # Budgeted reasoning branches deliberately cancel the slower judge
            # once another branch wins. Preserve the event, but distinguish
            # this zero-usage cancellation from a provider error.
            status = "cancelled"
            error = f"{type(exc).__name__}: {exc}"
            raise
        except BaseException as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ended = time.time()
            after = self._client.usage_snapshot()
            if observer_token is not None:
                self._client.reset_usage_observer(observer_token)
            # A task-local observer is authoritative even when it captured no
            # response.  In particular, an early-stopped concurrent call may
            # be cancelled after sibling calls have advanced the shared
            # client's aggregate snapshot.  Falling back to ``after-before``
            # in that case would attribute those siblings' usage to the
            # cancelled call (often more than once).
            delta = (
                {
                    key: sum(int(item.get(key, 0)) for item in captured_usage)
                    for key in USAGE_KEYS
                }
                if observer_token is not None
                else usage_delta(after, before)
            )
            proof_round = (
                self._round_from_call_index(call_index)
                if self._round_from_call_index is not None
                else None
            )
            await self._recorder.append(
                {
                    **self._context,
                    "role_call_index": call_index,
                    "proof_round": proof_round,
                    "started_at_unix_s": started,
                    "ended_at_unix_s": ended,
                    "elapsed_s": ended - started,
                    "status": status,
                    "error": error,
                    "usage_before": before,
                    "usage_after": after,
                    "usage_delta": delta,
                    "cost_usd": usage_cost_usd(delta),
                }
            )
        return response

    def usage_snapshot(self) -> dict[str, int]:
        return self._client.usage_snapshot()

    async def aclose(self) -> None:
        # Ownership remains with the underlying client list in the runner.
        return None


__all__ = [
    "DurableUsageRecorder",
    "USAGE_KEYS",
    "UsageTracingLLMClient",
    "usage_cost_usd",
    "usage_delta",
]
