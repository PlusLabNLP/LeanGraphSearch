from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar

import pytest

from leansearchv2.prove.telemetry import DurableUsageRecorder, UsageTracingLLMClient


class _FakeClient:
    profile = "fake"
    provider = "gemini_vertex"
    model = "fake-model"
    timeout = 10.0

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.usage = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    def usage_snapshot(self):
        return dict(self.usage)

    async def chat(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("provider failed")
        self.usage["requests"] += 1
        self.usage["input_tokens"] += 100
        self.usage["output_tokens"] += 25
        return "ok"


class _CancelledClient(_FakeClient):
    async def chat(self, *args, **kwargs):
        raise asyncio.CancelledError()


class _ConcurrentObservedClient(_FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.observer = ContextVar("test_usage_observer", default=None)

    def set_usage_observer(self, observer):
        return self.observer.set(observer)

    def reset_usage_observer(self, token):
        self.observer.reset(token)

    async def chat(self, tokens: int):
        await asyncio.sleep(0.01 if tokens == 100 else 0.02)
        usage = {
            "requests": 1,
            "input_tokens": tokens,
            "output_tokens": tokens // 10,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        for key, value in usage.items():
            self.usage[key] += value
        observer = self.observer.get()
        if observer is not None:
            observer(usage)
        return "ok"


def test_records_exact_usage_cost_and_round(tmp_path):
    async def run():
        recorder = DurableUsageRecorder(tmp_path / "calls.jsonl")
        traced = UsageTracingLLMClient(
            _FakeClient(),
            recorder,
            trajectory_id="trajectory-1",
            problem_id="Problem_1",
            mode="standard",
            phase="proof_stage",
            role="prover",
            round_from_call_index=lambda index: index - 1,
        )
        assert await traced.chat([]) == "ok"
        assert await traced.chat([]) == "ok"
        return recorder

    recorder = asyncio.run(run())

    events = recorder.events_for("trajectory-1")
    assert [event["proof_round"] for event in events] == [0, 1]
    assert all(event["usage_delta"]["requests"] == 1 for event in events)
    assert all(event["cost_usd"] == pytest.approx((100 * 2 + 25 * 12) / 1_000_000) for event in events)
    persisted = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [event["event_id"] for event in persisted] == [event["event_id"] for event in events]


def test_error_is_durably_recorded_without_fabricated_usage(tmp_path):
    async def run():
        recorder = DurableUsageRecorder(tmp_path / "calls.jsonl")
        traced = UsageTracingLLMClient(
            _FakeClient(fail=True),
            recorder,
            trajectory_id="trajectory-error",
            problem_id="Problem_2",
            mode="none",
            phase="proof_stage",
            role="prover",
        )
        with pytest.raises(RuntimeError, match="provider failed"):
            await traced.chat([])
        return recorder

    recorder = asyncio.run(run())
    event = recorder.events_for("trajectory-error")[0]
    assert event["status"] == "error"
    assert event["usage_delta"]["requests"] == 0
    assert event["cost_usd"] == 0.0


def test_cooperative_cancellation_is_distinct_from_provider_error(tmp_path):
    async def run():
        recorder = DurableUsageRecorder(tmp_path / "calls.jsonl")
        traced = UsageTracingLLMClient(
            _CancelledClient(),
            recorder,
            trajectory_id="trajectory-cancelled",
            problem_id="Problem_3",
            mode="reasoning_adaptive",
            phase="reasoning_prefetch",
            role="judge",
            branch_index=1,
        )
        with pytest.raises(asyncio.CancelledError):
            await traced.chat([])
        return recorder

    event = asyncio.run(run()).events_for("trajectory-cancelled")[0]
    assert event["status"] == "cancelled"
    assert event["usage_delta"]["requests"] == 0
    assert event["cost_usd"] == 0.0


def test_task_local_observer_avoids_concurrent_snapshot_double_counting(tmp_path):
    async def run():
        recorder = DurableUsageRecorder(tmp_path / "calls.jsonl")
        client = _ConcurrentObservedClient()
        left = UsageTracingLLMClient(
            client,
            recorder,
            trajectory_id="left",
            problem_id="Problem_4",
            mode="reasoning_adaptive",
            phase="reasoning_prefetch",
            role="filter",
            branch_index=0,
        )
        right = UsageTracingLLMClient(
            client,
            recorder,
            trajectory_id="right",
            problem_id="Problem_4",
            mode="reasoning_adaptive",
            phase="reasoning_prefetch",
            role="filter",
            branch_index=0,
        )
        await asyncio.gather(left.chat(100), right.chat(200))
        return recorder

    recorder = asyncio.run(run())
    assert recorder.events_for("left")[0]["usage_delta"]["input_tokens"] == 100
    assert recorder.events_for("right")[0]["usage_delta"]["input_tokens"] == 200


def test_cancelled_observed_call_does_not_inherit_sibling_usage(tmp_path):
    async def run():
        recorder = DurableUsageRecorder(tmp_path / "calls.jsonl")
        client = _ConcurrentObservedClient()
        completed = UsageTracingLLMClient(
            client,
            recorder,
            trajectory_id="completed",
            problem_id="Problem_5",
            mode="reasoning_adaptive",
            phase="reasoning_prefetch",
            role="filter",
            branch_index=0,
        )
        cancelled = UsageTracingLLMClient(
            client,
            recorder,
            trajectory_id="cancelled",
            problem_id="Problem_5",
            mode="reasoning_adaptive",
            phase="reasoning_prefetch",
            role="filter",
            branch_index=0,
        )

        completed_task = asyncio.create_task(completed.chat(100))
        cancelled_task = asyncio.create_task(cancelled.chat(200))
        await completed_task
        cancelled_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_task
        return recorder

    recorder = asyncio.run(run())
    completed_event = recorder.events_for("completed")[0]
    cancelled_event = recorder.events_for("cancelled")[0]
    assert completed_event["usage_delta"]["input_tokens"] == 100
    assert cancelled_event["status"] == "cancelled"
    assert cancelled_event["usage_delta"]["requests"] == 0
    assert cancelled_event["usage_delta"]["input_tokens"] == 0
