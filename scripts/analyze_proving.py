#!/usr/bin/env python3
"""Audit and score budgeted five-cell prove experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from leansearchv2.prove.integrity import find_forbidden_keywords
from leansearchv2.prove.telemetry import USAGE_KEYS, usage_cost_usd
from leansearchv2.imo_eval import load_imo_leanproofbench


ROOT = Path(__file__).resolve().parents[1]
KS = (8, 16, 32)
DEFAULT_BUDGET = {
    "max_model_calls": 40,
    "max_query_calls": 8,
    "max_compiler_calls": 32,
}
CELL_DIRS = {
    "no_retrieval": "none",
    "leansearch_standard": "standard",
    "graph_standard": "graph",
    "leansearch_reasoning": "reasoning_adaptive",
    "graph_reasoning": "graph_reasoning_adaptive",
}
DYNAMIC_MODES = {
    "standard", "graph", "reasoning_adaptive", "graph_reasoning_adaptive",
}
ADAPTIVE_MODES = {"reasoning_adaptive", "graph_reasoning_adaptive"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _benchmark(benchmark: str) -> tuple[str, int, set[str]]:
    if benchmark == "fate_h":
        rows = _jsonl(ROOT / "benchmark/FATE-H.jsonl")
        ids = {
            str(row.get("name") or row.get("problem_id") or f"row_{index}")
            for index, row in enumerate(rows)
        }
        return "FATE-H", 100, ids
    if benchmark in {"lean_imo_basic", "lean_imo_advanced"}:
        subset = benchmark.removeprefix("lean_imo_")
        rows = load_imo_leanproofbench(
            ROOT / "benchmark/LeanIMO.csv",
            subset=subset,
        )
        ids = {row.problem_id for row in rows}
        return f"Lean-IMO-Bench {subset.title()}", 30, ids
    ids = {
        line.strip()
        for line in (ROOT / "benchmark/MathlibMPR_Prop_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return "MathlibMPR-Prop", 50, ids


def _sum_usage(items: list[dict[str, int]]) -> dict[str, int]:
    return {key: sum(int(item.get(key, 0)) for item in items) for key in USAGE_KEYS}


def _zero_usage(value: dict[str, Any]) -> bool:
    return all(int(value.get(key, 0)) == 0 for key in USAGE_KEYS)


def _success_round(row: dict[str, Any], *, corrected: bool) -> int | None:
    rounds = []
    for attempt in row.get("attempts") or []:
        if attempt.get("success") is not True:
            continue
        if corrected and (
            attempt.get("integrity_valid") is not True
            or find_forbidden_keywords(str(attempt.get("proof") or ""))
        ):
            continue
        rounds.append(int(attempt["round"]))
    return min(rounds) if rounds else None


def _client_key(event: dict[str, Any]) -> tuple[Any, ...]:
    if event.get("phase") == "proof_stage":
        return ("proof_stage_shared_client",)
    return (event.get("phase"), event.get("branch_index"), event.get("role"))


def _reconstruct_client_usage(events: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[_client_key(event)].append(event)
    return {
        key: sum(
            max(int((event.get("usage_after") or {}).get(key, 0)) for event in group)
            for group in grouped.values()
        )
        for key in USAGE_KEYS
    }


def _audit_cell(
    path: Path,
    expected_ids: set[str],
    expected_n: int,
    require_complete: bool,
    strict_integrity_expected: bool,
    failure_memory_expected: bool,
    expected_budget: dict[str, int],
) -> dict[str, Any]:
    rows = _jsonl(path / "per_problem.jsonl")
    errors: list[str] = []
    ids = [str(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= expected_ids:
        errors.append("duplicate or unknown problem ids")
    if require_complete and (len(rows) != expected_n or set(ids) != expected_ids):
        errors.append(f"expected exact n={expected_n}, found {len(rows)}")

    selected_event_ids: set[str] = set()
    selected_trajectory_ids: set[str] = set()
    for row in rows:
        problem_id = str(row["id"])
        mode = str(row.get("retriever_mode"))
        if row.get("run_completed_normally") is not True or row.get("telemetry_valid") is not True:
            errors.append(f"{problem_id}: selected trajectory is not normal and telemetry-valid")

        budget = row.get("budget") or {}
        for key, expected in expected_budget.items():
            if int(budget.get(key, -1)) != expected:
                errors.append(f"{problem_id}: {key}={budget.get(key)!r}, expected {expected}")
        attempts = list(row.get("attempts") or [])
        rounds = [int(attempt.get("round", -1)) for attempt in attempts]
        if not attempts or rounds != list(range(len(attempts))):
            errors.append(f"{problem_id}: proof attempts are empty or non-contiguous")
        compiler_calls = int(budget.get("compiler_calls", -1))
        query_calls = int(budget.get("query_calls", -1))
        model_calls = int(budget.get("model_calls", -1))
        if compiler_calls != len(attempts):
            errors.append(f"{problem_id}: compiler calls do not match attempts")
        memory = row.get("failure_memory") or {}
        memory_events = list(row.get("failure_memory_events") or [])
        blocked_events = [
            event
            for event in memory_events
            if event.get("type") == "failure_memory_action_blocked"
        ]
        blocked_actions = len(blocked_events)
        if failure_memory_expected:
            if memory.get("single_trajectory") is not True:
                errors.append(f"{problem_id}: missing single-trajectory failure memory")
            if int(memory.get("prevented_repeated_actions", -1)) != blocked_actions:
                errors.append(f"{problem_id}: blocked-action memory/event mismatch")
            for summary_key, event_type in (
                ("dead_end_prompts", "failure_memory_dead_end_prompt"),
                ("failure_state_observations", "failure_memory_observation"),
            ):
                event_count = sum(
                    event.get("type") == event_type for event in memory_events
                )
                if int(memory.get(summary_key, -1)) != event_count:
                    errors.append(
                        f"{problem_id}: {summary_key} memory/event mismatch"
                    )
            repeated_events = sum(
                event.get("type") == "failure_memory_observation"
                and event.get("repeated") is True
                for event in memory_events
            )
            if int(memory.get("repeated_state_observations", -1)) != repeated_events:
                errors.append(f"{problem_id}: repeated-state memory/event mismatch")
            suppressed_queries = sum(
                int(event.get("count", 0))
                for event in memory_events
                if event.get("type") == "duplicate_query_suppressed"
            )
            if int(memory.get("duplicate_queries_suppressed", -1)) != suppressed_queries:
                errors.append(f"{problem_id}: duplicate-query memory/event mismatch")
        elif memory or memory_events:
            errors.append(f"{problem_id}: unexpected failure-memory state")
        if mode not in DYNAMIC_MODES and query_calls != 0:
            errors.append(f"{problem_id}: non-dynamic mode used query actions")
        if query_calls < 0 or query_calls > expected_budget["max_query_calls"]:
            errors.append(f"{problem_id}: query-call budget invalid")
        expected_model_calls = len(attempts) + query_calls + blocked_actions
        if model_calls != expected_model_calls:
            errors.append(
                f"{problem_id}: model calls {model_calls} != compiler candidates "
                f"{len(attempts)} + query actions {query_calls} + blocked exact "
                f"candidates {blocked_actions}"
            )

        successes = [attempt for attempt in attempts if attempt.get("success") is True]
        if bool(successes) != (row.get("success") is True):
            errors.append(f"{problem_id}: row/attempt success mismatch")
        if successes and successes[-1].get("proof") != row.get("final_proof"):
            errors.append(f"{problem_id}: final proof is not the successful attempt")
        if strict_integrity_expected:
            if any(attempt.get("integrity_valid") is not True for attempt in successes):
                errors.append(f"{problem_id}: strict compiler accepted an integrity violation")
            if bool(row.get("success")) != bool(row.get("strict_success")):
                errors.append(f"{problem_id}: strict success and compiler success differ")

        dynamic = [
            event for event in row.get("retrieval_trace") or []
            if str(event.get("source"))
            in {"adaptive_standard_search", "adaptive_graph_search"}
        ]
        if len(dynamic) != query_calls:
            errors.append(f"{problem_id}: dynamic trace/query-call mismatch")
        dynamic_rounds = [int(event.get("round", -1)) for event in dynamic]
        if dynamic_rounds != list(range(1, query_calls + 1)):
            errors.append(f"{problem_id}: dynamic query rounds are not contiguous 1..q")
        if any(len(event.get("dynamic_queries") or []) != 5 for event in dynamic):
            errors.append(f"{problem_id}: dynamic action did not contain exactly five queries")
        if mode in ADAPTIVE_MODES:
            reasoning = row.get("reasoning_trace") or {}
            if int(reasoning.get("branch_budget", -1)) != 2:
                errors.append(f"{problem_id}: reasoning branch budget is not two")
            if any(reasoning.get("branch_errors") or []):
                errors.append(f"{problem_id}: reasoning prefetch branch error")

        trajectory_id = str(row.get("usage_trajectory_id") or "")
        events = list(row.get("api_call_events") or [])
        selected_trajectory_ids.add(trajectory_id)
        if not trajectory_id or not events:
            errors.append(f"{problem_id}: missing trajectory id or events")
        if any(str(event.get("trajectory_id")) != trajectory_id for event in events):
            errors.append(f"{problem_id}: mixed trajectory ids")
        event_ids = [str(event.get("event_id") or "") for event in events]
        if (
            any(not event_id for event_id in event_ids)
            or len(event_ids) != len(set(event_ids))
            or bool(selected_event_ids.intersection(event_ids))
        ):
            errors.append(f"{problem_id}: missing or duplicate event ids")
        selected_event_ids.update(event_ids)
        for event in events:
            status = str(event.get("status"))
            if status not in {"ok", "cancelled"}:
                errors.append(f"{problem_id}: provider event status {status}")
            if status == "cancelled" and not _zero_usage(event.get("usage_delta") or {}):
                errors.append(f"{problem_id}: cancelled event has nonzero usage")

        proof_events = [event for event in events if event.get("phase") == "proof_stage"]
        reasoning_events = [
            event for event in events if event.get("phase") == "reasoning_prefetch"
        ]
        if _sum_usage([event.get("usage_delta") or {} for event in proof_events]) != (
            row.get("proof_stage_api_usage") or {}
        ):
            errors.append(f"{problem_id}: proof-stage usage mismatch")
        if _sum_usage([event.get("usage_delta") or {} for event in reasoning_events]) != (
            row.get("reasoning_api_usage") or {}
        ):
            errors.append(f"{problem_id}: reasoning usage mismatch")
        event_cost = sum(float(event.get("cost_usd", 0.0)) for event in events)
        if abs(event_cost - float(row.get("total_api_cost_usd", 0.0))) > 1e-8:
            errors.append(f"{problem_id}: event cost mismatch")
        prover_rounds = sorted(
            int(event["proof_round"])
            for event in proof_events
            if event.get("role") == "prover" and event.get("status") == "ok"
        )
        query_rounds = sorted(
            int(event["proof_round"])
            for event in proof_events
            if event.get("role") == "query" and event.get("status") == "ok"
        )
        blocked_rounds = sorted(int(event.get("round", -1)) for event in blocked_events)
        if prover_rounds != sorted(rounds + blocked_rounds):
            errors.append(f"{problem_id}: prover event rounds do not match attempts")
        if query_rounds != dynamic_rounds:
            errors.append(f"{problem_id}: query event rounds do not match query actions")

    summary = json.loads((path / "summary.json").read_text(encoding="utf-8")) if (
        path / "summary.json"
    ).exists() else {}
    solved = sum(row.get("success") is True for row in rows)
    if require_complete and summary.get("complete") is not True:
        errors.append("summary is not complete")
    if rows and int(summary.get("target_n", -1)) != expected_n:
        errors.append("summary target_n mismatch")
    if rows and int(summary.get("solved", -1)) != solved:
        errors.append("summary solved count mismatch")
    if rows and strict_integrity_expected and summary.get("validation_mode") != "integrity_strict":
        errors.append("summary does not record integrity-strict validation")

    pass_at_k: dict[str, Any] = {}
    corrected_at_k: dict[str, Any] = {}
    cost_at_k: dict[str, Any] = {}
    for k in KS:
        raw_ids = sorted(
            str(row["id"])
            for row in rows
            if (round_index := _success_round(row, corrected=False)) is not None
            and round_index <= k
        )
        corrected_ids = sorted(
            str(row["id"])
            for row in rows
            if (round_index := _success_round(row, corrected=True)) is not None
            and round_index <= k
        )
        retained = [
            event
            for row in rows
            for event in row.get("api_call_events") or []
            if event.get("phase") == "reasoning_prefetch"
            or (
                event.get("phase") == "proof_stage"
                and event.get("proof_round") is not None
                and int(event["proof_round"]) <= k
            )
        ]
        total_cost = sum(float(event.get("cost_usd", 0.0)) for event in retained)
        denominator = len(rows) or 1
        pass_at_k[str(k)] = {
            "solved": len(raw_ids), "accuracy": len(raw_ids) / denominator,
            "solved_ids": raw_ids,
        }
        corrected_at_k[str(k)] = {
            "solved": len(corrected_ids),
            "accuracy": len(corrected_ids) / denominator,
            "solved_ids": corrected_ids,
        }
        cost_at_k[str(k)] = {
            "total_api_cost_usd": total_cost,
            "api_requests": sum(
                int((event.get("usage_delta") or {}).get("requests", 0))
                for event in retained
            ),
            "cost_per_corrected_success_usd": (
                total_cost / len(corrected_ids) if corrected_ids else None
            ),
        }

    selected_total = sum(float(row.get("total_api_cost_usd", 0.0)) for row in rows)
    if rows and abs(cost_at_k["32"]["total_api_cost_usd"] - selected_total) > 1e-8:
        errors.append("pass@32 cost does not reproduce selected total")
    return {
        "path": str(path),
        "n": len(rows),
        "audit_pass": not errors,
        "audit_errors": errors,
        "pass_at_k": pass_at_k,
        "pass_at_k_corrected": corrected_at_k,
        "cost_at_k": cost_at_k,
        "failure_memory_statistics": {
            key: sum(
                int((row.get("failure_memory") or {}).get(key, 0))
                for row in rows
            )
            for key in (
                "failure_state_observations",
                "repeated_state_observations",
                "dead_end_prompts",
                "prevented_repeated_actions",
                "duplicate_queries_suppressed",
            )
        },
        "selected_total_api_cost_usd": selected_total,
        "selected_event_ids": sorted(selected_event_ids),
        "selected_trajectory_ids": sorted(selected_trajectory_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=(
            "mathlibmpr_prop", "fate_h", "lean_imo_basic", "lean_imo_advanced",
        ),
        required=True,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--max-model-calls", type=int, default=DEFAULT_BUDGET["max_model_calls"]
    )
    parser.add_argument(
        "--max-query-calls", type=int, default=DEFAULT_BUDGET["max_query_calls"]
    )
    parser.add_argument(
        "--max-compiler-calls", type=int, default=DEFAULT_BUDGET["max_compiler_calls"]
    )
    args = parser.parse_args()

    benchmark_label, expected_n, expected_ids = _benchmark(args.benchmark)
    if len(expected_ids) != expected_n:
        raise RuntimeError(f"{benchmark_label}: expected {expected_n} unique ids")
    root = Path(args.input).resolve()
    manifest_path = root / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    protocol = manifest.get("protocol") or {}
    expected_budget = {
        "max_model_calls": args.max_model_calls,
        "max_query_calls": args.max_query_calls,
        "max_compiler_calls": args.max_compiler_calls,
    }
    manifest_proof_stage = protocol.get("proof_stage") or {}
    manifest_budget = {
        "max_model_calls": manifest_proof_stage.get("max_model_calls"),
        "max_query_calls": manifest_proof_stage.get("max_query_actions"),
        "max_compiler_calls": manifest_proof_stage.get("max_compiler_calls"),
    }
    if manifest and manifest_budget != expected_budget:
        raise RuntimeError(
            f"manifest budget {manifest_budget} != requested audit budget {expected_budget}"
        )
    strict_integrity_expected = bool(
        protocol.get("compiler_reports_integrity_violations")
        or protocol.get("validation_mode") == "integrity_strict"
    )
    failure_memory_protocol = protocol.get("failure_memory") or {}
    failure_memory_expected = bool(failure_memory_protocol.get("enabled"))
    cells = {
        label: _audit_cell(
            root / directory,
            expected_ids,
            expected_n,
            not args.allow_incomplete,
            strict_integrity_expected,
            failure_memory_expected,
            expected_budget,
        )
        for label, directory in CELL_DIRS.items()
    }
    global_events = _jsonl(root / "api_call_events.jsonl")
    selected_event_ids = {
        event_id for cell in cells.values() for event_id in cell["selected_event_ids"]
    }
    unselected = [
        event for event in global_events
        if str(event.get("event_id")) not in selected_event_ids
    ]
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in unselected:
        by_trajectory[str(event.get("trajectory_id") or "")].append(event)
    reconstructed_usage = _sum_usage(
        [_reconstruct_client_usage(events) for events in by_trajectory.values()]
    )
    report = {
        "schema_version": 1,
        "benchmark": benchmark_label,
        "expected_n": expected_n,
        "complete": all(cell["n"] == expected_n for cell in cells.values()),
        "audit_pass": all(cell["audit_pass"] for cell in cells.values()),
        "compiler_integrity_enforced": strict_integrity_expected,
        "failure_memory_enabled": failure_memory_expected,
        "budget": expected_budget,
        "query_policy": (
            f"at most {expected_budget['max_query_calls']} five-query retrieval actions, "
            "one before the next actual compiler round while budget remains; retrieval "
            "does not consume pass@k"
        ),
        "failure_memory_policy": failure_memory_protocol,
        "cutoff_definition": (
            "round 0 initial proof plus reflection rounds 1..k; retrieval and "
            "reasoning prefetch do not consume k"
        ),
        "corrected_definition": (
            "compiler-success plus exact immutable proof prefix and suffix and exclusion "
            "of generated proof escape hatches; identical to raw success when the strict "
            "compiler gate is enabled"
        ),
        "cost_definition": (
            "exact response-reported per-call usage through proof_round <= k; fixed "
            "reasoning prefetch included"
        ),
        "cells": cells,
        "global_cost_ledger": {
            "events": len(global_events),
            "selected_normal_events": len(selected_event_ids),
            "unselected_or_incomplete_events": len(unselected),
            "selected_normal_cost_usd": sum(
                cell["selected_total_api_cost_usd"] for cell in cells.values()
            ),
            "unselected_or_incomplete_trajectories": len(by_trajectory),
            "unselected_or_incomplete_usage_reconstructed": reconstructed_usage,
            "unselected_or_incomplete_cost_usd": usage_cost_usd(reconstructed_usage),
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "benchmark": benchmark_label,
        "complete": report["complete"],
        "audit_pass": report["audit_pass"],
        "n": {label: cell["n"] for label, cell in cells.items()},
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
