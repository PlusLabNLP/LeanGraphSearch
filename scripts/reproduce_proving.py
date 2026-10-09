#!/usr/bin/env python3
"""Run the five Gemini 3.1 Pro proving conditions from the paper.

This is a modified Table-3 protocol: the benchmark, prompts, raw Lean score,
and retrieval contracts come from Table 3, while the proof-stage action caps
follow the paper configuration and can be set explicitly on
the command line.  Reasoning-mode preparation is executed and costed
separately before the proof-stage budget starts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leansearchv2 import StandardClient
from leansearchv2.config import get
from leansearchv2.imo_eval import load_imo_leanproofbench
from leansearchv2.llm import LLMClient
from leansearchv2.prove import (
    A2CallBudget,
    BudgetedLLMClient,
    LeanInteractVerifier,
    ProveLLMs,
    ProveProblem,
    run_prove,
    split_sorry_proof_hole,
)
from leansearchv2.reasoning import ReasoningLLMs
from leansearchv2.prove.telemetry import (
    DurableUsageRecorder,
    UsageTracingLLMClient,
    usage_cost_usd,
)


ROOT = Path(__file__).resolve().parents[1]
MPR = ROOT / "benchmark" / "MathlibMPR.json"
MPR_IDS = ROOT / "benchmark" / "MathlibMPR_Prop_ids.txt"
FATE_H = ROOT / "benchmark" / "FATE-H.jsonl"
LEAN_IMO = ROOT / "benchmark" / "LeanIMO.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "proving"
MODES = (
    "none", "standard", "graph",
    "reasoning_adaptive", "graph_reasoning_adaptive",
)
DEFAULT_MODES = (
    "none", "standard", "graph", "reasoning_adaptive", "graph_reasoning_adaptive",
)
USAGE_KEYS = (
    "requests", "input_tokens", "output_tokens",
    "cache_creation_input_tokens", "cache_read_input_tokens",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_mathlibmpr_prop() -> list[ProveProblem]:
    rows = {row["id"]: row for row in json.loads(MPR.read_text())}
    ids = [
        line.strip() for line in MPR_IDS.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise RuntimeError(f"expected 50 unique MathlibMPR-Prop ids, found {len(ids)}")
    missing = [problem_id for problem_id in ids if problem_id not in rows]
    if missing:
        raise RuntimeError(f"MathlibMPR-Prop ids missing from benchmark: {missing}")
    problems: list[ProveProblem] = []
    for problem_id in ids:
        formal_statement = rows[problem_id]["formal_statement"]
        contract = split_sorry_proof_hole(formal_statement)
        problems.append(ProveProblem(
            problem_id=problem_id,
            formal_statement=formal_statement,
            informal_statement=rows[problem_id].get("NL_main_result", ""),
            proof_prefix=contract.prefix,
            proof_suffix=contract.suffix,
        ))
    return problems


def _load_fate_h() -> list[ProveProblem]:
    rows = [json.loads(line) for line in FATE_H.read_text().splitlines() if line.strip()]
    problems = [
        ProveProblem(
            problem_id=str(row.get("name") or row.get("problem_id") or f"row_{index}"),
            formal_statement=(str(row.get("header") or "") + str(row["formal_statement"])).strip(),
            informal_statement=str(row.get("informal_statement") or ""),
            header=str(row.get("header") or ""),
        )
        for index, row in enumerate(rows)
    ]
    ids = [problem.problem_id for problem in problems]
    if len(problems) != 100 or len(set(ids)) != 100:
        raise RuntimeError(f"expected 100 unique FATE-H problems, found {len(problems)}")
    return problems


def _load_lean_imo(subset: str) -> list[ProveProblem]:
    """Load one 30-problem Lean-IMO-Bench set into the shared prove contract."""

    rows = load_imo_leanproofbench(LEAN_IMO, subset=subset)
    problems: list[ProveProblem] = []
    for row in rows:
        full_source = (
            f"{row.header.rstrip()}\n\n{row.formal_statement.strip()}"
            if row.header.strip()
            else row.formal_statement.strip()
        )
        contract = split_sorry_proof_hole(full_source)
        problems.append(ProveProblem(
            problem_id=row.problem_id,
            formal_statement=full_source,
            informal_statement=row.informal_statement,
            proof_prefix=contract.prefix,
            proof_suffix=contract.suffix,
        ))
    ids = [problem.problem_id for problem in problems]
    expected_prefix = f"PB-{subset.title()}-"
    if (
        len(problems) != 30
        or len(set(ids)) != 30
        or any(not problem_id.startswith(expected_prefix) for problem_id in ids)
    ):
        raise RuntimeError(
            f"expected 30 unique Lean-IMO-Bench {subset} problems, found {len(problems)}"
        )
    return problems


def _sum_usage(clients: list[LLMClient]) -> dict[str, int]:
    unique = {id(client): client for client in clients}
    return {
        key: sum(int(client.usage_snapshot().get(key, 0)) for client in unique.values())
        for key in USAGE_KEYS
    }


def _add_usage(*items: dict[str, int]) -> dict[str, int]:
    return {key: sum(int(item.get(key, 0)) for item in items) for key in USAGE_KEYS}


def _cost(usage: dict[str, int]) -> float:
    return (
        usage["input_tokens"] * 2.0
        + usage["output_tokens"] * 12.0
        + usage["cache_creation_input_tokens"] * 2.5
        + usage["cache_read_input_tokens"] * 0.2
    ) / 1_000_000


def _new_clients(
    mode: str,
    budget: A2CallBudget,
    recorder: DurableUsageRecorder,
    *,
    trajectory_id: str,
    problem_id: str,
) -> tuple[ProveLLMs, list[LLMClient], list[LLMClient]]:
    profile = str(get("PROVE_PROVER_LLM", "prove", "prover_llm", default="gemini31pro_vertex"))
    proof_client = LLMClient(profile)
    proof_traced = UsageTracingLLMClient(
        proof_client,
        recorder,
        trajectory_id=trajectory_id,
        problem_id=problem_id,
        mode=mode,
        phase="proof_stage",
        role="prover",
        # Failure-memory may reject an exact repeated candidate without a
        # compiler call.  Attribute every prover response to the next actual
        # compiler round rather than to its raw provider-call index.
        round_from_call_index=lambda _index: budget.compiler_calls,
    )
    query_traced = UsageTracingLLMClient(
        proof_client,
        recorder,
        trajectory_id=trajectory_id,
        problem_id=problem_id,
        mode=mode,
        phase="proof_stage",
        role="query",
        round_from_call_index=lambda _index: budget.compiler_calls,
    )
    proof_proxy = BudgetedLLMClient(proof_traced, budget)
    query_proxy = BudgetedLLMClient(query_traced, budget)
    proof_clients = [proof_client]
    reasoning_clients: list[LLMClient] = []
    branches: list[ReasoningLLMs] = []
    if mode in (
        "reasoning", "graph_reasoning", "reasoning_adaptive", "graph_reasoning_adaptive"
    ):
        branch_budget = int(get("REASONING_BRANCH_BUDGET", "reasoning", "branch_budget", default=2))
        for branch_index in range(branch_budget):
            by_role: dict[str, Any] = {}
            for role in ("sketch", "filter", "judge"):
                role_profile = str(get(
                    f"REASONING_{role.upper()}_LLM", "reasoning", f"{role}_llm",
                    default=profile,
                ))
                client = LLMClient(role_profile)
                if client.provider in ("gemini_vertex", "vertex_gemini"):
                    client._seed += branch_index
                by_role[role] = UsageTracingLLMClient(
                    client,
                    recorder,
                    trajectory_id=trajectory_id,
                    problem_id=problem_id,
                    mode=mode,
                    phase="reasoning_prefetch",
                    role=role,
                    branch_index=branch_index,
                )
                reasoning_clients.append(client)
            branches.append(ReasoningLLMs(**by_role))
    return (
        ProveLLMs(
            prover=proof_proxy,  # type: ignore[arg-type]
            query=query_proxy,  # type: ignore[arg-type]
            reasoning_branches=branches,
        ),
        proof_clients,
        reasoning_clients,
    )


async def _close_clients(clients: list[LLMClient]) -> None:
    for client in {id(client): client for client in clients}.values():
        try:
            await client.aclose()
        except Exception:
            pass


async def _run_problem(
    problem: ProveProblem,
    *,
    mode: str,
    retriever: StandardClient | None,
    sem: asyncio.Semaphore,
    verifier: LeanInteractVerifier,
    recorder: DurableUsageRecorder,
    max_model_calls: int,
    max_query_calls: int,
    max_compiler_calls: int,
    strict_integrity: bool,
    failure_memory: bool,
) -> dict[str, Any]:
    async with sem:
        trajectory_id = str(uuid.uuid4())
        budget = A2CallBudget(
            max_model_calls=max_model_calls,
            max_query_calls=max_query_calls,
            max_compiler_calls=max_compiler_calls,
        )
        llms, proof_clients, reasoning_clients = _new_clients(
            mode,
            budget,
            recorder,
            trajectory_id=trajectory_id,
            problem_id=problem.problem_id,
        )
        started = time.time()
        run_completed_normally = False
        try:
            adaptive_reasoning = mode in ("reasoning_adaptive", "graph_reasoning_adaptive")
            result = await asyncio.wait_for(
                run_prove(
                    problem,
                    llms,
                    retriever,
                    verifier,
                    retriever_mode=mode,
                    # Round 0 is the initial proof; every later compiler round
                    # may consume one query action.  With 32 compiler calls this
                    # exposes exactly reflection rounds 1..31.
                    reflection_rounds=max_compiler_calls - 1,
                    num_queries=5,
                    search_top_k=50,
                    retriever_concurrency=2,
                    reasoning_search_top_k=50 if adaptive_reasoning else 30,
                    reasoning_big_loop=3,
                    reasoning_output_top_k=100,
                    reasoning_filter_max_docs=50 if adaptive_reasoning else 30,
                    verify_timeout_s=600,
                    strict_integrity=strict_integrity,
                    strict_retrieval_errors=True,
                    call_budget=budget,
                    failure_memory_enabled=failure_memory,
                    failure_memory_repeat_threshold=2,
                    failure_memory_summary_max_states=3,
                    failure_memory_deduplicate_queries=True,
                    failure_memory_block_exact_proofs=True,
                ),
                timeout=115200,
            )
            record = result.to_dict()
            run_completed_normally = True
        except Exception as exc:
            record = {
                "id": problem.problem_id,
                "success": False,
                "strict_success": False,
                "retriever_mode": mode,
                "attempts": [],
                "final_proof": "",
                "final_error": f"{type(exc).__name__}: {exc}",
                "budget": budget.snapshot(),
            }
        proof_usage = _sum_usage(proof_clients)
        reasoning_usage = _sum_usage(reasoning_clients)
        total_usage = _add_usage(proof_usage, reasoning_usage)
        events = recorder.events_for(trajectory_id)
        proof_event_usage = _add_usage(*(
            event["usage_delta"] for event in events if event["phase"] == "proof_stage"
        )) if events else {key: 0 for key in USAGE_KEYS}
        reasoning_event_usage = _add_usage(*(
            event["usage_delta"] for event in events if event["phase"] == "reasoning_prefetch"
        )) if events else {key: 0 for key in USAGE_KEYS}
        telemetry_errors: list[str] = []
        if proof_event_usage != proof_usage:
            telemetry_errors.append("proof-stage event usage does not match client aggregate")
        if reasoning_event_usage != reasoning_usage:
            telemetry_errors.append("reasoning event usage does not match client aggregate")
        if any(event.get("status") == "error" for event in events):
            telemetry_errors.append("one or more provider calls ended with an error")
        if any(
            event.get("status") == "cancelled"
            and any(int(value) != 0 for value in (event.get("usage_delta") or {}).values())
            for event in events
        ):
            telemetry_errors.append("a cancelled call reported nonzero usage")
        record.update({
            "usage_trajectory_id": trajectory_id,
            "run_completed_normally": run_completed_normally,
            "telemetry_valid": not telemetry_errors,
            "telemetry_errors": telemetry_errors,
            "api_call_events": events,
            "elapsed_s": time.time() - started,
            "proof_stage_api_usage": proof_usage,
            "reasoning_api_usage": reasoning_usage,
            "total_api_usage": total_usage,
            "proof_stage_api_cost_usd": _cost(proof_usage),
            "reasoning_api_cost_usd": _cost(reasoning_usage),
            "total_api_cost_usd": _cost(total_usage),
        })
        await _close_clients(proof_clients + reasoning_clients)
        return record


def _summary(
    benchmark: str,
    mode: str,
    rows: list[dict[str, Any]],
    complete: bool,
    target_n: int = 50,
    strict_integrity: bool = False,
) -> dict[str, Any]:
    proof_usage = _add_usage(*(row["proof_stage_api_usage"] for row in rows)) if rows else {key: 0 for key in USAGE_KEYS}
    reasoning_usage = _add_usage(*(row["reasoning_api_usage"] for row in rows)) if rows else {key: 0 for key in USAGE_KEYS}
    total_usage = _add_usage(proof_usage, reasoning_usage)
    solved = sum(bool(row.get("success")) for row in rows)
    strict_solved = sum(bool(row.get("strict_success")) for row in rows)
    memory_rows = [row for row in rows if isinstance(row.get("failure_memory"), dict)]
    return {
        "benchmark": benchmark,
        "mode": mode,
        "complete": complete,
        "n": len(rows),
        "target_n": target_n,
        "solved": solved,
        "accuracy": solved / len(rows) if rows else 0.0,
        "strict_solved": strict_solved,
        "strict_accuracy": strict_solved / len(rows) if rows else 0.0,
        "failure_memory": {
            "enabled_rows": len(memory_rows),
            "prevented_repeated_actions": sum(
                int((row.get("failure_memory") or {}).get("prevented_repeated_actions", 0))
                for row in memory_rows
            ),
            "repeated_state_observations": sum(
                int((row.get("failure_memory") or {}).get("repeated_state_observations", 0))
                for row in memory_rows
            ),
            "dead_end_prompts": sum(
                int((row.get("failure_memory") or {}).get("dead_end_prompts", 0))
                for row in memory_rows
            ),
            "duplicate_queries_suppressed": sum(
                int((row.get("failure_memory") or {}).get("duplicate_queries_suppressed", 0))
                for row in memory_rows
            ),
        },
        "validation_mode": "integrity_strict" if strict_integrity else "paper_raw",
        "normal_telemetry_rows": sum(
            row.get("run_completed_normally") is True and row.get("telemetry_valid") is True
            for row in rows
        ),
        "proof_stage_api_usage": proof_usage,
        "reasoning_api_usage": reasoning_usage,
        "total_api_usage": total_usage,
        "proof_stage_api_cost_usd": _cost(proof_usage),
        "reasoning_api_cost_usd": _cost(reasoning_usage),
        "total_api_cost_usd": _cost(total_usage),
    }


async def _run_mode(
    mode: str,
    problems: list[ProveProblem],
    *,
    output: Path,
    retriever: StandardClient,
    parallelism: int,
    verifier: LeanInteractVerifier,
    recorder: DurableUsageRecorder,
    benchmark: str,
    resume: bool,
    max_trajectory_retries: int,
    max_model_calls: int,
    max_query_calls: int,
    max_compiler_calls: int,
    strict_integrity: bool,
    drain_file: Path | None,
    failure_memory: bool,
) -> dict[str, Any]:
    cell = output / mode
    records_path = cell / "per_problem.jsonl"
    summary_path = cell / "summary.json"
    records: list[dict[str, Any]] = []
    if resume and records_path.exists():
        records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if any(
        record.get("run_completed_normally") is not True
        or record.get("telemetry_valid") is not True
        for record in records
    ):
        raise RuntimeError(f"resume artifact for mode {mode} contains a non-normal trajectory")
    completed = {str(record["id"]) for record in records}
    expected = {problem.problem_id for problem in problems}
    if not completed <= expected or len(completed) != len(records):
        raise RuntimeError(f"invalid resume artifact for mode {mode}")
    pending = deque(problem for problem in problems if problem.problem_id not in completed)
    retries: dict[str, int] = {}
    target_n = len(problems)
    sem = asyncio.Semaphore(parallelism)
    active: dict[asyncio.Task[dict[str, Any]], ProveProblem] = {}
    abnormal_path = output / "abnormal_trajectories.jsonl"
    while pending or active:
        if drain_file is not None and drain_file.exists():
            # Do not cancel paid provider calls.  Stop admitting new problems,
            # let every active trajectory finish and record normally, then
            # return an intentionally incomplete cell to the outer runner.
            pending.clear()
        while pending and len(active) < parallelism:
            problem = pending.popleft()
            task = asyncio.create_task(_run_problem(
                problem,
                mode=mode,
                retriever=None if mode == "none" else retriever,
                sem=sem,
                verifier=verifier,
                recorder=recorder,
                max_model_calls=max_model_calls,
                max_query_calls=max_query_calls,
                max_compiler_calls=max_compiler_calls,
                strict_integrity=strict_integrity,
                failure_memory=failure_memory,
            ))
            active[task] = problem
        if not active:
            break
        done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            problem = active.pop(task)
            record = await task
            normal = (
                record.get("run_completed_normally") is True
                and record.get("telemetry_valid") is True
            )
            if not normal:
                with abnormal_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                attempt = retries.get(problem.problem_id, 0) + 1
                retries[problem.problem_id] = attempt
                if attempt > max_trajectory_retries:
                    if active:
                        finished = await asyncio.gather(*active, return_exceptions=True)
                        for finished_record in finished:
                            if isinstance(finished_record, dict):
                                with abnormal_path.open("a", encoding="utf-8") as handle:
                                    handle.write(json.dumps(finished_record, ensure_ascii=False) + "\n")
                    raise RuntimeError(
                        f"{mode}/{problem.problem_id} exceeded {max_trajectory_retries} abnormal-trajectory retries"
                    )
                pending.appendleft(problem)
                continue
            records.append(record)
            _atomic_jsonl(records_path, records)
            interim = _summary(
                benchmark,
                mode,
                records,
                complete=False,
                target_n=target_n,
                strict_integrity=strict_integrity,
            )
            _atomic_json(summary_path, interim)
            if len(records) % 5 == 0 or len(records) == len(problems):
                print(
                    f"PROGRESS mode={mode} n={len(records)}/{target_n} solved={interim['solved']} "
                    f"accuracy={interim['accuracy']:.3f} cost=${interim['total_api_cost_usd']:.2f}",
                    flush=True,
                )
    records.sort(key=lambda row: next(i for i, p in enumerate(problems) if p.problem_id == row["id"]))
    _atomic_jsonl(records_path, records)
    final = _summary(
        benchmark,
        mode,
        records,
        complete=len(records) == len(problems),
        target_n=target_n,
        strict_integrity=strict_integrity,
    )
    final["drained"] = bool(drain_file is not None and drain_file.exists())
    _atomic_json(summary_path, final)
    return final


async def run(args: argparse.Namespace) -> int:
    if args.benchmark == "fate_h":
        problems = _load_fate_h()
        benchmark_label = "FATE-H"
        benchmark_path = FATE_H
        ids_path: Path | None = None
    elif args.benchmark in ("lean_imo_basic", "lean_imo_advanced"):
        subset = args.benchmark.removeprefix("lean_imo_")
        problems = _load_lean_imo(subset)
        benchmark_label = f"Lean-IMO-Bench {subset.title()}"
        benchmark_path = LEAN_IMO
        ids_path = None
    else:
        problems = _load_mathlibmpr_prop()
        benchmark_label = "MathlibMPR-Prop"
        benchmark_path = MPR
        ids_path = MPR_IDS
    if args.ids:
        requested_ids = [item.strip() for item in args.ids.split(",") if item.strip()]
        if len(requested_ids) != len(set(requested_ids)):
            raise RuntimeError("--ids contains duplicates")
        by_id = {problem.problem_id: problem for problem in problems}
        missing_ids = [problem_id for problem_id in requested_ids if problem_id not in by_id]
        if missing_ids:
            raise RuntimeError(f"unknown --ids: {missing_ids}")
        problems = [by_id[problem_id] for problem_id in requested_ids]
    else:
        problems = problems[args.offset:]
    if args.limit is not None:
        problems = problems[:args.limit]
    output = Path(args.output)
    if (output / "INVALIDATED.md").exists():
        raise RuntimeError("This experiment was invalidated; select a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    unknown = set(modes) - set(MODES)
    if unknown:
        raise RuntimeError(f"unknown modes: {sorted(unknown)}")
    manifest = {
        "experiment": f"Gemini 3.1 Pro HIGH / {benchmark_label} / modified Table 3",
        "modes": list(modes),
        "problem_ids": [problem.problem_id for problem in problems],
        "protocol": {
            "proof_stage": {
                "max_model_calls": args.max_model_calls,
                "max_query_actions": args.max_query_calls,
                "max_compiler_calls": args.max_compiler_calls,
                "standard_queries_per_action": 5,
                "standard_top_k_per_query": 50,
            },
            "reasoning_separate_from_proof_budget": True,
            "reasoning_fixed": {"branch_budget": 2, "search_top_k": 30, "filter_max_docs": 30, "output_top_k": 100, "big_loop": 3, "reflection_retrieval": "fixed"},
            "reasoning_adaptive": {
                "branch_budget": 2,
                "search_top_k": 50,
                "filter_max_docs": 50,
                "output_top_k": 100,
                "big_loop": 3,
                "reflection_retrieval": (
                    f"5 queries x top-50, max {args.max_query_calls} actions"
                ),
            },
            "retriever_request_concurrency_per_problem": 2,
            "graph": {"initial_top_n": 50, "expand_m": 200, "final_top_k": 100, "rank_fusion": "qwen_ppr_zscore", "ppr_zscore_weight": 0.005},
            "model": "gemini-3.1-pro-preview-customtools",
            "thinking_level": "HIGH",
            "temperature": None,
            "max_tokens": 65536,
            "validation_mode": "integrity_strict" if args.strict_integrity else "paper_raw",
            "compiler_reports_integrity_violations": args.strict_integrity,
            "proof_contract": "host_assembled_tactic_body_v3_indent_preserved",
            "strict_response_format": (
                "tactic_body_only" if args.strict_integrity else "complete_lean_source"
            ),
            "prompt_version": (
                "strict_v3_1_contextual_body_repair"
                if args.strict_integrity else "paper_raw"
            ),
            "failure_memory": {
                "enabled": args.failure_memory,
                "mode": "free_failure_memory" if args.failure_memory else "free",
                "single_trajectory": True,
                "repeat_threshold": 2,
                "summary_max_states": 3,
                "deduplicate_queries_by_state": True,
                "block_exact_failed_proofs": True,
                "exact_block_does_not_consume_compiler_budget": True,
            },
            "per_call_usage_telemetry": True,
        },
        "provenance": {
            "git_commit": _git_commit(),
            "benchmark_sha256": _sha256(benchmark_path),
            "ids_sha256": _sha256(ids_path) if ids_path is not None else None,
            "runner_sha256": _sha256(Path(__file__)),
            "prove_runner_sha256": _sha256(ROOT / "src" / "leansearchv2" / "prove" / "run.py"),
            "prove_verifier_sha256": _sha256(ROOT / "src" / "leansearchv2" / "prove" / "verifier.py"),
            "prove_prompts_sha256": _sha256(
                ROOT / "src" / "leansearchv2" / "prove" / "prompts.py"
            ),
            "prove_integrity_sha256": _sha256(
                ROOT / "src" / "leansearchv2" / "prove" / "integrity.py"
            ),
            "reasoning_runner_sha256": _sha256(ROOT / "src" / "leansearchv2" / "reasoning" / "run.py"),
            "config_path": os.environ.get("LEANSEARCH_CONFIG"),
        },
    }
    _atomic_json(output / "run_manifest.json", manifest)
    recorder = DurableUsageRecorder(output / "api_call_events.jsonl")
    retriever = StandardClient(url=args.url, timeout=600)
    verifier = LeanInteractVerifier(
        project_dir=args.lean_project_dir,
        lean_version="v4.28.0-rc1",
        auto_build_project=False,
    )
    summaries = []
    drain_file = Path(args.drain_file).resolve() if args.drain_file else None
    try:
        # Build/start the single long-lived REPL before any paid model call.
        # LocalProject performs an expensive Mathlib cache check at startup;
        # sharing this verifier across all 250 cell-runs prevents repeating it.
        prewarm = await verifier.verify(
            "import Mathlib\nexample : True := by\n  trivial",
            timeout_s=600,
        )
        if not prewarm.success:
            raise RuntimeError(f"Lean REPL prewarm failed: {prewarm.error_msg}")
        for mode in modes:
            summaries.append(await _run_mode(
                mode, problems, output=output, retriever=retriever,
                parallelism=args.parallelism, verifier=verifier,
                recorder=recorder, benchmark=benchmark_label, resume=args.resume,
                max_trajectory_retries=args.max_trajectory_retries,
                max_model_calls=args.max_model_calls,
                max_query_calls=args.max_query_calls,
                max_compiler_calls=args.max_compiler_calls,
                strict_integrity=args.strict_integrity,
                drain_file=drain_file,
                failure_memory=args.failure_memory,
            ))
            if drain_file is not None and drain_file.exists():
                break
    finally:
        await verifier.close()
    _atomic_json(output / "summary.json", summaries)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--benchmark",
        choices=(
            "mathlibmpr_prop", "fate_h", "lean_imo_basic", "lean_imo_advanced",
        ),
        default="mathlibmpr_prop",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument("--lean-project-dir", default=str(ROOT / "external" / "Mathlib4"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default="", help="Comma-separated explicit problem ids")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-trajectory-retries", type=int, default=3)
    parser.add_argument("--max-model-calls", type=int, default=63)
    parser.add_argument("--max-query-calls", type=int, default=31)
    parser.add_argument("--max-compiler-calls", type=int, default=32)
    parser.add_argument(
        "--strict-integrity",
        action="store_true",
        help="Reject and reflect on proof escapes before invoking Lean",
    )
    parser.add_argument(
        "--drain-file",
        default="",
        help="When this path exists, admit no new problems and finish active trajectories",
    )
    parser.add_argument(
        "--failure-memory",
        action="store_true",
        help="Enable the prior single-trajectory advisory formal failure memory",
    )
    args = parser.parse_args()
    if (
        args.parallelism < 1
        or args.offset < 0
        or args.max_trajectory_retries < 0
        or args.max_model_calls < 1
        or args.max_query_calls < 0
        or args.max_compiler_calls < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("parallelism and limit must be positive; offset must be non-negative")
    if args.max_query_calls > args.max_compiler_calls:
        parser.error("max-query-calls cannot exceed max-compiler-calls")
    # Round 0 is the initial proof, so only the remaining compiler rounds can
    # consume a reflection query.  A user-facing ceiling equal to the compiler
    # budget is valid; its final slot is simply unreachable under this loop.
    reachable_query_calls = min(
        args.max_query_calls, max(0, args.max_compiler_calls - 1)
    )
    if args.max_model_calls < args.max_compiler_calls + reachable_query_calls:
        parser.error(
            "max-model-calls must cover every compiler/prover call plus every "
            "reachable reflection query call"
        )
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
