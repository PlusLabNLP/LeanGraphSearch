#!/usr/bin/env python3
"""Run a cost-accounted, paper-budgeted MathlibMPR reasoning evaluation.

This uses the paper's two independent reasoning branches per problem (unless
explicitly overridden), deterministic slicing, and response-reported API usage.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.asyncio import tqdm_asyncio

from leansearchv2 import StandardClient
from leansearchv2.config import get
from leansearchv2.eval.premise_metrics import KS, aggregate, score_one
from leansearchv2.llm import LLMClient
from leansearchv2.reasoning import (
    Problem,
    ReasoningLLMs,
    run_reasoning_budgeted,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = REPO_ROOT / "benchmark" / "MathlibMPR.json"
USAGE_KEYS = (
    "requests",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)
log = logging.getLogger("reproduce_premise_costed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
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


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _new_llms() -> tuple[ReasoningLLMs, dict[str, LLMClient]]:
    names = {
        "sketch": str(get("REASONING_SKETCH_LLM", "reasoning", "sketch_llm")),
        "filter": str(get("REASONING_FILTER_LLM", "reasoning", "filter_llm")),
        "judge": str(get("REASONING_JUDGE_LLM", "reasoning", "judge_llm")),
    }
    # Separate clients make per-role usage exact even when all roles use the
    # same profile. The requests remain stateless and use the original prompts.
    clients = {role: LLMClient(profile) for role, profile in names.items()}
    return ReasoningLLMs(**clients), clients


def _new_llm_branches(
    branch_budget: int,
) -> tuple[list[ReasoningLLMs], list[dict[str, LLMClient]]]:
    bundles: list[ReasoningLLMs] = []
    clients_by_branch: list[dict[str, LLMClient]] = []
    for branch_index in range(branch_budget):
        bundle, clients = _new_llms()
        # Gemini at temperature zero can otherwise produce duplicate branches
        # when every client inherits the same configured seed. Stable adjacent
        # seeds preserve the paper's independent-branch budget and make the
        # Standard/Graph comparison exactly matched.
        for client in clients.values():
            if client.provider in ("gemini_vertex", "vertex_gemini"):
                client._seed += branch_index
        bundles.append(bundle)
        clients_by_branch.append(clients)
    return bundles, clients_by_branch


def _sum_usage(usages: list[dict[str, int]]) -> dict[str, int]:
    return {key: sum(int(usage.get(key, 0)) for usage in usages) for key in USAGE_KEYS}


def _cost(
    usage: dict[str, int],
    *,
    input_price: float,
    output_price: float,
    cache_write_price: float,
    cache_read_price: float,
) -> float:
    return (
        usage["input_tokens"] * input_price
        + usage["output_tokens"] * output_price
        + usage["cache_creation_input_tokens"] * cache_write_price
        + usage["cache_read_input_tokens"] * cache_read_price
    ) / 1_000_000


def _retriever_search_config() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return request kwargs and auditable graph-only protocol metadata.

    Standard mode keeps an empty kwargs mapping.  Graph mode explicitly
    requests its larger internal final pool; reasoning/run.py slices that
    pool back to ``search_top_k`` before the LLM filter sees it.
    """
    raw = get("GRAPH_CONFIG", "graph", default={})
    graph = dict(raw) if isinstance(raw, dict) else {}
    if not bool(graph.get("graph_augment", False)):
        return {}, {"graph_augment": False}

    kwargs: dict[str, Any] = {"graph_augment": True, "return_metadata": True}
    field_map = {
        "initial_top_n": "graph_initial_top_n",
        "graph_expand_m": "graph_expand_m",
        "final_top_k": "graph_final_top_k",
        "rank_fusion": "rank_fusion",
    }
    for config_key, request_key in field_map.items():
        value = graph.get(config_key)
        if value is not None:
            kwargs[request_key] = value
    metadata = {
        "graph_augment": True,
        "request_kwargs": kwargs,
        "ppr_zscore_weight": graph.get("ppr_zscore_weight"),
        "graph_artifact_dir": str(graph.get("artifact_dir")),
        "graph_edge_profile": graph.get("graph_edge_profile"),
    }
    return kwargs, metadata


async def _run_one(
    row: dict[str, Any],
    retriever: StandardClient,
    *,
    search_top_k: int,
    filter_max_docs: int,
    big_loop: int,
    branch_budget: int,
    retriever_search_kwargs: dict[str, Any],
    sem: asyncio.Semaphore,
    prices: dict[str, float],
) -> dict[str, Any]:
    async with sem:
        llm_branches, clients_by_branch = _new_llm_branches(branch_budget)
        started = time.time()
        problem = Problem(
            problem_id=row["id"],
            formal_statement=row["formal_statement"],
            informal_statement=row.get("NL_main_result", ""),
            informal_proof="",
        )
        try:
            result = await run_reasoning_budgeted(
                problem,
                retriever,
                llm_branches,
                search_top_k=search_top_k,
                output_top_k=100,
                big_loop=big_loop,
                filter_max_docs=filter_max_docs,
                retriever_search_kwargs=retriever_search_kwargs,
            )
            retrieved_ids = [doc_id for doc_id, _score, _result in result.entries]
            scores = score_one(retrieved_ids, row["premise_group"])
            record: dict[str, Any] = {
                **result.to_dict(),
                "retrieved": retrieved_ids,
                "premise_group": row["premise_group"],
                "scores": {
                    str(k): {
                        "recall_group": scores["recall_group"][k],
                        "covered": scores["covered"][k],
                    }
                    for k in KS
                },
            }
        except Exception as exc:
            log.exception("FAIL %s", row["id"])
            empty_scores = score_one([], row["premise_group"])
            record = {
                "id": row["id"],
                "error": f"{type(exc).__name__}: {exc}",
                "branch_budget": branch_budget,
                "retrieved": [],
                "premise_group": row["premise_group"],
                "scores": {
                    str(k): {
                        "recall_group": empty_scores["recall_group"][k],
                        "covered": empty_scores["covered"][k],
                    }
                    for k in KS
                },
            }
        by_branch = [
            {role: client.usage_snapshot() for role, client in clients.items()}
            for clients in clients_by_branch
        ]
        by_role = {
            role: _sum_usage([branch[role] for branch in by_branch])
            for role in ("sketch", "filter", "judge")
        }
        usage = _sum_usage(list(by_role.values()))
        record["api_usage_by_branch"] = by_branch
        record["api_usage_by_role"] = by_role
        record["api_usage"] = usage
        record["api_cost_usd"] = _cost(usage, **prices)
        record["wall_time_s"] = time.time() - started
        for clients in clients_by_branch:
            for client in clients.values():
                await client.aclose()
        return record


async def _run(args: argparse.Namespace) -> None:
    benchmark_path = Path(args.benchmark).resolve()
    output_dir = Path(args.output).resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists; use --resume: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = json.loads(benchmark_path.read_text(encoding="utf-8"))
    rows = all_rows[args.offset : args.offset + args.limit if args.limit else None]
    if not rows:
        raise ValueError("selected MathlibMPR slice is empty")

    per_query_path = output_dir / "per_query.jsonl"
    records: list[dict[str, Any]] = []
    if args.resume and per_query_path.exists():
        records = [
            json.loads(line)
            for line in per_query_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed_ids = [str(record.get("id", "")) for record in records]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("resume artifact contains duplicate problem IDs")
    selected_ids = {str(row["id"]) for row in rows}
    if not set(completed_ids).issubset(selected_ids):
        raise ValueError("resume artifact contains problems outside the selected slice")
    mismatched_branch_budget = [
        record.get("id")
        for record in records
        if int(record.get("branch_budget", 1)) != args.branch_budget
    ]
    if mismatched_branch_budget:
        raise ValueError(
            "resume artifact branch budget does not match --branch-budget: "
            + ", ".join(str(problem_id) for problem_id in mismatched_branch_budget)
        )
    completed_set = set(completed_ids)
    pending_rows = [row for row in rows if str(row["id"]) not in completed_set]
    resumed_records = len(records)

    prices = {
        "input_price": args.input_cost_per_mtok,
        "output_price": args.output_cost_per_mtok,
        "cache_write_price": args.cache_write_cost_per_mtok,
        "cache_read_price": args.cache_read_cost_per_mtok,
    }
    retriever_search_kwargs, retriever_protocol = _retriever_search_config()
    retriever = StandardClient(url=args.url, timeout=args.retriever_timeout)
    health = await retriever.health()
    config_path = Path(os.environ.get("CONFIG_PATH", "config.yaml")).resolve()
    role_profiles = {
        role: str(get(f"REASONING_{role.upper()}_LLM", "reasoning", f"{role}_llm"))
        for role in ("sketch", "filter", "judge")
    }
    llm_protocol: dict[str, dict[str, Any]] = {}
    for role, profile in role_profiles.items():
        raw_profile = get(f"LLM_{profile.upper()}", "llm", profile, default={})
        profile_config = dict(raw_profile) if isinstance(raw_profile, dict) else {}
        llm_protocol[role] = {
            "profile": profile,
            "provider": profile_config.get("provider"),
            "model": profile_config.get("model"),
            "thinking_level": profile_config.get("thinking_level"),
            "enable_google_search": profile_config.get("enable_google_search", False),
            "max_tokens": profile_config.get("max_tokens"),
            "base_seed": profile_config.get("seed"),
            "branch_seed_offsets": list(range(args.branch_budget)),
        }
    immutable_run = {
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": _sha256(benchmark_path),
        "selected_problem_ids": [row["id"] for row in rows],
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "reasoning_runner_sha256": _sha256(
            REPO_ROOT / "src" / "leansearchv2" / "reasoning" / "run.py"
        ),
        "llm_client_sha256": _sha256(REPO_ROOT / "src" / "leansearchv2" / "llm.py"),
        "retriever_url": args.url,
        "search_top_k": args.search_top_k,
        "filter_max_docs": args.filter_max_docs,
        "output_top_k": 100,
        "big_loop": args.big_loop,
        "branch_budget": args.branch_budget,
        "problem_parallelism": args.parallelism,
        "retriever_timeout_s": args.retriever_timeout,
        "role_profiles": role_profiles,
        "llm_protocol": llm_protocol,
        "retriever_search_kwargs": retriever_search_kwargs,
        "retriever_protocol": retriever_protocol,
        "pricing_usd_per_mtok": {
            "input": args.input_cost_per_mtok,
            "output": args.output_cost_per_mtok,
            "cache_write": args.cache_write_cost_per_mtok,
            "cache_read": args.cache_read_cost_per_mtok,
            "basis": args.pricing_basis,
        },
    }
    immutable_fingerprint = _json_fingerprint(immutable_run)
    manifest_path = output_dir / "run_manifest.json"
    if args.resume and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("immutable_fingerprint") != immutable_fingerprint:
            raise ValueError(
                "resume run manifest does not match current data/config/code/protocol"
            )
    elif args.resume and records:
        raise ValueError("resume artifact has records but no run_manifest.json")
    else:
        _atomic_write_text(
            manifest_path,
            json.dumps(
                {
                    "schema_version": 1,
                    "started_unix_s": time.time(),
                    "launch_command": shlex.join(sys.argv),
                    "python": platform.python_version(),
                    "git_commit": _git_commit(),
                    "immutable_fingerprint": immutable_fingerprint,
                    "immutable": immutable_run,
                    "retriever_health": health,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
    sem = asyncio.Semaphore(args.parallelism)
    tasks = [
        asyncio.create_task(
            _run_one(
                row,
                retriever,
                search_top_k=args.search_top_k,
                filter_max_docs=args.filter_max_docs,
                big_loop=args.big_loop,
                branch_budget=args.branch_budget,
                retriever_search_kwargs=retriever_search_kwargs,
                sem=sem,
                prices=prices,
            )
        )
        for row in pending_rows
    ]
    order = {row["id"]: index for index, row in enumerate(rows)}
    for future in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="reasoning"):
        records.append(await future)
        records.sort(key=lambda record: order[record["id"]])
        _write_records(per_query_path, records)
        checkpoint = {
            "target_n": len(rows),
            "completed_n": len(records),
            "pending_n": len(rows) - len(records),
            "branch_budget": args.branch_budget,
            "completed_problem_ids": [record["id"] for record in records],
            "updated_unix_s": time.time(),
        }
        _atomic_write_text(
            output_dir / "checkpoint.json",
            json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
        )

    records.sort(key=lambda record: order[record["id"]])
    _write_records(per_query_path, records)

    score_rows = [
        {
            "recall_group": {k: record["scores"][str(k)]["recall_group"] for k in KS},
            "covered": {k: record["scores"][str(k)]["covered"] for k in KS},
        }
        for record in records
    ]
    metrics = aggregate(score_rows, KS)
    usage = _sum_usage([record["api_usage"] for record in records])
    total_cost = _cost(usage, **prices)
    summary = {
        "benchmark": "MathlibMPR",
        "evaluation_scope": "paper_aligned_table_2_metrics",
        "sample_selection": {
            "method": "contiguous benchmark-order slice",
            "offset": args.offset,
            "limit": args.limit,
            "problem_ids": [row["id"] for row in rows],
        },
        "n": len(records),
        "target_n": len(rows),
        "complete": len(records) == len(rows),
        "resumed_records": resumed_records,
        "errors": sum("error" in record for record in records),
        "metrics_percent": metrics,
        "api_usage": usage,
        "api_cost_usd": total_cost,
        "avg_api_cost_per_problem_usd": total_cost / len(records),
        "projected_69_problem_cost_usd": total_cost / len(records) * len(all_rows),
        "pricing_usd_per_mtok": {
            "input": args.input_cost_per_mtok,
            "output": args.output_cost_per_mtok,
            "cache_write_5m": args.cache_write_cost_per_mtok,
            "cache_read": args.cache_read_cost_per_mtok,
            "pricing_basis": args.pricing_basis,
        },
        "protocol": {
            "retriever_url": args.url,
            "retriever_health": health,
            "search_top_k": args.search_top_k,
            "filter_max_docs": args.filter_max_docs,
            "output_top_k": 100,
            "big_loop": args.big_loop,
            "branch_budget": args.branch_budget,
            "branch_policy": (
                "parallel branches; first judge-accepted branch wins and cancels "
                "unfinished siblings; pool final filtered lists if all branches fail"
            ),
            "problem_parallelism": args.parallelism,
            "role_profiles": role_profiles,
            "llm_protocol": llm_protocol,
            "retriever_search_kwargs": retriever_search_kwargs,
            "retriever_protocol": retriever_protocol,
        },
        "provenance": {
            "git_commit": _git_commit(),
            "run_manifest": str(manifest_path),
            "immutable_fingerprint": immutable_fingerprint,
            "command": shlex.join(sys.argv),
            "python": platform.python_version(),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "reasoning_runner_sha256": _sha256(
                REPO_ROOT / "src" / "leansearchv2" / "reasoning" / "run.py"
            ),
            "llm_client_sha256": _sha256(
                REPO_ROOT / "src" / "leansearchv2" / "llm.py"
            ),
            "benchmark_path": str(benchmark_path),
            "benchmark_sha256": _sha256(benchmark_path),
        },
    }
    _atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        output_dir / "checkpoint.json",
        json.dumps(
            {
                "target_n": len(rows),
                "completed_n": len(records),
                "pending_n": len(rows) - len(records),
                "branch_budget": args.branch_budget,
                "completed_problem_ids": [record["id"] for record in records],
                "updated_unix_s": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed rows in OUTPUT/per_query.jsonl and checkpoint each new row",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=69)
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Problems in flight; each problem already launches branch-budget paths",
    )
    parser.add_argument(
        "--branch-budget",
        type=int,
        default=2,
        help="Independent reasoning branches per problem (paper uses 2)",
    )
    parser.add_argument("--search-top-k", type=int, default=50)
    parser.add_argument(
        "--filter-max-docs",
        type=int,
        default=50,
        help="Maximum retrieved documents shown to the LLM filter per query",
    )
    parser.add_argument("--big-loop", type=int, default=3)
    parser.add_argument("--retriever-timeout", type=float, default=600)
    parser.add_argument("--input-cost-per-mtok", type=float, default=2.0)
    parser.add_argument("--output-cost-per-mtok", type=float, default=10.0)
    parser.add_argument("--cache-write-cost-per-mtok", type=float, default=2.5)
    parser.add_argument("--cache-read-cost-per-mtok", type=float, default=0.2)
    parser.add_argument(
        "--pricing-basis",
        default="Google Cloud global promotional price through 2026-08-31",
    )
    args = parser.parse_args()
    if (
        args.offset < 0
        or args.limit < 1
        or args.parallelism < 1
        or args.branch_budget < 1
        or args.filter_max_docs < 1
    ):
        parser.error(
            "offset must be non-negative; limit, parallelism, branch-budget, "
            "and filter-max-docs "
            "must be positive"
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
