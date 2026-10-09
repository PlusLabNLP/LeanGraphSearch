"""Reasoning-mode orchestrator.

Pure-asyncio implementation of the decompose → search → filter → judge
loop. Public entry point: `run_reasoning(problem, retriever, llm, ...)`.

Flow per problem:

    for attempt in range(big_loop + 1):
        plan = await decompose(...)           # LLM
        query_to_docs = await batch_search(...)   # HTTP (parallel)
        filtered = await filter_results(...)  # LLM (parallel)
        verdict = await judge(...)            # LLM
        if verdict == "good": status = "good"; break
        if attempt == big_loop: status = "fail"; break
        # else: keep going, feed plan+reasoning back into next decompose

The output is the union of `filtered` (dedup-by-doc-id) of the final
attempt, ranked by `aggregate_filtered_method1` (NDCG-discount summed
across sub-queries) so docs hit by multiple sub-queries surface first.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..llm import LLMClient
from ..standard_client import SearchResult, StandardClient
from .decompose import ProofPlan, decompose
from .filter import filter_results, union_unique_results
from .judge import judge


@dataclass
class ReasoningLLMs:
    """One LLMClient per reasoning-mode role; build via `from_config()` or
    pass clients directly to use the same model everywhere."""
    sketch: LLMClient
    filter: LLMClient
    judge: LLMClient

    @classmethod
    def from_config(cls) -> "ReasoningLLMs":
        from ..config import get
        sketch = str(get("REASONING_SKETCH_LLM", "reasoning", "sketch_llm", default="openai"))
        filt = str(get("REASONING_FILTER_LLM", "reasoning", "filter_llm", default="openai"))
        judge_ = str(get("REASONING_JUDGE_LLM", "reasoning", "judge_llm", default="openai"))
        cache: dict[str, LLMClient] = {}
        def _get(name: str) -> LLMClient:
            if name not in cache:
                cache[name] = LLMClient(name)
            return cache[name]
        return cls(sketch=_get(sketch), filter=_get(filt), judge=_get(judge_))


@dataclass
class Problem:
    problem_id: str
    formal_statement: str
    informal_statement: str = ""
    informal_proof: str = ""


@dataclass
class ReasoningResult:
    problem_id: str
    status: str  # "good" | "bad" | "fail"
    big_loop_count: int
    plan: ProofPlan
    filtered: dict[str, list[SearchResult]]
    filter_reasoning: str
    quality_reasoning: str
    entries: list[tuple[str, float, SearchResult]]
    """final ranked output: (doc_id, score, result) tuples, length <= output_top_k"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.problem_id,
            "sketch_status": self.status,
            "big_loop_count": self.big_loop_count,
            "definition_queries": self.plan.definition_queries,
            "highlevel_queries": self.plan.highlevel_queries,
            "informal_steps": [
                {
                    "step_id": i,
                    "description": s.get("description", ""),
                    "reasoning": s.get("reasoning", ""),
                    "queries": s.get("queries", []),
                }
                for i, s in enumerate(self.plan.steps)
            ],
            "filter_reasoning": self.filter_reasoning,
            "quality_reasoning": self.quality_reasoning,
            "entries": [
                {
                    "doc_id": did,
                    "score": score,
                    "name": ".".join(r.result.name),
                    "module": ".".join(r.result.module_name),
                    "kind": r.result.kind,
                    "informal_name": r.result.informal_name,
                    "distance": r.distance,
                }
                for did, score, r in self.entries
            ],
        }


@dataclass
class BudgetedReasoningResult:
    """Paper-style multi-branch reasoning result.

    MathlibMPR Table 2 uses two branches per query.  The first accepted
    branch supplies the output and stops any still-running sibling.  If no
    branch is accepted, the final filtered lists from every completed branch
    are pooled before applying the same rank-discount aggregation rule.
    """

    result: ReasoningResult
    branch_budget: int
    winning_branch: int | None
    branch_statuses: list[str]
    branch_results: list[ReasoningResult | None]
    branch_errors: list[str | None]

    @property
    def entries(self) -> list[tuple[str, float, SearchResult]]:
        return self.result.entries

    @property
    def status(self) -> str:
        return self.result.status

    def to_dict(self) -> dict[str, Any]:
        payload = self.result.to_dict()
        payload.update(
            {
                "branch_budget": self.branch_budget,
                "winning_branch": self.winning_branch,
                "branch_statuses": list(self.branch_statuses),
                "branch_errors": list(self.branch_errors),
                "branch_results": [
                    branch.to_dict() if branch is not None else None
                    for branch in self.branch_results
                ],
            }
        )
        return payload


def _step_desc_for_query(plan: ProofPlan, query: str) -> str:
    """Pick a human-readable step description to feed the filter prompt."""
    for q in plan.definition_queries:
        if q == query:
            return "Core definition used in the theorem statement"
    for q in plan.highlevel_queries:
        if q == query:
            return "High-level theorem that might directly bridge assumptions to conclusion"
    for i, s in enumerate(plan.steps):
        if query in (s.get("queries") or []):
            return f"Step {i}: {s.get('description', '')}"
    return "(no step context)"


async def _batch_search(
    retriever: StandardClient,
    queries: list[str],
    *,
    top_k: int,
    rerank: bool,
    concurrency: int,
    search_kwargs: dict[str, Any] | None = None,
    strict_errors: bool = False,
) -> dict[str, list[SearchResult]]:
    """Fan out one /search per query (concurrency-bounded). Returns
    `{query: results}`. Empty-results queries map to `[]`."""
    if not queries:
        return {}
    sem = asyncio.Semaphore(concurrency)

    async def one(q: str) -> tuple[str, list[SearchResult]]:
        async with sem:
            try:
                docs = await retriever.search(
                    q,
                    top_k=top_k,
                    rerank=rerank,
                    **dict(search_kwargs or {}),
                )
                # Graph search may build and return a larger internal pool
                # (for example graph_final_top_k=100) before exposing the
                # same top-k contract to the reasoning filter as standard
                # search.  Keep that internal budget invisible downstream.
                return q, docs[:top_k]
            except Exception:
                if strict_errors:
                    raise
                return q, []

    pairs = await asyncio.gather(*[one(q) for q in queries])
    out: dict[str, list[SearchResult]] = {}
    for q, docs in pairs:
        out[q] = docs
    return out


def _rank_method1(
    filtered: dict[str, list[SearchResult]], top_k: int,
) -> list[tuple[str, float, SearchResult]]:
    """NDCG-discount summed across sub-queries. A doc that surfaces in
    multiple sub-queries (or near the top within one) accumulates score."""
    score: dict[str, float] = {}
    doc_ref: dict[str, SearchResult] = {}
    for _q, docs in filtered.items():
        for i, d in enumerate(docs):
            doc_id = f"{'.'.join(d.result.module_name)}::{'.'.join(d.result.name)}"
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / math.log2(i + 2)
            doc_ref.setdefault(doc_id, d)
    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(did, sc, doc_ref[did]) for did, sc in ranked]


async def run_reasoning(
    problem: Problem,
    retriever: StandardClient,
    llms: ReasoningLLMs | LLMClient,
    *,
    search_top_k: int = 30,
    output_top_k: int = 100,
    big_loop: int = 3,
    retriever_concurrency: int = 8,
    retriever_rerank: bool = True,
    retriever_search_kwargs: dict[str, Any] | None = None,
    filter_max_docs: int = 20,
    before_search_batch: Callable[[], Awaitable[bool]] | None = None,
    strict_retrieval_errors: bool = False,
) -> ReasoningResult:
    """Run the GetQuery -> Search -> Filter -> Judge loop until the judge says
    "good" or we exhaust the big loop budget. Pass a `LLMClient` to use one
    model for every role, or a `ReasoningLLMs` to split sketch/filter/judge.
    """
    if isinstance(llms, LLMClient):
        llms = ReasoningLLMs(sketch=llms, filter=llms, judge=llms)
    plan: ProofPlan | None = None
    prev_reasoning: str = ""
    filtered: dict[str, list[SearchResult]] = {}
    filter_reason: str = ""
    quality_reason: str = ""
    status: str = "fail"
    big_loop_count = 0

    for attempt in range(big_loop + 1):
        plan = await decompose(
            llms.sketch,
            formal_statement=problem.formal_statement,
            informal_statement=problem.informal_statement,
            informal_proof=problem.informal_proof,
            previous_plan=plan,
            previous_quality_reasoning=prev_reasoning,
        )
        all_queries = plan.all_queries
        search_allowed = True
        if before_search_batch is not None:
            search_allowed = bool(await before_search_batch())
        if search_allowed:
            query_to_docs = await _batch_search(
                retriever,
                all_queries,
                top_k=search_top_k,
                rerank=retriever_rerank,
                concurrency=retriever_concurrency,
                search_kwargs=retriever_search_kwargs,
                strict_errors=strict_retrieval_errors,
            )
        else:
            query_to_docs = {query: [] for query in all_queries}
        query_to_step_desc = {q: _step_desc_for_query(plan, q) for q in all_queries}
        filtered, filter_reason = await filter_results(
            llms.filter,
            formal_statement=problem.formal_statement,
            query_to_docs=query_to_docs,
            query_to_step_desc=query_to_step_desc,
            max_docs_to_llm=filter_max_docs,
        )
        verdict, quality_reason = await judge(
            llms.judge,
            formal_statement=problem.formal_statement,
            plan_steps=plan.steps,
            filtered=filtered,
            filter_reasoning=filter_reason,
        )
        if verdict == "good":
            status = "good"
            break
        if attempt == big_loop:
            status = "fail"
            break
        big_loop_count += 1
        prev_reasoning = quality_reason

    entries = _rank_method1(filtered, top_k=output_top_k)
    return ReasoningResult(
        problem_id=problem.problem_id,
        status=status,
        big_loop_count=big_loop_count,
        plan=plan or ProofPlan(),
        filtered=filtered,
        filter_reasoning=filter_reason,
        quality_reasoning=quality_reason,
        entries=entries,
    )


async def run_reasoning_budgeted(
    problem: Problem,
    retriever: StandardClient,
    branch_llms: list[ReasoningLLMs | LLMClient],
    *,
    search_top_k: int = 30,
    output_top_k: int = 100,
    big_loop: int = 3,
    retriever_concurrency: int = 8,
    retriever_rerank: bool = True,
    retriever_search_kwargs: dict[str, Any] | None = None,
    filter_max_docs: int = 20,
    before_search_batch: Callable[[], Awaitable[bool]] | None = None,
    strict_retrieval_errors: bool = False,
) -> BudgetedReasoningResult:
    """Run the paper's independent-branch budget for one problem.

    Branches start concurrently.  The first branch judged ``good`` wins and
    any sibling that has not already completed is cancelled (``early_stop``).
    When no branch is accepted, all branches' final filtered lists are pooled
    and reranked with the paper's positional discount.  A distinct LLM bundle
    per branch keeps calls independent and makes per-branch usage auditable.
    """
    if not branch_llms:
        raise ValueError("branch_llms must contain at least one branch")

    branch_budget = len(branch_llms)
    branch_results: list[ReasoningResult | None] = [None] * branch_budget
    branch_errors: list[str | None] = [None] * branch_budget
    branch_statuses: list[str] = ["running"] * branch_budget
    completions: asyncio.Queue[tuple[int, ReasoningResult | None, BaseException | None]] = (
        asyncio.Queue()
    )

    async def run_branch(
        branch_index: int,
        llms: ReasoningLLMs | LLMClient,
    ) -> ReasoningResult:
        try:
            result = await run_reasoning(
                problem,
                retriever,
                llms,
                search_top_k=search_top_k,
                output_top_k=output_top_k,
                big_loop=big_loop,
                retriever_concurrency=retriever_concurrency,
                retriever_rerank=retriever_rerank,
                retriever_search_kwargs=retriever_search_kwargs,
                filter_max_docs=filter_max_docs,
                before_search_batch=before_search_batch,
                strict_retrieval_errors=strict_retrieval_errors,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await completions.put((branch_index, None, exc))
            raise
        await completions.put((branch_index, result, None))
        return result

    tasks = [
        asyncio.create_task(run_branch(index, llms), name=f"reasoning-branch-{index}")
        for index, llms in enumerate(branch_llms)
    ]
    winning_branch: int | None = None
    completed_count = 0

    try:
        while completed_count < branch_budget:
            branch_index, result, error = await completions.get()
            completed_count += 1
            if error is not None:
                branch_errors[branch_index] = f"{type(error).__name__}: {error}"
                branch_statuses[branch_index] = "error"
                continue
            assert result is not None
            branch_results[branch_index] = result
            branch_statuses[branch_index] = result.status
            if result.status != "good":
                continue

            winning_branch = branch_index
            # Give branches that completed in the same event-loop turn a
            # chance to publish their terminal result before cancellation.
            await asyncio.sleep(0)
            for index, task in enumerate(tasks):
                if index != branch_index and not task.done():
                    branch_statuses[index] = "early_stop"
                    task.cancel()
            break
    finally:
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    # A sibling may have finished between the winning queue event and the
    # cancellation sweep.  Preserve that real terminal status in the trace.
    for index, outcome in enumerate(outcomes):
        if isinstance(outcome, ReasoningResult):
            branch_results[index] = outcome
            branch_statuses[index] = outcome.status
        elif isinstance(outcome, asyncio.CancelledError):
            branch_statuses[index] = "early_stop"
        elif isinstance(outcome, BaseException) and branch_errors[index] is None:
            branch_errors[index] = f"{type(outcome).__name__}: {outcome}"
            branch_statuses[index] = "error"

    if strict_retrieval_errors and any(branch_errors):
        errors = "; ".join(error for error in branch_errors if error)
        raise RuntimeError(f"reasoning branch retrieval failed: {errors}")

    if winning_branch is not None:
        winner = branch_results[winning_branch]
        assert winner is not None
        final_result = winner
    else:
        completed_results = [result for result in branch_results if result is not None]
        if not completed_results:
            errors = "; ".join(error for error in branch_errors if error)
            raise RuntimeError(f"all reasoning branches errored: {errors}")

        pooled: dict[str, list[SearchResult]] = {}
        filter_reasoning: list[str] = []
        quality_reasoning: list[str] = []
        for branch_index, branch in enumerate(branch_results):
            if branch is None:
                continue
            for query_index, (query, docs) in enumerate(branch.filtered.items()):
                # Prefixing prevents identical query text in two independent
                # branches from overwriting one another or losing a rank vote.
                pooled[f"branch-{branch_index}:{query_index}:{query}"] = docs
            filter_reasoning.append(
                f"[branch {branch_index}]\n{branch.filter_reasoning}"
            )
            quality_reasoning.append(
                f"[branch {branch_index}]\n{branch.quality_reasoning}"
            )
        representative = completed_results[0]
        final_result = ReasoningResult(
            problem_id=problem.problem_id,
            status="fail",
            big_loop_count=max(result.big_loop_count for result in completed_results),
            plan=representative.plan,
            filtered=pooled,
            filter_reasoning="\n\n".join(filter_reasoning),
            quality_reasoning="\n\n".join(quality_reasoning),
            entries=_rank_method1(pooled, top_k=output_top_k),
        )

    return BudgetedReasoningResult(
        result=final_result,
        branch_budget=branch_budget,
        winning_branch=winning_branch,
        branch_statuses=branch_statuses,
        branch_results=branch_results,
        branch_errors=branch_errors,
    )


# Re-export for convenience
__all__ = [
    "BudgetedReasoningResult",
    "Problem",
    "ReasoningLLMs",
    "ReasoningResult",
    "run_reasoning",
    "run_reasoning_budgeted",
    "union_unique_results",
]
