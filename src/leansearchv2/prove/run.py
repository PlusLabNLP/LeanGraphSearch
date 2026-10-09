"""Prove-task orchestrator: simple reflection loop, one theorem -> verified proof.

Public entry point: `run_prove(problem, llm, retriever, verifier, ...)`.

Flow per problem:

    # 1. Optional pre-retrieval (only for retriever_mode='reasoning', which
    #    keys on the theorem statement; standard/none defer to reflection).
    if retriever_mode == 'reasoning':
        pre = await run_reasoning(problem, retriever, llm)
        initial_search = format(pre.entries)
    elif retriever_mode == 'standard_initial':
        queries = await get_init_queries(llm, problem)
        initial_search = format(await batch_search(retriever, queries))
    else:
        initial_search = ""

    # 2. Initial attempt.
    proof = await prover_init(llm, problem, initial_search)
    result = await verifier.verify(proof)
    if result.success: return SUCCESS

    # 3. Reflection rounds.
    for _ in range(reflection_rounds):
        if retriever_mode in ('standard', 'graph'):
            queries = await get_reflect_queries(llm, problem, proof, result.error_msg)
            reflect_search = format(await batch_search(
                retriever, queries, graph_augment=(retriever_mode == 'graph')
            ))
        elif retriever_mode == 'reasoning':
            reflect_search = initial_search  # keep the pre-retrieval
        else:
            reflect_search = ""
        proof = await prover_reflect(llm, problem, proof, result.error_msg, reflect_search)
        result = await verifier.verify(proof)
        if result.success: return SUCCESS
    return FAIL

`retriever_mode` values:
- ``"none"`` — no retrieval at all (baseline (i) in Table 3).
- ``"standard"`` — query is generated from the prover's current attempt /
  error trace; fires only during reflection (matches paper's prose for
  the semantic search engines).
- ``"graph"`` — the same reflection-only query contract as ``"standard"``,
  but each query opts into graph augmentation and rank fusion.
- ``"reasoning"`` — runs reasoning mode once on the theorem statement
  up front; same retrieval payload reused throughout reflection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLMClient
from ..standard_client import SearchResult, StandardClient
from ..reasoning import ReasoningLLMs, run_reasoning_budgeted
from ..reasoning.decompose import _parse_json
from ..reasoning.filter import _pretty_inline
from ..reasoning.run import Problem as ReasoningProblem
from ..imo_eval.failure_memory import (
    ProblemFailureMemory,
    formal_failure_state_from_source,
)
from ..imo_eval.failure_state import normalized_query, proof_state_fingerprint
from . import prompts
from .integrity import (
    IntegrityResult,
    assemble_generated_proof_body,
    extract_generated_proof_body,
    validate_proof_integrity,
)
from .verifier import Verifier, VerifyResult, extract_lean_code
from .budget import A2CallBudget, BudgetExhausted


log = logging.getLogger("leansearchv2.prove")

_DEFAULT_NUM_QUERIES = 5


@dataclass
class ProveLLMs:
    """LLM bundle for the prove task. `prover` handles initial/reflect proof
    generation; `query` generates reflect-time queries when retriever_mode
    is "standard"; `reasoning` is required when retriever_mode is
    "reasoning"."""
    prover: LLMClient
    query: LLMClient
    reasoning: ReasoningLLMs | None = None
    reasoning_branches: list[ReasoningLLMs] = field(default_factory=list)

    @classmethod
    def from_config(cls, *, with_reasoning: bool = False) -> "ProveLLMs":
        from ..config import get
        prover_name = str(get("PROVE_PROVER_LLM", "prove", "prover_llm", default="openai"))
        query_name = str(get("PROVE_QUERY_LLM", "prove", "query_llm", default=prover_name))
        cache: dict[str, LLMClient] = {}
        def _get(name: str) -> LLMClient:
            if name not in cache:
                cache[name] = LLMClient(name)
            return cache[name]
        return cls(
            prover=_get(prover_name),
            query=_get(query_name),
            reasoning=ReasoningLLMs.from_config() if with_reasoning else None,
        )


@dataclass
class ProveProblem:
    problem_id: str
    formal_statement: str
    """Full Lean source (with imports, namespaces, and a `:= by sorry` to fill in)."""
    informal_statement: str = ""
    header: str = ""
    """Optional prefix prepended to the prover's output when it omits `import` lines."""
    proof_prefix: str | None = None
    """Exact source through the target theorem's existing ``:= by``."""
    proof_suffix: str = ""
    """Exact source after the generated tactic body, such as ``end Namespace``."""

    @property
    def immutable_proof_prefix(self) -> str:
        # FATE-H already supplies a source prefix ending in ``:= by``.  The
        # explicit field is needed for full MathlibMPR files with ``:= sorry``
        # and immutable namespace-closing commands after the proof hole.
        return self.formal_statement if self.proof_prefix is None else self.proof_prefix


@dataclass
class ProveAttempt:
    round: int
    proof: str
    success: bool
    error_msg: str = ""
    integrity_valid: bool | None = None
    integrity_error: str = ""


@dataclass
class ProveResult:
    problem_id: str
    success: bool
    rounds_used: int
    final_proof: str
    final_error: str
    attempts: list[ProveAttempt] = field(default_factory=list)
    retriever_mode: str = ""
    statement_prefix_valid: bool = False
    statement_suffix_valid: bool = True
    proof_integrity_valid: bool | None = None
    validation_mode: str = "paper_raw"
    budget: dict[str, int] | None = None
    reasoning_trace: dict[str, Any] | None = None
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    failure_memory: dict[str, Any] | None = None
    failure_memory_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def strict_success(self) -> bool:
        """Success that also obeys the prompt's immutable-prefix contract.

        The upstream Table-3 verifier intentionally remains the source of the
        raw ``success`` value (Lean compiles and reports no ``sorry``).  This
        separate audit flag catches generated files that compile only after
        inserting declarations before, or otherwise rewriting, the supplied
        formal statement.
        """
        return (
            self.success
            and self.statement_prefix_valid
            and self.statement_suffix_valid
            and self.proof_integrity_valid is not False
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.problem_id,
            "success": self.success,
            "rounds_used": self.rounds_used,
            "retriever_mode": self.retriever_mode,
            "final_proof": self.final_proof,
            "final_error": self.final_error,
            "statement_prefix_valid": self.statement_prefix_valid,
            "statement_suffix_valid": self.statement_suffix_valid,
            "proof_integrity_valid": self.proof_integrity_valid,
            "validation_mode": self.validation_mode,
            "strict_success": self.strict_success,
            "budget": self.budget,
            "reasoning_trace": self.reasoning_trace,
            "retrieval_trace": self.retrieval_trace,
            "failure_memory": self.failure_memory,
            "failure_memory_events": self.failure_memory_events,
            "attempts": [
                {
                    "round": a.round,
                    "proof": a.proof,
                    "success": a.success,
                    "error_msg": a.error_msg,
                    "integrity_valid": a.integrity_valid,
                    "integrity_error": a.integrity_error,
                }
                for a in self.attempts
            ],
        }


def _format_search_results(results: list[SearchResult]) -> str:
    if not results:
        return "(No search results provided.)"
    return "\n\n".join(f"[{i}] {_pretty_inline(r)}" for i, r in enumerate(results))


async def _get_queries(
    llm: LLMClient,
    *,
    formal_statement: str,
    informal_statement: str,
    proof: str | None,
    error_msg: str | None,
    num_queries: int,
    strict_errors: bool = False,
    strict_integrity: bool = False,
    proof_prefix: str = "",
    proof_suffix: str = "",
) -> list[str]:
    if proof is None:
        prompt = prompts.GET_QUERY_INIT.format(
            num_queries=num_queries,
            lean_code=formal_statement,
            informal_statement=informal_statement or "(no informal description)",
        )
    else:
        query_template = (
            prompts.GET_QUERY_REFLECT_STRICT if strict_integrity
            else prompts.GET_QUERY_REFLECT
        )
        proof_for_prompt = (
            extract_generated_proof_body(proof, proof_prefix, proof_suffix)
            if strict_integrity
            else proof
        )
        prompt = query_template.format(
            num_queries=num_queries,
            lean_code=formal_statement,
            informal_statement=informal_statement or "(no informal description)",
            proof=proof_for_prompt,
            error_msg=error_msg or "(no error reported)",
        )
    try:
        response = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        data = _parse_json(response)
        queries = [str(q) for q in (data.get("queries") or [])]
        if strict_errors and len(queries) < num_queries:
            raise ValueError(
                f"query generator returned {len(queries)} queries; expected {num_queries}"
            )
        return queries[:num_queries]
    except Exception as e:
        log.warning(f"get_queries failed: {type(e).__name__}: {e}")
        if strict_errors:
            raise RuntimeError(
                f"query generation failed: {type(e).__name__}: {e}"
            ) from e
        return []


async def _batch_search(
    retriever: StandardClient,
    queries: list[str],
    top_k: int,
    concurrency: int,
    *,
    graph_augment: bool = False,
    strict_errors: bool = False,
) -> list[SearchResult]:
    nested = await _batch_search_groups(
        retriever,
        queries,
        top_k,
        concurrency,
        graph_augment=graph_augment,
        strict_errors=strict_errors,
    )
    seen: set[str] = set()
    out: list[SearchResult] = []
    for group in nested:
        for r in group:
            key = f"{'.'.join(r.result.module_name)}::{'.'.join(r.result.name)}"
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    return out


async def _batch_search_groups(
    retriever: StandardClient,
    queries: list[str],
    top_k: int,
    concurrency: int,
    *,
    graph_augment: bool = False,
    strict_errors: bool = False,
) -> list[list[SearchResult]]:
    if not queries:
        return []
    sem = asyncio.Semaphore(concurrency)

    async def one(q: str):
        async with sem:
            try:
                rows = await retriever.search(
                    q,
                    top_k=top_k,
                    graph_augment=graph_augment,
                    graph_initial_top_n=50 if graph_augment else None,
                    graph_expand_m=200 if graph_augment else None,
                    graph_final_top_k=100 if graph_augment else None,
                    rank_fusion="qwen_ppr_zscore" if graph_augment else None,
                )
                # Graph mode keeps an internal 100-candidate union so the
                # second rerank and fusion use the frozen evaluation profile.
                # The prover still receives the same top_k-per-query budget as
                # the paper-standard group.
                return rows[:top_k]
            except Exception:
                if strict_errors:
                    raise
                return []

    return await asyncio.gather(*[one(q) for q in queries])


async def _batch_search_with_failure_memory(
    retriever: StandardClient,
    queries: list[str],
    top_k: int,
    concurrency: int,
    *,
    graph_augment: bool,
    strict_errors: bool,
    memory: ProblemFailureMemory,
    cache: dict[tuple[str, str], list[SearchResult]],
) -> tuple[list[SearchResult], int, list[str], str]:
    state_fingerprint = proof_state_fingerprint(
        memory.current_proof,
        memory.current_snapshot,
    )
    keys = [
        (state_fingerprint, normalized_query(query))
        for query in queries
    ]
    missing_keys: list[tuple[str, str]] = []
    missing_queries: list[str] = []
    scheduled: set[tuple[str, str]] = set()
    suppressed = 0
    for query, key in zip(queries, keys):
        if key in cache or key in scheduled:
            suppressed += 1
            continue
        scheduled.add(key)
        missing_keys.append(key)
        missing_queries.append(query)
    groups = await _batch_search_groups(
        retriever,
        missing_queries,
        top_k,
        concurrency,
        graph_augment=graph_augment,
        strict_errors=strict_errors,
    )
    for key, group in zip(missing_keys, groups):
        cache[key] = group
    memory.duplicate_queries_suppressed += suppressed
    merged = _dedup_results(*(cache[key] for key in keys))
    return merged, suppressed, missing_queries, state_fingerprint


def _result_key(result: SearchResult) -> str:
    return f"{'.'.join(result.result.module_name)}::{'.'.join(result.result.name)}"


def _dedup_results(*groups: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    merged: list[SearchResult] = []
    for group in groups:
        for result in group:
            key = _result_key(result)
            if key in seen:
                continue
            seen.add(key)
            merged.append(result)
    return merged


def _format_reasoning_sketch(trace: dict[str, Any] | None) -> str:
    if not trace:
        return "(No reasoning sketch provided.)"
    steps = list(trace.get("informal_steps") or [])
    if not steps:
        return "(No reasoning sketch provided.)"
    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        description = str(step.get("description") or "").strip()
        reasoning = str(step.get("reasoning") or "").strip()
        lines.append(f"Step {index}: {description}")
        if reasoning:
            lines.append(f"  Rationale: {reasoning}")
    return "\n".join(lines)


async def _prove_call(
    llm: LLMClient,
    *,
    lean_code: str,
    search_results_str: str,
    failing_proof: str | None = None,
    error_msg: str | None = None,
    strict_integrity: bool = False,
    proof_prefix: str = "",
    proof_suffix: str = "",
    failure_memory_enabled: bool = False,
) -> str:
    memory_guidance = ""
    if failure_memory_enabled:
        memory_guidance = (
            prompts.FAILURE_MEMORY_GUIDANCE_STRICT if strict_integrity
            else prompts.FAILURE_MEMORY_GUIDANCE
        )
    if failing_proof is None:
        if strict_integrity:
            prompt = prompts.PROVER_INIT_STRICT.format(
                proof_prefix=proof_prefix,
                proof_suffix=proof_suffix,
                search_results=search_results_str,
                integrity_rules=prompts.PROOF_INTEGRITY_RULES,
                failure_memory_guidance=memory_guidance,
            )
        else:
            prompt = prompts.PROVER_INIT.format(
                lean_code=lean_code, search_results=search_results_str,
            )
            if failure_memory_enabled:
                prompt += "\n\n" + memory_guidance
    else:
        if strict_integrity:
            prompt = prompts.PROVER_REFLECT_STRICT.format(
                proof_prefix=proof_prefix,
                proof_suffix=proof_suffix,
                proof=extract_generated_proof_body(
                    failing_proof, proof_prefix, proof_suffix,
                ),
                error_msg=error_msg or "",
                search_results=search_results_str,
                integrity_rules=prompts.PROOF_INTEGRITY_RULES,
                failure_memory_guidance=memory_guidance,
            )
        else:
            prompt = prompts.PROVER_REFLECT.format(
                proof=failing_proof,
                error_msg=error_msg or "",
                search_results=search_results_str,
                lean_code=lean_code,
            )
            if failure_memory_enabled:
                prompt += "\n\n" + memory_guidance
    response = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
    code = extract_lean_code(response, preserve_indentation=strict_integrity)
    generated = code or response
    if strict_integrity:
        return assemble_generated_proof_body(
            generated, proof_prefix, proof_suffix,
        )
    return generated


async def _verify_candidate(
    verifier: Verifier,
    proof: str,
    *,
    required_prefix: str,
    required_suffix: str,
    timeout_s: int,
    strict_integrity: bool,
) -> tuple[VerifyResult, IntegrityResult]:
    integrity = validate_proof_integrity(proof, required_prefix, required_suffix)
    if strict_integrity and not integrity.valid:
        return (
            VerifyResult(
                complete=False,
                error_msg=integrity.error_msg,
                has_sorry=bool(
                    {"sorry", "admit"}.intersection(integrity.forbidden_keywords)
                ),
                raw={"proof_integrity": integrity.to_dict()},
            ),
            integrity,
        )
    try:
        result = await verifier.verify(proof, timeout_s=timeout_s)
    except Exception as exc:
        # Pathological but syntactically valid candidates can crash the Lean
        # REPL itself (for example, by forcing an enormous numeral exponent).
        # LeanInteractVerifier has already reset the dead process at this
        # point.  Preserve the candidate as a failed attempt and let the next
        # reflection round repair it instead of aborting the whole problem.
        detail = str(exc)
        if len(detail) > 4000:
            detail = detail[:4000] + "\n...(verifier exception truncated)"
        result = VerifyResult(
            complete=False,
            error_msg=(
                f"Lean verifier process failed with {type(exc).__name__}: {detail}\n"
                "Revise the proof to avoid pathological computations or oversized terms."
            ),
            has_sorry=False,
            raw={"failure_kind": "infrastructure"},
        )
    return result, integrity


def _failure_memory_payload(
    result: VerifyResult,
    integrity: IntegrityResult,
) -> dict[str, Any]:
    raw = result.raw if isinstance(result.raw, dict) else {}
    failure_kind = str(raw.get("failure_kind") or "")
    if not integrity.valid:
        failure_kind = "rejected"
    elif failure_kind not in {"infrastructure", "configuration"}:
        failure_kind = "proof"
    return {
        "success": result.success,
        "complete": result.complete,
        "has_sorry": result.has_sorry,
        "failure_kind": failure_kind,
        "error": result.error_msg,
    }


def _with_failure_memory_feedback(
    result: VerifyResult,
    memory_payload: dict[str, Any],
) -> VerifyResult:
    rendered = json.dumps(memory_payload, ensure_ascii=False, sort_keys=True)
    separator = "\n\n" if result.error_msg else ""
    return VerifyResult(
        complete=result.complete,
        error_msg=(
            result.error_msg
            + separator
            + "failure_memory: "
            + rendered
        ),
        has_sorry=result.has_sorry,
        raw={
            **(result.raw if isinstance(result.raw, dict) else {}),
            "failure_memory": memory_payload,
        },
    )


def _record_failure_memory(
    memory: ProblemFailureMemory | None,
    events: list[dict[str, Any]],
    proof: str,
    result: VerifyResult,
    integrity: IntegrityResult,
) -> VerifyResult:
    if memory is None or result.success:
        return result
    payload = _failure_memory_payload(result, integrity)
    if payload["failure_kind"] not in {"proof", "rejected"}:
        return result
    observation = memory.record_failure(proof, payload)
    events.append({
        "type": "failure_memory_observation",
        "state": observation["state"],
        "modification": observation["modification"],
        "occurrence": observation["occurrence"],
        "repeated": observation["repeated"],
    })
    if observation["repeated"]:
        events.append({
            "type": "failure_memory_dead_end_prompt",
            "state_fingerprint": observation["state"]["state_fingerprint"],
            "occurrence": observation["occurrence"],
            "summary": observation["dead_end_summary"],
        })
    return _with_failure_memory_feedback(result, observation["dead_end_summary"])


def _note_failure_memory_success(
    memory: ProblemFailureMemory | None,
    events: list[dict[str, Any]],
    proof: str,
) -> None:
    if memory is None:
        return
    success = memory.note_success(proof)
    events.append({"type": "failure_memory_success", **success})


async def run_prove(
    problem: ProveProblem,
    llms: ProveLLMs | LLMClient,
    retriever: StandardClient | None,
    verifier: Verifier,
    *,
    retriever_mode: str = "standard",  # none | standard | graph | reasoning variants
    reflection_rounds: int = 8,
    num_queries: int = _DEFAULT_NUM_QUERIES,
    search_top_k: int = 10,
    retriever_concurrency: int = 8,
    reasoning_search_top_k: int = 30,
    reasoning_big_loop: int = 3,
    reasoning_output_top_k: int = 30,
    reasoning_filter_max_docs: int = 20,
    verify_timeout_s: int = 600,
    strict_integrity: bool = False,
    strict_retrieval_errors: bool = False,
    call_budget: A2CallBudget | None = None,
    failure_memory_enabled: bool = False,
    failure_memory_repeat_threshold: int = 2,
    failure_memory_summary_max_states: int = 3,
    failure_memory_deduplicate_queries: bool = True,
    failure_memory_block_exact_proofs: bool = True,
) -> ProveResult:
    fixed_reasoning_modes = ("reasoning", "graph_reasoning")
    adaptive_reasoning_modes = ("reasoning_adaptive", "graph_reasoning_adaptive")
    reasoning_modes = fixed_reasoning_modes + adaptive_reasoning_modes
    if retriever_mode not in ("none", "standard", "graph", *reasoning_modes):
        raise ValueError(f"unknown retriever_mode: {retriever_mode}")
    if retriever_mode != "none" and retriever is None:
        raise ValueError(f"retriever_mode={retriever_mode!r} requires a StandardClient")
    if isinstance(llms, LLMClient):
        llms = ProveLLMs(prover=llms, query=llms, reasoning=ReasoningLLMs(sketch=llms, filter=llms, judge=llms))
    if retriever_mode in reasoning_modes and not (
        llms.reasoning is not None or llms.reasoning_branches
    ):
        raise ValueError(f"retriever_mode={retriever_mode!r} requires reasoning LLMs")
    if failure_memory_repeat_threshold < 2:
        raise ValueError("failure_memory_repeat_threshold must be >= 2")
    if failure_memory_summary_max_states < 1:
        raise ValueError("failure_memory_summary_max_states must be >= 1")

    failure_memory: ProblemFailureMemory | None = None
    failure_memory_events: list[dict[str, Any]] = []
    failure_memory_query_cache: dict[
        tuple[str, str], list[SearchResult]
    ] = {}
    if failure_memory_enabled:
        failure_memory = ProblemFailureMemory(
            problem=problem,  # type: ignore[arg-type]
            repeat_threshold=failure_memory_repeat_threshold,
            summary_max_states=failure_memory_summary_max_states,
            state_builder=formal_failure_state_from_source,
        )

    # Pre-retrieval.
    initial_results: list[SearchResult] = []
    reasoning_trace: dict[str, Any] | None = None
    retrieval_trace: list[dict[str, Any]] = []
    if retriever_mode in reasoning_modes:
        reasoning_problem = ReasoningProblem(
            problem_id=problem.problem_id,
            formal_statement=problem.formal_statement,
            informal_statement=problem.informal_statement,
        )
        graph_reasoning_kwargs = None
        if retriever_mode in ("graph_reasoning", "graph_reasoning_adaptive"):
            graph_reasoning_kwargs = {
                "graph_augment": True,
                "return_metadata": True,
                "graph_initial_top_n": 50,
                "graph_expand_m": 200,
                "graph_final_top_k": 100,
                "rank_fusion": "qwen_ppr_zscore",
            }
        try:
            branches = list(llms.reasoning_branches)
            if not branches and llms.reasoning is not None:
                branches = [llms.reasoning]
            pre = await run_reasoning_budgeted(
                reasoning_problem,
                retriever,  # type: ignore[arg-type]
                branches,
                search_top_k=reasoning_search_top_k,
                output_top_k=reasoning_output_top_k,
                big_loop=reasoning_big_loop,
                filter_max_docs=reasoning_filter_max_docs,
                retriever_search_kwargs=graph_reasoning_kwargs,
                strict_retrieval_errors=strict_retrieval_errors,
            )
            initial_results = [r for _did, _score, r in pre.entries]
            reasoning_trace = pre.to_dict()
        except Exception as e:
            log.warning(f"[{problem.problem_id}] reasoning prefetch failed: {e}")
            if strict_retrieval_errors:
                raise
            reasoning_trace = {"error": f"{type(e).__name__}: {e}"}

    initial_search_str = _format_search_results(initial_results)
    adaptive_reasoning = retriever_mode in adaptive_reasoning_modes
    reasoning_sketch_str = _format_reasoning_sketch(reasoning_trace)
    if adaptive_reasoning:
        initial_prompt_context = (
            "## Reasoning Proof Sketch\n\n"
            f"{reasoning_sketch_str}\n\n"
            "## Initial Reasoning Premises\n\n"
            f"{initial_search_str}"
        )
    else:
        initial_prompt_context = initial_search_str
    retrieval_trace.append({
        "round": 0,
        "source": "reasoning_prefetch" if retriever_mode in reasoning_modes else "none",
        "initial_reasoning_docs": len(initial_results),
        "dynamic_queries": [],
        "dynamic_docs": 0,
        "combined_unique_docs": len(initial_results),
    })

    # Initial attempt.
    attempts: list[ProveAttempt] = []
    try:
        if call_budget is not None and call_budget.remaining_compiler() < 1:
            raise BudgetExhausted("compiler-call budget exhausted before initial proof")
        proof = await _prove_call(
            llms.prover, lean_code=problem.formal_statement, search_results_str=initial_prompt_context,
            strict_integrity=strict_integrity,
            proof_prefix=problem.immutable_proof_prefix,
            proof_suffix=problem.proof_suffix,
            failure_memory_enabled=failure_memory_enabled,
        )
    except BudgetExhausted as exc:
        return ProveResult(
            problem_id=problem.problem_id, success=False, rounds_used=0,
            final_proof="", final_error=str(exc), attempts=[], retriever_mode=retriever_mode,
            validation_mode="integrity_strict" if strict_integrity else "paper_raw",
            budget=call_budget.snapshot() if call_budget is not None else None,
            reasoning_trace=reasoning_trace,
            retrieval_trace=retrieval_trace,
            failure_memory=failure_memory.summary() if failure_memory else None,
            failure_memory_events=failure_memory_events,
        )
    if proof and problem.header and not proof.strip().startswith("import"):
        proof = problem.header + "\n\n" + proof
    if call_budget is not None:
        await call_budget.reserve_compiler()
    result, integrity = await _verify_candidate(
        verifier,
        proof,
        required_prefix=problem.immutable_proof_prefix,
        required_suffix=problem.proof_suffix,
        timeout_s=verify_timeout_s,
        strict_integrity=strict_integrity,
    )
    result = _record_failure_memory(
        failure_memory,
        failure_memory_events,
        proof,
        result,
        integrity,
    )
    attempts.append(ProveAttempt(
        round=0,
        proof=proof,
        success=result.success,
        error_msg=result.error_msg,
        integrity_valid=integrity.valid,
        integrity_error=integrity.error_msg,
    ))
    if result.success:
        _note_failure_memory_success(failure_memory, failure_memory_events, proof)
        return ProveResult(
            problem_id=problem.problem_id, success=True, rounds_used=0,
            final_proof=proof, final_error="", attempts=attempts,
            retriever_mode=retriever_mode,
            statement_prefix_valid=integrity.statement_prefix_valid,
            statement_suffix_valid=integrity.statement_suffix_valid,
            proof_integrity_valid=integrity.valid,
            validation_mode="integrity_strict" if strict_integrity else "paper_raw",
            budget=call_budget.snapshot() if call_budget is not None else None,
            reasoning_trace=reasoning_trace,
            retrieval_trace=retrieval_trace,
            failure_memory=failure_memory.summary() if failure_memory else None,
            failure_memory_events=failure_memory_events,
        )

    # Reflection rounds.  Failure-memory blocks exact deterministic failures
    # without spending compiler budget, so compiler round and model-call index
    # are intentionally distinct.  Continue until an actual compiler budget,
    # model budget, or the requested compiler-round cap is reached.
    max_compiler_attempts = reflection_rounds + 1
    generated_candidates = 1
    prepared_round: int | None = None
    search_str = initial_prompt_context
    dynamic_search_mode = retriever_mode in (
        "standard", "graph", "reasoning_adaptive", "graph_reasoning_adaptive"
    )
    while len(attempts) < max_compiler_attempts:
        if call_budget is not None and (
            call_budget.remaining_model() < 1
            or call_budget.remaining_compiler() < 1
        ):
            break
        if call_budget is None and generated_candidates >= max_compiler_attempts:
            break

        r_idx = len(attempts)
        if prepared_round != r_idx:
            can_query = dynamic_search_mode and (
                call_budget is None
                or (
                    call_budget.remaining_query() >= 1
                    and call_budget.remaining_model() >= 2
                )
            )
            if can_query:
                if call_budget is not None:
                    await call_budget.reserve_query()
                queries = await _get_queries(
                    llms.query,
                    formal_statement=problem.formal_statement,
                    informal_statement=problem.informal_statement,
                    proof=proof,
                    error_msg=result.error_msg,
                    num_queries=num_queries,
                    strict_errors=strict_retrieval_errors,
                    strict_integrity=strict_integrity,
                    proof_prefix=problem.immutable_proof_prefix,
                    proof_suffix=problem.proof_suffix,
                )
                suppressed_queries = 0
                actual_search_queries = list(queries)
                memory_state_fingerprint: str | None = None
                if failure_memory is not None and failure_memory_deduplicate_queries:
                    (
                        search_results,
                        suppressed_queries,
                        actual_search_queries,
                        memory_state_fingerprint,
                    ) = await _batch_search_with_failure_memory(
                        retriever,  # type: ignore[arg-type]
                        queries,
                        top_k=search_top_k,
                        concurrency=retriever_concurrency,
                        graph_augment=retriever_mode
                        in ("graph", "graph_reasoning_adaptive"),
                        strict_errors=strict_retrieval_errors,
                        memory=failure_memory,
                        cache=failure_memory_query_cache,
                    )
                    if suppressed_queries:
                        failure_memory_events.append({
                            "type": "duplicate_query_suppressed",
                            "round": r_idx,
                            "mode": "free_failure_memory",
                            "count": suppressed_queries,
                            "proof_state_fingerprint": memory_state_fingerprint,
                            "queries": list(queries),
                            "actual_search_queries": list(actual_search_queries),
                        })
                else:
                    search_results = await _batch_search(
                        retriever,  # type: ignore[arg-type]
                        queries,
                        top_k=search_top_k,
                        concurrency=retriever_concurrency,
                        graph_augment=retriever_mode
                        in ("graph", "graph_reasoning_adaptive"),
                        strict_errors=strict_retrieval_errors,
                    )
                if adaptive_reasoning:
                    combined_results = _dedup_results(initial_results, search_results)
                    search_str = (
                        "## Fixed Reasoning Proof Sketch\n\n"
                        f"{reasoning_sketch_str}\n\n"
                        "## Initial Reasoning Premises and Current Reflection Search Results\n\n"
                        f"{_format_search_results(combined_results)}"
                    )
                else:
                    combined_results = search_results
                    search_str = _format_search_results(search_results)
                retrieval_trace.append({
                    "round": r_idx,
                    "source": "adaptive_graph_search"
                    if retriever_mode in ("graph", "graph_reasoning_adaptive")
                    else "adaptive_standard_search",
                    "initial_reasoning_docs": len(initial_results)
                    if adaptive_reasoning
                    else 0,
                    "dynamic_queries": list(queries),
                    "actual_search_queries": list(actual_search_queries),
                    "duplicate_queries_suppressed": suppressed_queries,
                    "failure_memory_state_fingerprint": memory_state_fingerprint,
                    "dynamic_docs": len(search_results),
                    "combined_unique_docs": len(combined_results),
                })
            elif adaptive_reasoning:
                search_str = initial_prompt_context
                retrieval_trace.append({
                    "round": r_idx,
                    "source": "fixed_reasoning_after_query_budget",
                    "initial_reasoning_docs": len(initial_results),
                    "dynamic_queries": [],
                    "dynamic_docs": 0,
                    "combined_unique_docs": len(initial_results),
                })
            elif retriever_mode in fixed_reasoning_modes:
                search_str = initial_search_str
                retrieval_trace.append({
                    "round": r_idx,
                    "source": "fixed_reasoning_reuse",
                    "initial_reasoning_docs": len(initial_results),
                    "dynamic_queries": [],
                    "dynamic_docs": 0,
                    "combined_unique_docs": len(initial_results),
                })
            else:
                search_str = "(No retrieval enabled.)"
                retrieval_trace.append({
                    "round": r_idx,
                    "source": "none",
                    "initial_reasoning_docs": 0,
                    "dynamic_queries": [],
                    "dynamic_docs": 0,
                    "combined_unique_docs": 0,
                })
            prepared_round = r_idx

        try:
            proof = await _prove_call(
                llms.prover,
                lean_code=problem.formal_statement,
                search_results_str=search_str,
                failing_proof=proof,
                error_msg=result.error_msg,
                strict_integrity=strict_integrity,
                proof_prefix=problem.immutable_proof_prefix,
                proof_suffix=problem.proof_suffix,
                failure_memory_enabled=failure_memory_enabled,
            )
            generated_candidates += 1
        except BudgetExhausted:
            break
        if proof and problem.header and not proof.strip().startswith("import"):
            proof = problem.header + "\n\n" + proof

        if (
            failure_memory is not None
            and failure_memory_block_exact_proofs
            and failure_memory.known_failed_proof(proof) is not None
        ):
            blocked = failure_memory.block_exact_failure(proof)
            failure_memory_events.append({
                "type": "failure_memory_action_blocked",
                "round": r_idx,
                **blocked,
            })
            integrity = validate_proof_integrity(
                proof,
                problem.immutable_proof_prefix,
                problem.proof_suffix,
            )
            result = VerifyResult(
                complete=False,
                error_msg=blocked["directive"],
                has_sorry=False,
                raw={"failure_kind": "proof", "cached": True},
            )
            memory_summary = failure_memory.dead_end_summary(
                latest_fingerprint=blocked["state_fingerprint"]
            )
            result = _with_failure_memory_feedback(
                result,
                {"prevented_repeated_action": True, **blocked, **memory_summary},
            )
            # Keep the already prepared retrieval context for the same formal
            # compiler round.  The next model call must submit a different
            # proof and does not receive another automatic query action.
            continue

        if call_budget is not None:
            await call_budget.reserve_compiler()
        result, integrity = await _verify_candidate(
            verifier,
            proof,
            required_prefix=problem.immutable_proof_prefix,
            required_suffix=problem.proof_suffix,
            timeout_s=verify_timeout_s,
            strict_integrity=strict_integrity,
        )
        result = _record_failure_memory(
            failure_memory,
            failure_memory_events,
            proof,
            result,
            integrity,
        )
        attempts.append(ProveAttempt(
            round=r_idx,
            proof=proof,
            success=result.success,
            error_msg=result.error_msg,
            integrity_valid=integrity.valid,
            integrity_error=integrity.error_msg,
        ))
        prepared_round = None
        if result.success:
            _note_failure_memory_success(failure_memory, failure_memory_events, proof)
            return ProveResult(
                problem_id=problem.problem_id,
                success=True,
                rounds_used=r_idx,
                final_proof=proof,
                final_error="",
                attempts=attempts,
                retriever_mode=retriever_mode,
                statement_prefix_valid=integrity.statement_prefix_valid,
                statement_suffix_valid=integrity.statement_suffix_valid,
                proof_integrity_valid=integrity.valid,
                validation_mode="integrity_strict" if strict_integrity else "paper_raw",
                budget=call_budget.snapshot() if call_budget is not None else None,
                reasoning_trace=reasoning_trace,
                retrieval_trace=retrieval_trace,
                failure_memory=failure_memory.summary() if failure_memory else None,
                failure_memory_events=failure_memory_events,
            )

    return ProveResult(
        problem_id=problem.problem_id,
        success=False,
        rounds_used=max(0, len(attempts) - 1),
        final_proof=proof,
        final_error=result.error_msg,
        attempts=attempts,
        retriever_mode=retriever_mode,
        statement_prefix_valid=integrity.statement_prefix_valid,
        statement_suffix_valid=integrity.statement_suffix_valid,
        proof_integrity_valid=integrity.valid,
        validation_mode="integrity_strict" if strict_integrity else "paper_raw",
        budget=call_budget.snapshot() if call_budget is not None else None,
        reasoning_trace=reasoning_trace,
        retrieval_trace=retrieval_trace,
        failure_memory=failure_memory.summary() if failure_memory else None,
        failure_memory_events=failure_memory_events,
    )
