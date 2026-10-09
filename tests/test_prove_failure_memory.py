from __future__ import annotations

import asyncio
import json

from leansearchv2.prove import A2CallBudget, BudgetedLLMClient
from leansearchv2.prove.run import ProveLLMs, ProveProblem, run_prove
from leansearchv2.prove.verifier import VerifyResult


PREFIX = "import Mathlib\n\ntheorem failure_memory_demo : True := by"
BAD = PREFIX + "\n  exact False.elim (by contradiction)"
GOOD = PREFIX + "\n  trivial"
ERROR = (
    "line 4, column 3: error: tactic 'contradiction' failed\n"
    "case h\n"
    "⊢ False"
)


class _SequenceLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def chat(self, messages, **_kwargs):
        self.prompts.append(messages[0]["content"])
        return self.responses.pop(0)


class _SequenceVerifier:
    def __init__(self, results: list[VerifyResult]) -> None:
        self.results = list(results)
        self.proofs: list[str] = []

    async def verify(self, code: str, timeout_s: int = 600) -> VerifyResult:
        self.proofs.append(code)
        return self.results.pop(0)


class _EmptyRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str, **_kwargs):
        self.queries.append(query)
        return []


def _failed(error: str = ERROR) -> VerifyResult:
    return VerifyResult(complete=False, error_msg=error, has_sorry=False)


def _success() -> VerifyResult:
    return VerifyResult(complete=True, error_msg="", has_sorry=False)


def _problem() -> ProveProblem:
    return ProveProblem(problem_id="failure-memory-demo", formal_statement=PREFIX)


def test_exact_failed_candidate_is_blocked_without_compiler_budget() -> None:
    llm = _SequenceLLM([BAD, BAD, GOOD])
    verifier = _SequenceVerifier([_failed(), _success()])
    budget = A2CallBudget(max_model_calls=3, max_query_calls=0, max_compiler_calls=2)
    budgeted = BudgetedLLMClient(llm, budget)

    result = asyncio.run(
        run_prove(
            _problem(),
            ProveLLMs(prover=budgeted, query=budgeted),  # type: ignore[arg-type]
            retriever=None,
            verifier=verifier,  # type: ignore[arg-type]
            retriever_mode="none",
            reflection_rounds=1,
            strict_integrity=True,
            call_budget=budget,
            failure_memory_enabled=True,
        )
    )

    assert result.success is True
    assert budget.snapshot()["model_calls"] == 3
    assert budget.snapshot()["compiler_calls"] == 2
    assert verifier.proofs == [BAD, GOOD]
    assert [attempt.round for attempt in result.attempts] == [0, 1]
    assert result.failure_memory["prevented_repeated_actions"] == 1
    assert sum(
        event["type"] == "failure_memory_action_blocked"
        for event in result.failure_memory_events
    ) == 1
    assert "exact proof already has a deterministic failed result" in llm.prompts[2]
    assert "Failure Memory" in llm.prompts[0]


def test_repeated_formal_state_prompts_materially_different_repair() -> None:
    def long_bad(name: str) -> str:
        return (
            PREFIX
            + f"\n  have {name} : True := by trivial"
            + "\n  have h2 : True := by trivial"
            + "\n  have h3 : True := by trivial"
            + "\n  have h4 : True := by trivial"
            + "\n  have h5 : True := by trivial"
            + "\n  have h6 : True := by trivial"
            + "\n  contradiction"
        )

    bad_one = long_bad("first")
    bad_two = long_bad("different")
    repeated_error = ERROR.replace("line 4", "line 10")
    llm = _SequenceLLM([bad_one, bad_two, GOOD])
    verifier = _SequenceVerifier([
        _failed(repeated_error),
        _failed(repeated_error),
        _success(),
    ])
    budget = A2CallBudget(max_model_calls=3, max_query_calls=0, max_compiler_calls=3)
    budgeted = BudgetedLLMClient(llm, budget)

    result = asyncio.run(
        run_prove(
            _problem(),
            ProveLLMs(prover=budgeted, query=budgeted),  # type: ignore[arg-type]
            retriever=None,
            verifier=verifier,  # type: ignore[arg-type]
            retriever_mode="none",
            reflection_rounds=2,
            strict_integrity=True,
            call_budget=budget,
            failure_memory_enabled=True,
        )
    )

    assert result.success is True
    prompts = [
        event
        for event in result.failure_memory_events
        if event["type"] == "failure_memory_dead_end_prompt"
    ]
    assert len(prompts) == 1
    assert prompts[0]["occurrence"] == 2
    assert "DEAD_END_REPEATED" in prompts[0]["summary"]["directive"]
    assert "DEAD_END_REPEATED" in llm.prompts[2]
    assert result.failure_memory["dead_end_prompts"] == 1


def test_query_deduplication_uses_formal_state_and_normalized_query() -> None:
    prover = _SequenceLLM([BAD, GOOD])
    queries = [
        "theorem proving True",
        " theorem   proving true ",
        "THEOREM PROVING TRUE",
        "theorem proving True",
        "theorem proving true",
    ]
    query = _SequenceLLM([
        "```json\n" + json.dumps({"queries": queries}) + "\n```"
    ])
    verifier = _SequenceVerifier([_failed(), _success()])
    retriever = _EmptyRetriever()
    budget = A2CallBudget(max_model_calls=3, max_query_calls=1, max_compiler_calls=2)
    budgeted_prover = BudgetedLLMClient(prover, budget)
    budgeted_query = BudgetedLLMClient(query, budget)

    result = asyncio.run(
        run_prove(
            _problem(),
            ProveLLMs(  # type: ignore[arg-type]
                prover=budgeted_prover,
                query=budgeted_query,
            ),
            retriever=retriever,  # type: ignore[arg-type]
            verifier=verifier,  # type: ignore[arg-type]
            retriever_mode="standard",
            reflection_rounds=1,
            strict_integrity=True,
            call_budget=budget,
            failure_memory_enabled=True,
        )
    )

    assert result.success is True
    assert budget.snapshot() == {
        "max_model_calls": 3,
        "max_query_calls": 1,
        "max_compiler_calls": 2,
        "model_calls": 3,
        "query_calls": 1,
        "compiler_calls": 2,
    }
    assert retriever.queries == ["theorem proving True"]
    assert result.failure_memory["duplicate_queries_suppressed"] == 4
    trace = result.retrieval_trace[1]
    assert trace["dynamic_queries"] == queries
    assert trace["actual_search_queries"] == ["theorem proving True"]
    assert trace["duplicate_queries_suppressed"] == 4


def test_failure_memory_respects_40_model_32_compiler_budget() -> None:
    llm = _SequenceLLM([BAD] * 40)
    verifier = _SequenceVerifier([_failed()])
    budget = A2CallBudget(max_model_calls=40, max_query_calls=8, max_compiler_calls=32)
    budgeted = BudgetedLLMClient(llm, budget)

    result = asyncio.run(
        run_prove(
            _problem(),
            ProveLLMs(prover=budgeted, query=budgeted),  # type: ignore[arg-type]
            retriever=None,
            verifier=verifier,  # type: ignore[arg-type]
            retriever_mode="none",
            reflection_rounds=31,
            strict_integrity=True,
            call_budget=budget,
            failure_memory_enabled=True,
        )
    )

    assert result.success is False
    assert budget.model_calls == 40
    assert budget.compiler_calls == 1
    assert budget.query_calls == 0
    assert len(result.attempts) == 1
    assert len(verifier.proofs) == 1
    assert result.failure_memory["prevented_repeated_actions"] == 39
