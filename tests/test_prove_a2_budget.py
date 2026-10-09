import json
import unittest
from unittest.mock import AsyncMock, patch

from leansearchv2.prove import A2CallBudget, BudgetedLLMClient, ProveLLMs, ProveProblem, run_prove
from leansearchv2.prove.verifier import VerifyResult
from leansearchv2.reasoning import ReasoningLLMs


class _FakeLLM:
    profile = "fake"
    provider = "fake"
    model = "fake"

    def __init__(self):
        self.requests = 0

    async def chat(self, messages, **kwargs):
        self.requests += 1
        prompt = messages[-1]["content"]
        if "queries" in prompt.lower():
            return json.dumps({
                "queries": [
                    "group theory lemma one",
                    "group theory lemma two",
                    "group theory lemma three",
                    "group theory lemma four",
                    "group theory lemma five",
                ]
            })
        return "```lean4\nimport Mathlib\nexample : True := by\n  exact True.intro\n```"

    def usage_snapshot(self):
        return {"requests": self.requests}


class _AlwaysFailVerifier:
    async def verify(self, code, timeout_s=600):
        return VerifyResult(complete=False, error_msg="failed", has_sorry=False)


class _MalformedQueryLLM(_FakeLLM):
    async def chat(self, messages, **kwargs):
        self.requests += 1
        prompt = messages[-1]["content"]
        if "queries" in prompt.lower():
            return "not valid JSON"
        return "```lean4\nimport Mathlib\nexample : True := by\n  exact True.intro\n```"


class _FakeRetriever:
    def __init__(self):
        self.requests = 0

    async def search(self, *args, **kwargs):
        self.requests += 1
        return []


class _FakeReasoningResult:
    entries = []

    def to_dict(self):
        return {
            "sketch_status": "good",
            "informal_steps": [
                {"description": "Prove the goal directly.", "reasoning": "Use the available hypotheses."}
            ],
            "entries": [],
        }


class ProveA2BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_retrieval_stops_at_32_compiles(self):
        budget = A2CallBudget()
        base = _FakeLLM()
        proxy = BudgetedLLMClient(base, budget)
        result = await run_prove(
            ProveProblem("p", "import Mathlib\nexample : True := by\n  sorry"),
            ProveLLMs(prover=proxy, query=proxy),
            None,
            _AlwaysFailVerifier(),
            retriever_mode="none",
            reflection_rounds=39,
            call_budget=budget,
        )
        self.assertFalse(result.success)
        self.assertEqual(len(result.attempts), 32)
        self.assertEqual(result.budget["compiler_calls"], 32)
        self.assertEqual(result.budget["model_calls"], 32)
        self.assertEqual(result.budget["query_calls"], 0)

    async def test_standard_uses_8_query_actions_and_40_model_calls(self):
        budget = A2CallBudget()
        base = _FakeLLM()
        proxy = BudgetedLLMClient(base, budget)
        result = await run_prove(
            ProveProblem("p", "import Mathlib\nexample : True := by\n  sorry"),
            ProveLLMs(prover=proxy, query=proxy),
            _FakeRetriever(),
            _AlwaysFailVerifier(),
            retriever_mode="standard",
            reflection_rounds=39,
            call_budget=budget,
        )
        self.assertFalse(result.success)
        self.assertEqual(len(result.attempts), 32)
        self.assertEqual(result.budget["compiler_calls"], 32)
        self.assertEqual(result.budget["query_calls"], 8)
        self.assertEqual(result.budget["model_calls"], 40)

    async def test_standard_can_query_after_all_31_failed_reflections(self):
        budget = A2CallBudget(
            max_model_calls=63,
            max_query_calls=31,
            max_compiler_calls=32,
        )
        base = _FakeLLM()
        proxy = BudgetedLLMClient(base, budget)
        retriever = _FakeRetriever()
        result = await run_prove(
            ProveProblem("p", "import Mathlib\nexample : True := by\n  sorry"),
            ProveLLMs(prover=proxy, query=proxy),
            retriever,
            _AlwaysFailVerifier(),
            retriever_mode="standard",
            reflection_rounds=31,
            call_budget=budget,
        )
        self.assertFalse(result.success)
        self.assertEqual(len(result.attempts), 32)
        self.assertEqual(result.budget["compiler_calls"], 32)
        self.assertEqual(result.budget["query_calls"], 31)
        self.assertEqual(result.budget["model_calls"], 63)
        self.assertEqual(retriever.requests, 31 * 5)
        self.assertEqual(
            [event["round"] for event in result.retrieval_trace[1:]],
            list(range(1, 32)),
        )

    async def test_adaptive_reasoning_keeps_separate_prefetch_and_standard_query_budget(self):
        budget = A2CallBudget()
        base = _FakeLLM()
        proxy = BudgetedLLMClient(base, budget)
        retriever = _FakeRetriever()
        reasoning = ReasoningLLMs(sketch=base, filter=base, judge=base)
        with patch(
            "leansearchv2.prove.run.run_reasoning_budgeted",
            new=AsyncMock(return_value=_FakeReasoningResult()),
        ):
            result = await run_prove(
                ProveProblem("p", "import Mathlib\nexample : True := by\n  sorry"),
                ProveLLMs(
                    prover=proxy,
                    query=proxy,
                    reasoning_branches=[reasoning],
                ),
                retriever,
                _AlwaysFailVerifier(),
                retriever_mode="reasoning_adaptive",
                reflection_rounds=39,
                call_budget=budget,
            )
        self.assertFalse(result.success)
        self.assertEqual(len(result.attempts), 32)
        self.assertEqual(result.budget["compiler_calls"], 32)
        self.assertEqual(result.budget["query_calls"], 8)
        self.assertEqual(result.budget["model_calls"], 40)
        # Eight A2 query actions, each requesting five independent searches.
        self.assertEqual(retriever.requests, 40)
        self.assertEqual(result.retrieval_trace[0]["source"], "reasoning_prefetch")
        self.assertEqual(result.retrieval_trace[1]["source"], "adaptive_standard_search")
        self.assertEqual(
            result.retrieval_trace[-1]["source"],
            "fixed_reasoning_after_query_budget",
        )

    async def test_strict_mode_rejects_malformed_query_generation(self):
        budget = A2CallBudget()
        base = _MalformedQueryLLM()
        proxy = BudgetedLLMClient(base, budget)
        with self.assertRaisesRegex(RuntimeError, "query generation failed"):
            await run_prove(
                ProveProblem("p", "import Mathlib\nexample : True := by\n  sorry"),
                ProveLLMs(prover=proxy, query=proxy),
                _FakeRetriever(),
                _AlwaysFailVerifier(),
                retriever_mode="standard",
                reflection_rounds=39,
                strict_retrieval_errors=True,
                call_budget=budget,
            )
