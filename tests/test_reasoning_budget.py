from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from leansearchv2.reasoning.decompose import ProofPlan
from leansearchv2.reasoning.run import (
    Problem,
    ReasoningResult,
    _batch_search,
    run_reasoning_budgeted,
)
from leansearchv2.standard_client import ResultData, SearchResult


def _doc(name: str) -> SearchResult:
    return SearchResult(
        result=ResultData(
            module_name=["Mathlib", "Test"],
            kind="theorem",
            name=[name],
            signature="",
            type="True",
        ),
        distance=0.0,
    )


def _result(
    status: str,
    filtered: dict[str, list[SearchResult]] | None = None,
) -> ReasoningResult:
    return ReasoningResult(
        problem_id="p",
        status=status,
        big_loop_count=0,
        plan=ProofPlan(),
        filtered=filtered or {},
        filter_reasoning=f"filter-{status}",
        quality_reasoning=f"judge-{status}",
        entries=[],
    )


class ReasoningBranchBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_search_kwargs_are_forwarded_and_internal_pool_is_sliced(self) -> None:
        class Retriever:
            calls: list[dict] = []

            async def search(self, _query, **kwargs):
                self.calls.append(kwargs)
                return [_doc("A"), _doc("B"), _doc("C")]

        retriever = Retriever()
        result = await _batch_search(
            retriever,  # type: ignore[arg-type]
            ["query"],
            top_k=2,
            rerank=True,
            concurrency=1,
            search_kwargs={
                "graph_augment": True,
                "graph_final_top_k": 100,
                "rank_fusion": "qwen_ppr_zscore",
            },
        )

        self.assertEqual([doc.result.name[-1] for doc in result["query"]], ["A", "B"])
        self.assertEqual(retriever.calls[0]["top_k"], 2)
        self.assertEqual(retriever.calls[0]["graph_final_top_k"], 100)
        self.assertEqual(retriever.calls[0]["rank_fusion"], "qwen_ppr_zscore")

    async def test_first_good_cancels_unfinished_sibling(self) -> None:
        sibling_cancelled = asyncio.Event()

        async def fake_run(_problem, _retriever, llm, **_kwargs):
            if llm == "winner":
                return _result("good")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        with patch("leansearchv2.reasoning.run.run_reasoning", side_effect=fake_run):
            result = await run_reasoning_budgeted(
                Problem("p", "theorem p : True"),
                object(),  # type: ignore[arg-type]
                ["winner", "sibling"],  # type: ignore[list-item]
            )

        self.assertEqual(result.winning_branch, 0)
        self.assertEqual(result.branch_statuses, ["good", "early_stop"])
        self.assertTrue(sibling_cancelled.is_set())
        self.assertEqual(result.to_dict()["branch_budget"], 2)

    async def test_two_failed_branches_pool_rank_votes(self) -> None:
        common = _doc("Common")
        left = _doc("Left")
        right = _doc("Right")
        branch_outputs = {
            "left": _result("fail", {"q": [common, left]}),
            "right": _result("fail", {"q": [common, right]}),
        }

        async def fake_run(_problem, _retriever, llm, **_kwargs):
            return branch_outputs[llm]

        with patch("leansearchv2.reasoning.run.run_reasoning", side_effect=fake_run):
            result = await run_reasoning_budgeted(
                Problem("p", "theorem p : True"),
                object(),  # type: ignore[arg-type]
                ["left", "right"],  # type: ignore[list-item]
                output_top_k=3,
            )

        self.assertIsNone(result.winning_branch)
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.branch_statuses, ["fail", "fail"])
        self.assertEqual(len(result.result.filtered), 2)
        self.assertEqual(
            [doc_id for doc_id, _score, _doc in result.entries],
            [
                "Mathlib.Test::Common",
                "Mathlib.Test::Left",
                "Mathlib.Test::Right",
            ],
        )

    async def test_completed_sibling_keeps_real_status(self) -> None:
        async def fake_run(_problem, _retriever, _llm, **_kwargs):
            return _result("good")

        with patch("leansearchv2.reasoning.run.run_reasoning", side_effect=fake_run):
            result = await run_reasoning_budgeted(
                Problem("p", "theorem p : True"),
                object(),  # type: ignore[arg-type]
                ["first", "second"],  # type: ignore[list-item]
            )

        self.assertEqual(result.winning_branch, 0)
        self.assertEqual(result.branch_statuses, ["good", "good"])

    async def test_all_branch_errors_are_not_reported_as_success(self) -> None:
        async def fake_run(_problem, _retriever, llm, **_kwargs):
            raise ValueError(f"broken-{llm}")

        with patch("leansearchv2.reasoning.run.run_reasoning", side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError, "all reasoning branches errored"):
                await run_reasoning_budgeted(
                    Problem("p", "theorem p : True"),
                    object(),  # type: ignore[arg-type]
                    ["left", "right"],  # type: ignore[list-item]
                )


if __name__ == "__main__":
    unittest.main()
