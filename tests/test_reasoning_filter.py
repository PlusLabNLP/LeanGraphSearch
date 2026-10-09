from __future__ import annotations

import asyncio
import json

import pytest

from leansearchv2.reasoning.filter import filter_results
from leansearchv2.standard_client import ResultData, SearchResult


def _doc(index: int) -> SearchResult:
    return SearchResult(
        result=ResultData(
            module_name=["Mathlib", "Test"],
            kind="theorem",
            name=[f"T{index}"],
            signature="",
            type="True",
        ),
        distance=float(index),
    )


class _CaptureFilterLLM:
    def __init__(self) -> None:
        self.prompt = ""

    async def chat(self, messages, **_kwargs):
        self.prompt = messages[0]["content"]
        return json.dumps({"kept_results": [0, 19, 20, 29, 30]})


@pytest.mark.parametrize(
    ("max_docs", "expected_names", "last_visible", "first_hidden"),
    [
        (20, ["T0", "T19"], "[19]", "[20]"),
        (30, ["T0", "T19", "T20", "T29"], "[29]", "[30]"),
    ],
)
def test_filter_document_cap_is_explicit_and_preserves_default(
    max_docs: int,
    expected_names: list[str],
    last_visible: str,
    first_hidden: str,
) -> None:
    llm = _CaptureFilterLLM()
    filtered, _ = asyncio.run(
        filter_results(
            llm,  # type: ignore[arg-type]
            formal_statement="theorem p : True",
            query_to_docs={"q": [_doc(i) for i in range(35)]},
            query_to_step_desc={"q": "step"},
            max_docs_to_llm=max_docs,
        )
    )

    assert [doc.result.name[-1] for doc in filtered["q"]] == expected_names
    assert last_visible in llm.prompt
    assert first_hidden not in llm.prompt


def test_filter_document_cap_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(
            filter_results(
                _CaptureFilterLLM(),  # type: ignore[arg-type]
                formal_statement="theorem p : True",
                query_to_docs={"q": [_doc(0)]},
                query_to_step_desc={"q": "step"},
                max_docs_to_llm=0,
            )
        )
