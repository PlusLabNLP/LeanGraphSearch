"""Pre-sketch quality judge.

Returns one of {"good", "bad"} (plus reasoning + improvement suggestions).
"good" terminates the big loop; "bad" feeds reasoning back into
`decompose(..., previous_plan=..., previous_quality_reasoning=...)`.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..standard_client import SearchResult
from . import prompts
from .decompose import _parse_json
from .filter import _pretty_inline


_MAX_THEOREMS_TO_LLM = 30


async def judge(
    llm: LLMClient,
    *,
    formal_statement: str,
    plan_steps: list[dict],
    filtered: dict[str, list[SearchResult]],
    filter_reasoning: str = "",
) -> tuple[str, str]:
    """Return (verdict, reasoning_with_suggestions). verdict ∈ {good, bad}.

    Parse failures are conservatively mapped to "bad".
    """
    plan_str_lines: list[str] = []
    for i, s in enumerate(plan_steps):
        plan_str_lines.append(f"- Step {i}: {s.get('description', '')}")
        plan_str_lines.append(f"  Reasoning: {s.get('reasoning', '')}")
    plan_str = "\n".join(plan_str_lines)

    all_docs: list[SearchResult] = []
    seen: set[str] = set()
    for q, docs in filtered.items():
        for d in docs:
            key = f"{'.'.join(d.result.module_name)}::{'.'.join(d.result.name)}"
            if key in seen:
                continue
            seen.add(key)
            all_docs.append(d)
            if len(all_docs) >= _MAX_THEOREMS_TO_LLM:
                break
        if len(all_docs) >= _MAX_THEOREMS_TO_LLM:
            break
    theorems_str = "\n".join(f"- {_pretty_inline(d)}" for d in all_docs) or "(No theorems retrieved)"
    filter_reason = filter_reasoning or "(No filter reasoning)"

    prompt = prompts.JUDGE.format(
        formal_statement=formal_statement,
        informal_plan=plan_str,
        available_theorems=theorems_str,
        filter_reasoning=filter_reason,
    )
    try:
        response = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        data = _parse_json(response)
        verdict = str(data.get("judgment") or "bad").strip().lower()
        if verdict not in ("good", "bad"):
            verdict = "bad"
        reasoning = str(data.get("reasoning") or "")
        suggestions = str(data.get("suggestions_if_bad") or "")
        if suggestions:
            reasoning = f"{reasoning}\n\nSuggestions: {suggestions}".strip()
        return verdict, reasoning
    except Exception as e:
        return "bad", f"judge parse failure ({type(e).__name__}: {e}); defaulting to bad"
