"""Regression tests for lossless transport of multiline tactic bodies."""
import asyncio
import os
from pathlib import Path

import pytest

from leansearchv2.prove.integrity import (
    assemble_generated_proof_body,
    extract_generated_proof_body,
    validate_proof_integrity,
)
from leansearchv2.prove.run import _prove_call, _get_queries
from leansearchv2.prove import prompts
from leansearchv2.prove.verifier import LeanInteractVerifier, extract_lean_code

PREFIX = "import Mathlib\nnamespace Demo\nexample : True ∧ True := by"
SUFFIX = "\nend Demo"
BODIES = [
    "constructor\n· trivial\n· trivial",
    "have h : True := by\n  trivial\nexact ⟨h, h⟩",
    "constructor <;>\n  trivial",
]


def test_strict_prompt_routing_keeps_paper_prompts_separate():
    class FakeLLM:
        async def chat(self, messages, **kwargs):
            self.prompt = messages[0]["content"]
            return '{"queries": ["fact A", "fact B"]}'

    async def check():
        for strict in (False, True):
            llm = FakeLLM()
            queries = await _get_queries(llm, formal_statement=PREFIX,
                informal_statement="target", proof=PREFIX + "\n  bad",
                error_msg="unknown local name", num_queries=2,
                strict_errors=True, strict_integrity=strict,
                proof_prefix=PREFIX)
            template = prompts.GET_QUERY_REFLECT_STRICT if strict else prompts.GET_QUERY_REFLECT
            assert llm.prompt == template.format(lean_code=PREFIX,
                informal_statement="target", proof="bad" if strict else PREFIX + "\n  bad",
                error_msg="unknown local name", num_queries=2)
            assert queries == ["fact A", "fact B"]
            await _prove_call(llm, lean_code=PREFIX, search_results_str="",
                strict_integrity=strict, proof_prefix=PREFIX,
                failure_memory_enabled=True)
            guidance = prompts.FAILURE_MEMORY_GUIDANCE_STRICT if strict else prompts.FAILURE_MEMORY_GUIDANCE
            assert guidance in llm.prompt
    asyncio.run(check())


def test_integrity_feedback_requests_body_only():
    result = validate_proof_integrity("axiom escape : True\n" + PREFIX, PREFIX)

    assert result.valid is False
    assert "return only an honest replacement tactic body" in result.error_msg
    assert "return the exact immutable prefix" not in result.error_msg.lower()


@pytest.mark.parametrize("body", BODIES)
@pytest.mark.parametrize("indent", [0, 2, 4])
@pytest.mark.parametrize("wrapper", [False, True])
def test_body_transport_and_reflection_roundtrip(body, indent, wrapper):
    generated = ("by\n" + "\n".join("  " + s for s in body.splitlines())) if wrapper else body
    generated = "\n".join(" " * indent + s for s in generated.splitlines())
    response = "Plan\n```lean4\n\n" + generated + "\n```"
    code = extract_lean_code(response, preserve_indentation=True)
    assembled = assemble_generated_proof_body(code, PREFIX, SUFFIX)
    expected = PREFIX + "\n" + "\n".join("  " + s for s in body.splitlines()) + SUFFIX
    assert assembled == expected
    reflected = extract_generated_proof_body(assembled, PREFIX, SUFFIX)
    assert reflected == body
    assert assemble_generated_proof_body(reflected, PREFIX, SUFFIX) == expected


def test_same_line_suffix_cannot_be_swallowed_by_comment():
    code = assemble_generated_proof_body("trivial -- model comment", "example : True := by", " -- source comment\n")
    assert "-- model comment\n -- source comment" in code


@pytest.mark.skipif(not os.environ.get("LEAN_BODY_TEST_PROJECT"), reason="Set LEAN_BODY_TEST_PROJECT for real Lean compilation")
def test_real_lean_prover_and_reflection_transport():
    class FakeLLM:
        def __init__(self, response):
            self.response = response
            self.seen = []

        async def chat(self, messages, **kwargs):
            self.seen.append(messages[0]["content"])
            return self.response

    async def check():
        verifier = LeanInteractVerifier(project_dir=str(Path(os.environ["LEAN_BODY_TEST_PROJECT"]).resolve()), auto_build_project=False)
        try:
            for body in BODIES:
                for wrapper in (False, True):
                    content = "\n".join("  " + line for line in body.splitlines())
                    if wrapper:
                        content = "by\n" + content
                    llm = FakeLLM("```lean4\n" + content + "\n```")
                    code = await _prove_call(llm, lean_code=PREFIX, search_results_str="", strict_integrity=True, proof_prefix=PREFIX, proof_suffix=SUFFIX)
                    result = await verifier.verify(code, timeout_s=45)
                    assert result.success, result.error_msg
                    code2 = await _prove_call(llm, lean_code=PREFIX, search_results_str="", failing_proof=code, error_msg="diagnostic replay", strict_integrity=True, proof_prefix=PREFIX, proof_suffix=SUFFIX)
                    assert code2 == code
                    assert "```lean4\n" + body + "\n```" in llm.seen[-1]
            bad = await verifier.verify("import Mathlib\nexample : True := by\n  exact unknownNameForRegression", timeout_s=45)
            assert not bad.success
            assert "[line 3, column " in bad.error_msg
        finally:
            await verifier.close()
    asyncio.run(check())
