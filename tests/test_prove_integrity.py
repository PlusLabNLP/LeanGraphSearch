from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from leansearchv2.prove import prompts
from leansearchv2.prove.integrity import (
    assemble_generated_proof_body,
    extract_generated_proof_body,
    find_forbidden_keywords,
    split_sorry_proof_hole,
    validate_proof_integrity,
)
from leansearchv2.prove.run import ProveLLMs, ProveProblem, run_prove
from leansearchv2.prove.verifier import LeanInteractVerifier, VerifyResult


PREFIX = "import Mathlib\n\ntheorem demo : True := by"


class _SequenceLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def chat(self, messages, **_kwargs):
        self.prompts.append(messages[0]["content"])
        return self.responses.pop(0)


class _AlwaysCompleteVerifier:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def verify(self, code: str, timeout_s: int = 600) -> VerifyResult:
        self.calls.append(code)
        return VerifyResult(complete=True, error_msg="", has_sorry=False)


def test_integrity_accepts_honest_body_and_ignores_comments_and_strings() -> None:
    code = PREFIX + '''
  -- The word axiom in a comment is harmless.
  have label : String := "sorry axiom opaque"
  trivial
'''

    result = validate_proof_integrity(code, PREFIX)

    assert result.valid is True
    assert result.forbidden_keywords == ()
    assert result.extra_commands == ()


def test_forbidden_keyword_scan_is_prefix_independent() -> None:
    code = '''
-- axiom and sorry are explanatory words here.
def label : String := "unsafe opaque admit"
axiom cheat : True
'''

    assert find_forbidden_keywords(code) == ("axiom",)


@pytest.mark.parametrize(
    "keyword", ["axiom", "constant", "opaque", "unsafe", "unsound", "sorry", "admit"]
)
def test_integrity_rejects_forbidden_keywords(keyword: str) -> None:
    result = validate_proof_integrity(PREFIX + f"\n  {keyword}", PREFIX)

    assert result.valid is False
    assert keyword in result.forbidden_keywords
    assert "Proof integrity violation" in result.error_msg


def test_integrity_rejects_prefix_rewrite_and_extra_declaration() -> None:
    rewritten = validate_proof_integrity(
        "axiom cheat : True\n" + PREFIX + "\n  exact cheat",
        PREFIX,
    )
    appended = validate_proof_integrity(
        PREFIX + "\n  trivial\n\ntheorem extra : True := by trivial",
        PREFIX,
    )

    assert rewritten.statement_prefix_valid is False
    assert rewritten.valid is False
    assert appended.valid is False
    assert appended.extra_commands == ("theorem",)


def test_mathlibmpr_sorry_hole_preserves_immutable_suffix() -> None:
    source = (
        "import Mathlib\n\nnamespace Demo\n\n"
        "theorem target : True := sorry\n\nend Demo"
    )
    contract = split_sorry_proof_hole(source)
    honest = contract.prefix + "\n  trivial" + contract.suffix

    assert contract.prefix.endswith("theorem target : True := by")
    assert contract.suffix == "\n\nend Demo"
    result = validate_proof_integrity(honest, contract.prefix, contract.suffix)
    assert result.valid is True
    assert result.statement_prefix_valid is True
    assert result.statement_suffix_valid is True


def test_strict_body_assembly_locks_source_and_accepts_optional_by_wrapper() -> None:
    source = "namespace Demo\ntheorem target : True := sorry\nend Demo"
    contract = split_sorry_proof_hole(source)

    assembled = assemble_generated_proof_body(
        "by\n  have helper : True := by trivial\n  exact helper",
        contract.prefix,
        contract.suffix,
    )

    assert assembled == (
        contract.prefix
        + "\n  have helper : True := by trivial\n  exact helper"
        + contract.suffix
    )
    assert extract_generated_proof_body(
        assembled, contract.prefix, contract.suffix
    ) == "have helper : True := by trivial\nexact helper"
    assert validate_proof_integrity(
        assembled, contract.prefix, contract.suffix
    ).valid is True


def test_strict_body_assembly_keeps_exact_legacy_full_source() -> None:
    honest = PREFIX + "\n  trivial"

    assert assemble_generated_proof_body(honest, PREFIX) == honest


def test_integrity_reports_modified_or_appended_suffix() -> None:
    source = "namespace Demo\ntheorem target : True := by sorry\nend Demo"
    contract = split_sorry_proof_hole(source)
    missing = validate_proof_integrity(
        contract.prefix + "\n  trivial", contract.prefix, contract.suffix
    )
    appended = validate_proof_integrity(
        contract.prefix + "\n  trivial" + contract.suffix + "\n#check True",
        contract.prefix,
        contract.suffix,
    )

    assert missing.valid is False
    assert missing.statement_suffix_valid is False
    assert "immutable source suffix" in missing.error_msg
    assert appended.valid is False
    assert appended.statement_suffix_valid is False


def test_sorry_hole_split_ignores_comments_and_requires_uniqueness() -> None:
    source = (
        "-- fake := sorry\n"
        "def label : String := \"also := sorry\"\n"
        "theorem target : True := sorry"
    )
    contract = split_sorry_proof_hole(source)
    assert contract.prefix.endswith("theorem target : True := by")
    with pytest.raises(ValueError, match="found 2"):
        split_sorry_proof_hole("theorem a : True := sorry\ntheorem b : True := sorry")


def test_strict_mode_reflects_after_axiom_instead_of_accepting_it() -> None:
    llm = _SequenceLLM([
        "axiom cheat : True\n" + PREFIX + "\n  exact cheat",
        "```lean4\ntrivial\n```",
    ])
    verifier = _AlwaysCompleteVerifier()

    result = asyncio.run(
        run_prove(
            ProveProblem(problem_id="demo", formal_statement=PREFIX),
            ProveLLMs(prover=llm, query=llm),  # type: ignore[arg-type]
            retriever=None,
            verifier=verifier,  # type: ignore[arg-type]
            retriever_mode="none",
            reflection_rounds=1,
            strict_integrity=True,
        )
    )

    assert result.success is True
    assert result.rounds_used == 1
    assert result.validation_mode == "integrity_strict"
    assert result.strict_success is True
    assert len(verifier.calls) == 1
    assert verifier.calls == [PREFIX + "\n  trivial"]
    assert "forbidden keyword(s): axiom" in result.attempts[0].error_msg
    assert all("Non-negotiable Proof-Integrity Rules" in text for text in llm.prompts)
    assert "Mandatory Repair of the Previous Attempt" in llm.prompts[1]


def test_failure_memory_only_keeps_paper_raw_prompt_and_compiler_semantics() -> None:
    escaped = "axiom cheat : True\n" + PREFIX + "\n  exact cheat"
    llm = _SequenceLLM([escaped])
    verifier = _AlwaysCompleteVerifier()

    result = asyncio.run(
        run_prove(
            ProveProblem(problem_id="paper-raw-memory", formal_statement=PREFIX),
            ProveLLMs(prover=llm, query=llm),  # type: ignore[arg-type]
            retriever=None,
            verifier=verifier,  # type: ignore[arg-type]
            retriever_mode="none",
            reflection_rounds=0,
            strict_integrity=False,
            failure_memory_enabled=True,
        )
    )

    # Failure Memory is the treatment, but it must not silently turn on the
    # corrected strict prompt/compiler protocol used by the superseded run.
    assert result.success is True
    assert result.strict_success is False
    assert result.validation_mode == "paper_raw"
    assert verifier.calls == [escaped]
    assert "Failure Memory" in llm.prompts[0]
    assert "Non-negotiable Proof-Integrity Rules" not in llm.prompts[0]
    assert "You are an expert at using Lean for formal proofs" in llm.prompts[0]


def test_strict_mode_reflects_on_suffix_violation_and_preserves_contract() -> None:
    source = "namespace Demo\ntheorem target : True := sorry\nend Demo"
    contract = split_sorry_proof_hole(source)
    invalid = contract.prefix + "\n  trivial\nend Wrong"
    honest = contract.prefix + "\n  trivial" + contract.suffix
    llm = _SequenceLLM([invalid, "```lean4\ntrivial\n```"])
    verifier = _AlwaysCompleteVerifier()

    result = asyncio.run(
        run_prove(
            ProveProblem(
                problem_id="suffix-demo",
                formal_statement=source,
                proof_prefix=contract.prefix,
                proof_suffix=contract.suffix,
            ),
            ProveLLMs(prover=llm, query=llm),  # type: ignore[arg-type]
            retriever=None,
            verifier=verifier,  # type: ignore[arg-type]
            retriever_mode="none",
            reflection_rounds=1,
            strict_integrity=True,
        )
    )

    assert result.success is True
    assert result.strict_success is True
    assert result.statement_suffix_valid is True
    assert verifier.calls == [honest]
    assert "immutable source suffix" in result.attempts[0].error_msg
    assert contract.prefix in llm.prompts[1]
    assert contract.suffix in llm.prompts[1]
    assert llm.prompts[1].count(contract.prefix) == 1
    assert "Previous Failed Tactic Body" in llm.prompts[1]


def test_paper_raw_mode_remains_backward_compatible() -> None:
    llm = _SequenceLLM(["axiom cheat : True\n" + PREFIX + "\n  exact cheat"])
    verifier = _AlwaysCompleteVerifier()

    result = asyncio.run(
        run_prove(
            ProveProblem(problem_id="demo", formal_statement=PREFIX),
            ProveLLMs(prover=llm, query=llm),  # type: ignore[arg-type]
            retriever=None,
            verifier=verifier,  # type: ignore[arg-type]
            retriever_mode="none",
            reflection_rounds=0,
        )
    )

    assert result.success is True
    assert result.validation_mode == "paper_raw"
    assert result.proof_integrity_valid is False
    assert result.strict_success is False
    assert len(verifier.calls) == 1
    assert "Non-negotiable Proof-Integrity Rules" not in llm.prompts[0]


def test_verifier_process_failure_becomes_a_reflection_attempt() -> None:
    class _CrashThenCompleteVerifier:
        def __init__(self) -> None:
            self.calls = 0

        async def verify(self, code: str, timeout_s: int = 600) -> VerifyResult:
            self.calls += 1
            if self.calls == 1:
                raise ConnectionAbortedError("Lean server closed unexpectedly")
            return VerifyResult(complete=True, error_msg="", has_sorry=False)

    llm = _SequenceLLM([PREFIX + "\n  trivial", PREFIX + "\n  trivial"])
    verifier = _CrashThenCompleteVerifier()

    result = asyncio.run(
        run_prove(
            ProveProblem(problem_id="demo", formal_statement=PREFIX),
            ProveLLMs(prover=llm, query=llm),  # type: ignore[arg-type]
            retriever=None,
            verifier=verifier,  # type: ignore[arg-type]
            retriever_mode="none",
            reflection_rounds=1,
            strict_integrity=True,
        )
    )

    assert result.success is True
    assert result.rounds_used == 1
    assert len(result.attempts) == 2
    assert "Lean verifier process failed with ConnectionAbortedError" in result.attempts[0].error_msg
    assert result.attempts[0].integrity_valid is True


def test_verifier_resets_server_after_timeout(monkeypatch) -> None:
    class _DeadServer:
        def __init__(self) -> None:
            self.killed = False

        def run(self, *_args, **_kwargs):
            raise TimeoutError("verification timed out")

        def kill(self):
            self.killed = True

    dead = _DeadServer()
    verifier = LeanInteractVerifier(project_dir="unused")
    verifier._server = dead
    monkeypatch.setitem(
        sys.modules,
        "lean_interact",
        SimpleNamespace(Command=lambda cmd: SimpleNamespace(cmd=cmd)),
    )

    with pytest.raises(TimeoutError):
        asyncio.run(verifier.verify(PREFIX + "\n  trivial", timeout_s=1))

    assert dead.killed is True
    assert verifier._server is None


def test_strict_prompts_name_every_forbidden_escape_hatch() -> None:
    init_prompt = prompts.PROVER_INIT_STRICT.format(
        proof_prefix=PREFIX,
        proof_suffix="",
        search_results="(none)",
        integrity_rules=prompts.PROOF_INTEGRITY_RULES,
        failure_memory_guidance="",
    )
    reflect_prompt = prompts.PROVER_REFLECT_STRICT.format(
        proof_prefix=PREFIX,
        proof_suffix="",
        proof=PREFIX + "\n  trivial",
        error_msg="example error",
        search_results="(none)",
        integrity_rules=prompts.PROOF_INTEGRITY_RULES,
        failure_memory_guidance="",
    )
    for keyword in (
        "axiom", "constant", "opaque", "unsafe", "unsound", "sorry", "admit"
    ):
        assert f"`{keyword}`" in init_prompt
        assert f"`{keyword}`" in reflect_prompt
    assert "Compiler-Tool Feedback" in reflect_prompt
    assert "contains only the honest tactic commands" in init_prompt
    assert "Do not include the theorem statement" in init_prompt
    assert "Previous Failed Tactic Body" in reflect_prompt
