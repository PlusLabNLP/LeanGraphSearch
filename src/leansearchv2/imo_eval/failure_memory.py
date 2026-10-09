"""Problem-local deterministic failure memory for the proving trajectory."""


from __future__ import annotations

import hashlib

import json

import re

from dataclasses import dataclass, field

from typing import Any, Callable

from .source_utils import normalize_header, replace_terminal_sorry

from .problem import BenchmarkProblem

from .diagnostics import classify_proof_edit

from .failure_state import formal_score, formal_snapshot

_LINE_COLUMN_RE = re.compile(r"line\s+(\d+),\s*column\s+(\d+)", re.IGNORECASE)

_GOAL_RE = re.compile(r"(?m)^\s*⊢\s*(.+)$")

_CASE_RE = re.compile(r"(?m)^\s*case\s+(.+)$")

_QUOTED_IDENTIFIER_RE = re.compile(
    r"[`']([A-Za-z_][A-Za-z0-9_'.]*(?:\.[A-Za-z_][A-Za-z0-9_']*)*)[`']"
)

_TACTIC_HEAD_RE = re.compile(
    r"^(?:[·+*-]\s*)?(?:case\s+\S+\s*=>\s*)?"
    r"([A-Za-z_][A-Za-z0-9_'.]*)"
)

_LOCAL_CONTEXT_RE = re.compile(
    r"^\s*(?:[A-Za-z_][A-Za-z0-9_✝₀-₉']*(?:\s+[A-Za-z_][A-Za-z0-9_✝₀-₉']*)*\s*:)"
)

def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def _proof_relative_location(
    problem: BenchmarkProblem,
    proof: str,
    absolute_line: int,
) -> int | None:
    if absolute_line < 1:
        return None
    try:
        statement = replace_terminal_sorry(problem.formal_statement, proof)
        header = normalize_header(problem.header)
    except ValueError:
        return None
    source = (
        f"{header.rstrip()}\n\n{statement.lstrip(chr(10) + chr(13))}"
        if header
        else statement.lstrip("\r\n")
    )
    marker = ":= " + proof
    marker_index = source.rfind(marker)
    if marker_index < 0:
        return None
    proof_index = marker_index + len(":= ")
    proof_start_line = source[:proof_index].count("\n") + 1
    relative = absolute_line - proof_start_line + 1
    return relative if relative >= 1 else None

def _code_context(
    problem: BenchmarkProblem,
    proof: str,
    error: str,
    *,
    radius: int = 2,
) -> tuple[list[str], str | None]:
    lines = proof.splitlines()
    match = _LINE_COLUMN_RE.search(error)
    relative = (
        _proof_relative_location(problem, proof, int(match.group(1)))
        if match
        else None
    )
    if relative is None or relative > len(lines):
        nonempty = [index for index, line in enumerate(lines) if line.strip()]
        center = nonempty[-1] if nonempty else 0
    else:
        center = relative - 1
    start = max(0, center - radius)
    end = min(len(lines), center + radius + 1)
    snippet = [line.rstrip() for line in lines[start:end]]
    failing_line = lines[center].strip() if 0 <= center < len(lines) else ""
    tactic_match = _TACTIC_HEAD_RE.match(failing_line)
    return snippet, tactic_match.group(1) if tactic_match else None

def _source_code_context(
    proof: str,
    error: str,
    *,
    radius: int = 2,
) -> tuple[list[str], str | None]:
    """Locate a failure directly in a complete Lean source candidate."""

    lines = proof.splitlines()
    match = _LINE_COLUMN_RE.search(error)
    absolute = int(match.group(1)) if match else None
    if absolute is None or absolute < 1 or absolute > len(lines):
        nonempty = [index for index, line in enumerate(lines) if line.strip()]
        center = nonempty[-1] if nonempty else 0
    else:
        center = absolute - 1
    start = max(0, center - radius)
    end = min(len(lines), center + radius + 1)
    snippet = [line.rstrip() for line in lines[start:end]]
    failing_line = lines[center].strip() if 0 <= center < len(lines) else ""
    tactic_match = _TACTIC_HEAD_RE.match(failing_line)
    return snippet, tactic_match.group(1) if tactic_match else None

def _local_context(error: str, *, limit: int = 8) -> list[str]:
    lines = error.splitlines()
    goal_positions = [index for index, line in enumerate(lines) if "⊢" in line]
    end = goal_positions[-1] if goal_positions else len(lines)
    context = [
        line.strip()
        for line in lines[:end]
        if _LOCAL_CONTEXT_RE.match(line) or _CASE_RE.match(line)
    ]
    return context[-limit:]

def formal_failure_state(
    problem: BenchmarkProblem,
    proof: str,
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the auditable formal-state material required by A2."""

    error = str(result_payload.get("error") or "")
    snippet, failed_tactic = _code_context(problem, proof, error)
    return _formal_failure_state(proof, result_payload, snippet, failed_tactic)

def formal_failure_state_from_source(
    _problem: Any,
    proof: str,
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the same A2 state from a complete source-file candidate.

    The prove/Table-3 pipeline submits complete files rather than MiniF2F proof
    bodies.  Lean diagnostics are therefore already relative to ``proof`` and
    need no terminal-``sorry`` reconstruction.
    """

    error = str(result_payload.get("error") or "")
    snippet, failed_tactic = _source_code_context(proof, error)
    return _formal_failure_state(proof, result_payload, snippet, failed_tactic)

def _formal_failure_state(
    proof: str,
    result_payload: dict[str, Any],
    snippet: list[str],
    failed_tactic: str | None,
) -> dict[str, Any]:
    snapshot = formal_snapshot(proof, result_payload)
    error = str(result_payload.get("error") or "")
    quoted = sorted(set(_QUOTED_IDENTIFIER_RE.findall(error)))[:8]
    goals = [match.group(1).strip() for match in _GOAL_RE.finditer(error)]
    material = {
        "primary_error_category": snapshot["primary_error_category"],
        "normalized_error": snapshot["normalized_error"],
        "unresolved_goals": goals or snapshot["unresolved_goals"],
        "key_local_context": _local_context(error),
        "failed_tactic": failed_tactic,
        "failed_declarations": quoted,
        "related_code_snippet": snippet,
    }
    rendered = json.dumps(material, ensure_ascii=False, sort_keys=True)
    return {
        **material,
        "state_fingerprint": _sha256(rendered),
        "formal_snapshot": snapshot,
    }

def failed_modification(
    previous_proof: str | None,
    proof: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    edit = classify_proof_edit(previous_proof, proof)
    action_material = {
        "edit_kind": edit["kind"],
        "failed_tactic": state.get("failed_tactic"),
        "failed_declarations": state.get("failed_declarations", []),
        "related_code_snippet": state.get("related_code_snippet", []),
    }
    return {
        **action_material,
        "line_similarity": edit.get("line_similarity"),
        "proof_sha256": _sha256(proof),
        "action_fingerprint": _sha256(
            json.dumps(action_material, ensure_ascii=False, sort_keys=True)
        ),
    }

@dataclass
class ProblemFailureMemory:
    problem: BenchmarkProblem
    repeat_threshold: int = 2
    summary_max_states: int = 3
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed_proofs: dict[str, dict[str, Any]] = field(default_factory=dict)
    previous_proof: str | None = None
    current_proof: str | None = None
    current_snapshot: dict[str, Any] | None = None
    best_proof: str | None = None
    best_snapshot: dict[str, Any] | None = None
    latest_compiler_feedback: dict[str, Any] | None = None
    repeated_state_observations: int = 0
    dead_end_prompts: int = 0
    prevented_repeated_actions: int = 0
    duplicate_queries_suppressed: int = 0
    last_dead_end_action_fingerprint: str | None = None
    last_dead_end_proof_sha256: str | None = None
    state_builder: Callable[[Any, str, dict[str, Any]], dict[str, Any]] = field(
        default=formal_failure_state,
        repr=False,
        compare=False,
    )

    def known_failed_proof(self, proof: str) -> dict[str, Any] | None:
        return self.failed_proofs.get(_sha256(proof))

    def record_failure(
        self,
        proof: str,
        result_payload: dict[str, Any],
    ) -> dict[str, Any]:
        state = self.state_builder(self.problem, proof, result_payload)
        modification = failed_modification(self.previous_proof, proof, state)
        fingerprint = str(state["state_fingerprint"])
        entry = self.entries.setdefault(
            fingerprint,
            {
                "state": state,
                "occurrences": 0,
                "failed_modifications": [],
            },
        )
        entry["occurrences"] += 1
        entry["failed_modifications"].append(modification)
        occurrence = int(entry["occurrences"])
        repeated = occurrence >= self.repeat_threshold
        if occurrence > 1:
            self.repeated_state_observations += 1
        if repeated:
            self.dead_end_prompts += 1
            self.last_dead_end_action_fingerprint = modification["action_fingerprint"]
            self.last_dead_end_proof_sha256 = modification["proof_sha256"]
        stable = str(result_payload.get("failure_kind") or "") in {
            "proof",
            "rejected",
        }
        if stable:
            self.failed_proofs[modification["proof_sha256"]] = {
                "result_payload": dict(result_payload),
                "state_fingerprint": fingerprint,
                "modification": modification,
            }
        self.previous_proof = proof
        self.current_proof = proof
        self.current_snapshot = state["formal_snapshot"]
        self.latest_compiler_feedback = {
            "error": str(result_payload.get("error") or ""),
            "failure_kind": result_payload.get("failure_kind"),
            "state_fingerprint": fingerprint,
        }
        if self.best_snapshot is None or formal_score(self.current_snapshot) > formal_score(
            self.best_snapshot
        ):
            self.best_proof = proof
            self.best_snapshot = self.current_snapshot
        return {
            "state": state,
            "modification": modification,
            "occurrence": occurrence,
            "repeated": repeated,
            "dead_end_summary": self.dead_end_summary(latest_fingerprint=fingerprint),
        }

    def dead_end_summary(self, *, latest_fingerprint: str) -> dict[str, Any]:
        repeated = [
            (fingerprint, entry)
            for fingerprint, entry in self.entries.items()
            if int(entry["occurrences"]) >= self.repeat_threshold
        ]
        repeated.sort(key=lambda item: (-int(item[1]["occurrences"]), item[0]))
        visible = []
        for fingerprint, entry in repeated[: self.summary_max_states]:
            state = entry["state"]
            actions = entry["failed_modifications"]
            visible.append(
                {
                    "state_fingerprint": fingerprint,
                    "occurrences": entry["occurrences"],
                    "error_category": state["primary_error_category"],
                    "error": str(state["normalized_error"])[:280],
                    "goals": list(state["unresolved_goals"])[:2],
                    "failed_actions": [
                        {
                            "edit_kind": action["edit_kind"],
                            "tactic": action["failed_tactic"],
                            "declarations": action["failed_declarations"],
                        }
                        for action in actions[-3:]
                    ],
                }
            )
        latest = self.entries[latest_fingerprint]
        summary: dict[str, Any] = {
            "latest_state_fingerprint": latest_fingerprint,
            "latest_occurrence": latest["occurrences"],
            "repeated_dead_ends": visible,
            "best_candidate_sha256": _sha256(self.best_proof)
            if self.best_proof
            else None,
        }
        if int(latest["occurrences"]) >= self.repeat_threshold:
            latest_action = latest["failed_modifications"][-1]
            summary["directive"] = (
                "DEAD_END_REPEATED: this formal state has now appeared at least "
                "twice. Do not repeat the same tactic, lemma, declaration, or proof "
                "modification. You remain free to change the mathematical strategy, "
                "rewrite the whole proof, make a different local repair, or issue a "
                "new Mathlib query; the controller will not choose for you."
            )
            summary["do_not_repeat"] = {
                "tactic": latest_action["failed_tactic"],
                "declarations": latest_action["failed_declarations"],
                "action_fingerprint": latest_action["action_fingerprint"],
            }
        return summary

    def block_exact_failure(self, proof: str) -> dict[str, Any]:
        known = self.known_failed_proof(proof)
        if known is None:
            raise KeyError("proof is not a known stable failure")
        self.prevented_repeated_actions += 1
        return {
            "state_fingerprint": known["state_fingerprint"],
            "proof_sha256": _sha256(proof),
            "modification": known["modification"],
            "directive": (
                "This exact proof already has a deterministic failed result and was "
                "not recompiled. Submit a different action."
            ),
        }

    def note_success(self, proof: str) -> dict[str, Any]:
        proof_hash = _sha256(proof)
        action_state = {
            "primary_error_category": "success",
            "failed_tactic": None,
            "failed_declarations": [],
            "related_code_snippet": proof.splitlines()[-5:],
        }
        modification = failed_modification(self.previous_proof, proof, action_state)
        changed_action = bool(
            self.last_dead_end_action_fingerprint
            and modification["action_fingerprint"]
            != self.last_dead_end_action_fingerprint
            and proof_hash != self.last_dead_end_proof_sha256
        )
        return {
            "after_dead_end_prompt": self.dead_end_prompts > 0,
            "changed_action_after_dead_end": changed_action,
            "successful_proof_sha256": proof_hash,
            "successful_modification": modification,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "type": "free_failure_memory_summary",
            "single_trajectory": True,
            "unique_failure_states": len(self.entries),
            "failure_state_observations": sum(
                int(entry["occurrences"]) for entry in self.entries.values()
            ),
            "repeated_state_observations": self.repeated_state_observations,
            "dead_end_prompts": self.dead_end_prompts,
            "prevented_repeated_actions": self.prevented_repeated_actions,
            "duplicate_queries_suppressed": self.duplicate_queries_suppressed,
            "best_proof": self.best_proof,
            "best_snapshot": self.best_snapshot,
            "latest_compiler_feedback": self.latest_compiler_feedback,
            "failure_states": [
                {
                    "state_fingerprint": fingerprint,
                    "occurrences": entry["occurrences"],
                    "state": entry["state"],
                    "failed_modifications": entry["failed_modifications"],
                }
                for fingerprint, entry in sorted(self.entries.items())
            ],
        }
