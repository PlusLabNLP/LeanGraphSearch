"""Formal-state normalization and fingerprints for failure memory."""


from __future__ import annotations

import hashlib

import json

import re

from typing import Any

from .diagnostics import classify_lean_error, normalize_lean_error

_LINE_COLUMN_PATTERNS = (
    re.compile(r"line\s+(\d+),\s*column\s+(\d+)", re.IGNORECASE),
    re.compile(r"Main\.lean:(\d+):(\d+)", re.IGNORECASE),
)

_GOAL_PATTERN = re.compile(r"(?m)^\s*⊢\s*(.*)$")

_ERROR_PATTERN = re.compile(r"(?im)(?:^|:\s*)error:")

_HELPER_PATTERN = re.compile(
    r"(?m)^\s*(?:have|let|set|suffices|obtain|rcases)\b"
)

_SPACE_PATTERN = re.compile(r"\s+")

_CATEGORY_STAGE = {
    "syntax_or_parse": 0,
    "unknown_declaration": 1,
    "typeclass_inference": 2,
    "type_mismatch": 2,
    "rewrite_or_tactic": 3,
    "unsolved_goals": 4,
    "other": 1,
}

def _first_error_position(error: str) -> tuple[int, int]:
    positions: list[tuple[int, int, int]] = []
    for pattern in _LINE_COLUMN_PATTERNS:
        for match in pattern.finditer(error):
            positions.append((match.start(), int(match.group(1)), int(match.group(2))))
    if not positions:
        return 0, 0
    _, line, column = min(positions)
    return line, column

def formal_snapshot(
    proof: str,
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    """Extract a stable, JSON-serializable snapshot of Lean formal progress."""

    error = str(result_payload.get("error") or "")
    normalized = normalize_lean_error(error)
    classification = classify_lean_error(error)
    line, column = _first_error_position(error)
    proof_lines = proof.splitlines()
    # Lean locations include the immutable wrapper.  The absolute position is
    # still comparable within one problem.  The clipped prefix is explicitly
    # approximate and is used only as a tiebreaking signal.
    prefix_lines = min(max(line - 1, 0), len(proof_lines)) if line else 0
    prefix_text = "\n".join(proof_lines[:prefix_lines])
    goals = [match.group(1).strip() for match in _GOAL_PATTERN.finditer(error)]
    return {
        "success": bool(result_payload.get("success")),
        "failure_kind": str(result_payload.get("failure_kind") or "other"),
        "primary_error_category": classification["primary"],
        "error_categories": classification["categories"],
        "normalized_error": normalized,
        "error_fingerprint": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "diagnostic_error_count": max(len(_ERROR_PATTERN.findall(error)), 1)
        if error
        else 0,
        "unresolved_goal_count": len(goals),
        "unresolved_goals": goals,
        "error_line": line,
        "error_column": column,
        "approx_elaborated_prefix_lines": prefix_lines,
        "approx_elaborated_prefix_chars": len(prefix_text),
        "helper_steps_before_error": len(_HELPER_PATTERN.findall(prefix_text)),
        "proof_lines": len(proof_lines),
        "proof_chars": len(proof),
    }

def formal_score(snapshot: dict[str, Any]) -> list[int]:
    """Return a lexicographic score used only for soft route prioritization."""

    if snapshot.get("success"):
        return [1, 0, 0, 0, 0, 0]
    goals = int(snapshot.get("unresolved_goal_count", 0))
    errors = int(snapshot.get("diagnostic_error_count", 0))
    category = str(snapshot.get("primary_error_category") or "other")
    return [
        0,
        int(snapshot.get("error_line", 0)),
        int(snapshot.get("approx_elaborated_prefix_chars", 0)),
        -goals,
        int(snapshot.get("helper_steps_before_error", 0)),
        _CATEGORY_STAGE.get(category, 1) - errors,
    ]

def proof_state_fingerprint(
    proof: str | None,
    snapshot: dict[str, Any] | None,
) -> str:
    material = {
        "proof": str(proof or ""),
        "error_fingerprint": (snapshot or {}).get("error_fingerprint"),
        "unresolved_goals": (snapshot or {}).get("unresolved_goals", []),
    }
    rendered = json.dumps(material, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

def normalized_query(query: str) -> str:
    return _SPACE_PATTERN.sub(" ", query.strip().lower())
