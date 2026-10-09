"""Lean diagnostic and proof-edit classification for failure memory."""


from __future__ import annotations

import difflib

import re

from typing import Any

ERROR_CATEGORIES = (
    "syntax_or_parse",
    "unknown_declaration",
    "type_mismatch",
    "typeclass_inference",
    "rewrite_or_tactic",
    "unsolved_goals",
    "other",
)

_CATEGORY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "syntax_or_parse": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"unexpected token",
            r"unexpected end of input",
            r"unterminated",
            r"invalid syntax",
            r"parser error",
            r"expected (?:a |an )?(?:token|term|command|identifier)",
            r"declaration has metavariables",
            r"candidate proof must start with `by`",
            r"forbidden Lean bypass token 'command:",
        )
    ),
    "unknown_declaration": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"unknown (?:identifier|constant|declaration|namespace|tactic)",
            r"invalid field notation",
            r"invalid field\b",
            r"unknown parser",
        )
    ),
    "type_mismatch": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"type mismatch",
            r"application type mismatch",
            r"has type[\s\S]{0,800}?but is expected to have type",
            r"function expected at",
            r"type is not known to be inhabited",
            r"invalid argument name",
            r"invalid 'calc' step",
            r"numerals are data in Lean, but the expected type is a proposition",
            r"expected type must not contain free variables",
        )
    ),
    "typeclass_inference": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"failed to synthesize",
            r"failed to infer instance",
            r"typeclass instance problem",
            r"cannot synthesize",
            r"synthesized opaque value",
        )
    ),
    "rewrite_or_tactic": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"tactic [`']?rw[`']? failed",
            r"rewrite tactic failed",
            r"did not find instance of the pattern",
            r"`simp` made no progress",
            r"simp made no progress",
            r"tactic [`'][^`']+[`'] failed",
            r"no goals to be solved",
            r"linarith failed",
            r"omega could not prove",
            r"deterministic\) timeout",
            r"maximum recursion depth has been reached",
            r"tactic `decide` proved that the proposition",
            r"'change' tactic failed",
        )
    ),
    "unsolved_goals": tuple(
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"unsolved goals?",
            r"declaration uses 'sorry'",
            r"contains metavariables",
            r"goal is not assigned",
            r"candidate proof rejected: forbidden Lean bypass token 'sorry'",
            r"warning: declaration uses [`']sorry[`']",
            r"`exact\?` could not close the goal",
        )
    ),
}

def classify_lean_error(error: str) -> dict[str, Any]:
    """Classify a Lean diagnostic into the requested repair categories.

    ``primary`` follows the first matching diagnostic position. ``categories``
    is multi-label because one compiler call can report several independent
    errors.
    """

    value = str(error or "")
    matches: list[tuple[int, str]] = []
    for category, patterns in _CATEGORY_PATTERNS.items():
        positions = [match.start() for pattern in patterns if (match := pattern.search(value))]
        if positions:
            matches.append((min(positions), category))
    matches.sort(key=lambda item: (item[0], ERROR_CATEGORIES.index(item[1])))
    categories = [category for _, category in matches]
    if not categories:
        categories = ["other"]
    return {
        "primary": categories[0],
        "categories": categories,
    }

_PATH_RE = re.compile(r"/(?:tmp|var/tmp)/[^\s:]+/Main\.lean")

_LOCATION_RE = re.compile(
    r"(?:line\s+\d+,\s*column\s+\d+:\s*)|(?:Main\.lean:\d+:\d+:\s*)",
    re.IGNORECASE,
)

_SPACE_RE = re.compile(r"[ \t]+")

def normalize_lean_error(error: str, *, max_chars: int = 4000) -> str:
    """Remove volatile paths and locations while retaining the proof state."""

    value = _PATH_RE.sub("Main.lean", str(error or ""))
    value = _LOCATION_RE.sub("", value)
    lines = [_SPACE_RE.sub(" ", line.rstrip()) for line in value.splitlines()]
    value = "\n".join(line for line in lines if line.strip()).strip()
    return value[:max_chars]

def classify_proof_edit(previous: str | None, current: str) -> dict[str, Any]:
    """Classify successive complete proof submissions by retained line content."""

    if previous is None:
        return {
            "kind": "initial",
            "line_similarity": None,
            "retained_line_ratio": None,
            "changed_lines": None,
        }
    old_lines = [line.rstrip() for line in previous.splitlines() if line.strip()]
    new_lines = [line.rstrip() for line in current.splitlines() if line.strip()]
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    denominator = max(len(old_lines), len(new_lines), 1)
    retained = matched / denominator
    changed = len(old_lines) + len(new_lines) - 2 * matched
    similarity = matcher.ratio()
    if (similarity >= 0.70 and retained >= 0.55) or (
        changed <= 8 and retained >= 0.45
    ):
        kind = "local_patch"
    elif similarity < 0.35 or retained <= 0.35:
        kind = "full_rewrite"
    else:
        kind = "substantial_revision"
    return {
        "kind": kind,
        "line_similarity": similarity,
        "retained_line_ratio": retained,
        "changed_lines": changed,
        "previous_lines": len(old_lines),
        "current_lines": len(new_lines),
    }
