"""Strict loader for Google DeepMind's released IMO-LeanProofBench CSV."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .problem import BenchmarkProblem


EXPECTED_COLUMNS = {
    "Problem ID",
    "Problem",
    "Solution",
    "Grading guidelines",
    "Category",
    "Level",
    "Short Answer",
    "Source",
    "Lean Statement",
}
_DECLARATION_RE = re.compile(r"(?m)^(?:theorem|lemma)\s+")
_TERMINAL_PLACEHOLDER_RE = re.compile(r":=\s+by\s+sorry\s*\Z")


def _strip_lean_comments(text: str) -> str:
    """Remove nested block and line comments from a trusted header."""

    output: list[str] = []
    index = 0
    block_depth = 0
    line_comment = False
    while index < len(text):
        if line_comment:
            if text[index] in "\r\n":
                line_comment = False
                output.append(text[index])
            index += 1
            continue
        if block_depth:
            if text.startswith("/-", index):
                block_depth += 1
                index += 2
            elif text.startswith("-/", index):
                block_depth -= 1
                index += 2
            else:
                if text[index] in "\r\n":
                    output.append(text[index])
                index += 1
            continue
        if text.startswith("--", index):
            line_comment = True
            index += 2
        elif text.startswith("/-", index):
            block_depth = 1
            index += 2
        else:
            output.append(text[index])
            index += 1
    if block_depth:
        raise ValueError("unterminated Lean block comment in benchmark header")
    return "".join(output)


def _split_lean_statement(source: str, problem_id: str) -> tuple[str, str]:
    match = _DECLARATION_RE.search(source)
    if match is None:
        raise ValueError(f"{problem_id}: Lean Statement has no theorem or lemma")
    header = _strip_lean_comments(source[: match.start()]).strip()
    statement = source[match.start() :].strip()
    if not _TERMINAL_PLACEHOLDER_RE.search(statement):
        raise ValueError(
            f"{problem_id}: theorem must end with the exact `:= by sorry` placeholder"
        )
    # The shared compiler accepts only a frozen theorem ending in `:= sorry`
    # and replaces exactly that terminal placeholder with the model's `by ...`
    # proof body. This changes no theorem semantics.
    statement = _TERMINAL_PLACEHOLDER_RE.sub(":= sorry", statement)
    return header, statement


def load_imo_leanproofbench(
    path: str | Path,
    *,
    subset: str = "basic",
) -> list[BenchmarkProblem]:
    """Load Basic or Advanced rows without exposing solutions to the model."""

    normalized_subset = subset.strip().lower()
    if normalized_subset not in {"basic", "advanced"}:
        raise ValueError("subset must be basic or advanced")
    target_prefix = f"PB-{normalized_subset.title()}-"
    problems: list[BenchmarkProblem] = []
    seen: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(EXPECTED_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ValueError("IMO-LeanProofBench CSV is missing columns: " + ", ".join(missing))
        for source_index, row in enumerate(reader):
            problem_id = row["Problem ID"].strip()
            if not problem_id.startswith(target_prefix):
                continue
            if problem_id in seen:
                raise ValueError(f"duplicate IMO-LeanProofBench id: {problem_id}")
            seen.add(problem_id)
            header, statement = _split_lean_statement(row["Lean Statement"], problem_id)
            problems.append(
                BenchmarkProblem(
                    problem_id=problem_id,
                    split=normalized_subset,
                    formal_statement=statement,
                    header=header,
                    informal_statement=row["Problem"].strip(),
                    source_index=source_index,
                    category=row["Category"].strip() or None,
                )
            )
    if not problems:
        raise ValueError(f"IMO-LeanProofBench {normalized_subset} subset is empty")
    return problems
