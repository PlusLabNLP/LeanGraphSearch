"""Source-context helpers for formal failure-memory diagnostics."""


from __future__ import annotations

import re

LEGACY_IMPORT_COMPATIBILITY = {
    "Mathlib.Algebra.BigOperators.Basic": "Mathlib.Algebra.BigOperators.Fin",
    "Mathlib.Data.Nat.Digits": "Mathlib.Data.Nat.Digits.Defs",
}

_TERMINAL_SORRY_RE = re.compile(r":= sorry(?P<trailing>[ \t\r\n]*)\Z")

def normalize_header(header: str) -> str:
    """Translate only pinned, obsolete Lean import commands for v4.28."""

    if type(header) is not str:
        raise ValueError("Lean header must be a string")
    normalized: list[str] = []
    for line in header.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        newline = line[len(stripped) :]
        prefix = "import "
        if stripped.startswith(prefix):
            module = stripped[len(prefix) :].strip()
            replacement = LEGACY_IMPORT_COMPATIBILITY.get(module)
            if replacement is not None:
                indentation = stripped[: len(stripped) - len(stripped.lstrip())]
                line = f"{indentation}{prefix}{replacement}{newline}"
        normalized.append(line)
    return "".join(normalized)

def replace_terminal_sorry(statement: str, proof: str) -> str:
    """Replace only an exact terminal ``:= sorry`` with a ``by`` proof."""

    if type(statement) is not str:
        raise ValueError("formal statement must be a string")
    if (
        type(proof) is not str
        or not proof.startswith("by")
        or (len(proof) > 2 and _is_identifier_continue(proof[2]))
    ):
        raise ValueError("candidate proof must start with `by`")
    match = _TERMINAL_SORRY_RE.search(statement)
    if match is None:
        raise ValueError("formal statement must end with the exact terminal placeholder `:= sorry`")
    return statement[: match.start()] + ":= " + proof + match.group("trailing")

def _is_identifier_continue(character: str) -> bool:
    return character == "_" or character == "'" or character.isalnum()
