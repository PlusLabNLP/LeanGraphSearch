"""Soundness-oriented candidate checks for the prove task.

The paper-compatible verifier deliberately accepts any Lean file that compiles
without a live ``sorry``.  The integrity-aware protocol is stricter: the
supplied declarations around one proof hole must remain unchanged and escape
hatches such as ``axiom`` are forbidden.  Keeping the proof prefix and suffix
explicit is important for MathlibMPR, whose source files place ``:= sorry``
inside a namespace and therefore have immutable commands *after* the hole.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass


FORBIDDEN_KEYWORDS = (
    "axiom",
    "constant",
    "opaque",
    "unsafe",
    "unsound",
    "sorry",
    "admit",
)

_FORBIDDEN_RE = re.compile(
    rf"\b(?:{'|'.join(re.escape(word) for word in FORBIDDEN_KEYWORDS)})\b"
)

_SORRY_PROOF_HOLE_RE = re.compile(
    r":=(?P<assign_ws>[ \t\r\n]*)(?:(?P<by>by)(?P<by_ws>[ \t\r\n]+))?sorry\b"
)

# The required prefix already contains the benchmark's imports, namespaces,
# auxiliary declarations, and target theorem header ending in ``:= by``.  A
# new command in the suffix is therefore outside the permitted proof body.
_TOP_LEVEL_COMMAND_RE = re.compile(
    r"(?m)^[ \t]*(?:private[ \t]+|protected[ \t]+|noncomputable[ \t]+|partial[ \t]+)?"
    r"(theorem|lemma|def|abbrev|instance|structure|class|inductive|coinductive|"
    r"namespace|section|end|import|open|export|attribute|set_option|example|"
    r"syntax|macro|macro_rules|elab|elab_rules|declare_syntax_cat|initialize|"
    r"builtin_initialize|foreign|mutual|include|omit|variable|universe|notation|"
    r"infix|infixl|infixr|prefix|postfix|scoped)\b"
)
_HASH_COMMAND_RE = re.compile(r"(?m)^[ \t]*#([A-Za-z_][A-Za-z0-9_]*)\b")


@dataclass(frozen=True)
class IntegrityResult:
    valid: bool
    statement_prefix_valid: bool
    statement_suffix_valid: bool = True
    forbidden_keywords: tuple[str, ...] = ()
    extra_commands: tuple[str, ...] = ()
    error_msg: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "statement_prefix_valid": self.statement_prefix_valid,
            "statement_suffix_valid": self.statement_suffix_valid,
            "forbidden_keywords": list(self.forbidden_keywords),
            "extra_commands": list(self.extra_commands),
            "error_msg": self.error_msg,
        }


@dataclass(frozen=True)
class ProofHoleContract:
    """Immutable source surrounding exactly one generated tactic proof body."""

    prefix: str
    suffix: str

    @property
    def prompt_skeleton(self) -> str:
        marker = "\n  -- Replace this marker with an honest tactic proof body.\n"
        return self.prefix + marker + self.suffix


def extract_generated_proof_body(
    code: str,
    required_prefix: str,
    required_suffix: str = "",
) -> str:
    """Return the generated tactic body from a contract-valid full source.

    Reflection prompts should show the model the failed proof body, not ask it
    to copy the benchmark's immutable source a second time.  A candidate with
    the exact prefix but a damaged suffix is reduced to everything after that
    prefix, leaving the offending tail visible without duplicating the entire
    benchmark source. Prefix-invalid candidates are returned unchanged.
    """

    if not code.startswith(required_prefix):
        return code.strip()
    suffix_valid = bool(required_suffix and code.endswith(required_suffix))
    end = len(code) - len(required_suffix) if suffix_valid else len(code)
    return textwrap.dedent(code[len(required_prefix) : end]).strip("\r\n")


def assemble_generated_proof_body(
    generated: str,
    required_prefix: str,
    required_suffix: str = "",
) -> str:
    """Insert a model-generated tactic body into the immutable source.

    The corrected protocol asks the model for only the contents of the
    existing ``:= by`` block.  Host-side assembly makes statement copying a
    non-task and removes an otherwise common source of false/escaped success.
    Exact legacy full-source responses remain accepted for compatibility and
    are still checked by :func:`validate_proof_integrity` before compilation.
    """

    candidate = textwrap.dedent(generated).strip("\r\n")
    # Preserve a legacy full-source response whenever it starts from the exact
    # prefix, even if it damaged the suffix.  The integrity validator can then
    # report the precise suffix violation instead of nesting the whole file as
    # if it were a tactic body.
    if candidate.startswith(required_prefix):
        return candidate

    # Models sometimes wrap a requested tactic body in one extra ``by``.  The
    # immutable prefix already ends in ``:= by``, so remove only a leading,
    # standalone wrapper token.  Never rewrite an internal ``exact by``.
    candidate = re.sub(r"^by(?=\s|$)", "", candidate, count=1).strip("\r\n")
    body = textwrap.dedent(candidate).strip()
    indented_body = textwrap.indent(body, "  ") if body else ""
    separator = "\n" if indented_body else ""
    # A trailing model line comment must not swallow a same-line source suffix.
    suffix_separator = "\n" if required_suffix and not required_suffix.startswith(("\n", "\r")) else ""
    return required_prefix + separator + indented_body + suffix_separator + required_suffix


def _strip_comments_and_strings(text: str) -> str:
    """Replace Lean comments and string contents while preserving newlines.

    Lean block comments nest.  Preserving newlines keeps the top-level command
    checks meaningful while avoiding false positives from explanatory comments
    or string literals.
    """

    out: list[str] = []
    i = 0
    block_depth = 0
    in_line_comment = False
    in_string = False
    escaped = False

    while i < len(text):
        pair = text[i : i + 2]
        char = text[i]

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                out.append("\n")
            else:
                out.append(" ")
            i += 1
            continue

        if block_depth:
            if pair == "/-":
                block_depth += 1
                out.extend((" ", " "))
                i += 2
            elif pair == "-/":
                block_depth -= 1
                out.extend((" ", " "))
                i += 2
            else:
                out.append("\n" if char == "\n" else " ")
                i += 1
            continue

        if in_string:
            if char == "\n":
                out.append("\n")
            else:
                out.append(" ")
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            i += 1
            continue

        if pair == "--":
            in_line_comment = True
            out.extend((" ", " "))
            i += 2
        elif pair == "/-":
            block_depth = 1
            out.extend((" ", " "))
            i += 2
        elif char == '"':
            in_string = True
            out.append(" ")
            i += 1
        else:
            out.append(char)
            i += 1

    return "".join(out)


def find_forbidden_keywords(text: str) -> tuple[str, ...]:
    """Return soundness escape hatches used in Lean code.

    Comments and string literals are ignored, so an explanatory mention of
    ``sorry`` or ``axiom`` does not make an otherwise valid proof fail.  This
    helper is intentionally independent of the immutable-prefix policy: the
    paper-compatible MPR runner emits complete files, while a soundness audit
    must still reject newly introduced axioms even when it does not enforce
    FATE's exact proof-body format.
    """

    sanitized = _strip_comments_and_strings(text)
    return tuple(sorted(set(_FORBIDDEN_RE.findall(sanitized))))


def split_sorry_proof_hole(code: str) -> ProofHoleContract:
    """Turn one live ``:= sorry``/``:= by sorry`` into a ``:= by`` contract.

    Comments and strings are ignored while locating the hole.  Refusing zero or
    multiple live placeholders prevents silently choosing the wrong theorem in
    a benchmark file.
    """

    sanitized = _strip_comments_and_strings(code)
    matches = list(_SORRY_PROOF_HOLE_RE.finditer(sanitized))
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one live `:= sorry` or `:= by sorry` proof hole; "
            f"found {len(matches)}"
        )
    match = matches[0]
    assignment_ws = code[match.start("assign_ws") : match.end("assign_ws")]
    prefix = code[: match.start()] + ":=" + assignment_ws + "by"
    suffix = code[match.end() :]
    return ProofHoleContract(prefix=prefix, suffix=suffix)


def validate_proof_integrity(
    code: str,
    required_prefix: str,
    required_suffix: str = "",
) -> IntegrityResult:
    """Validate an immutable prefix/suffix and proof-body-only contract."""

    prefix_valid = code.startswith(required_prefix)
    suffix_valid = not required_suffix or code.endswith(required_suffix)
    # Scan only the generated body when either boundary is valid, so legal
    # benchmark declarations in the immutable source do not trigger policy.
    body_start = len(required_prefix) if prefix_valid else 0
    body_end = (
        len(code) - len(required_suffix)
        if required_suffix and suffix_valid
        else len(code)
    )
    generated = code[body_start:body_end] if body_start <= body_end else code
    sanitized = _strip_comments_and_strings(generated)
    forbidden = find_forbidden_keywords(generated)
    commands = [match.group(1) for match in _TOP_LEVEL_COMMAND_RE.finditer(sanitized)]
    commands.extend(f"#{match.group(1)}" for match in _HASH_COMMAND_RE.finditer(sanitized))
    extra_commands = tuple(sorted(set(commands)))

    errors: list[str] = []
    if not prefix_valid:
        errors.append(
            "the candidate does not begin with the exact immutable formal-statement prefix"
        )
    if not suffix_valid:
        errors.append(
            "the candidate does not end with the exact immutable source suffix"
        )
    if forbidden:
        errors.append("forbidden keyword(s): " + ", ".join(forbidden))
    if extra_commands:
        errors.append(
            "new top-level command(s) outside the target proof body: "
            + ", ".join(extra_commands)
        )

    if errors:
        detail = "; ".join(errors)
        message = (
            "Proof integrity violation: "
            + detail
            + ". Discard the invalid construction. The host preserves the immutable "
            "prefix and suffix; return only an honest replacement tactic body using "
            "existing Mathlib declarations. Do not copy or edit the surrounding source, "
            "add assumptions or declarations, or use trusted-code escape hatches."
        )
    else:
        message = ""

    return IntegrityResult(
        valid=not errors,
        statement_prefix_valid=prefix_valid,
        statement_suffix_valid=suffix_valid,
        forbidden_keywords=forbidden,
        extra_commands=extra_commands,
        error_msg=message,
    )
