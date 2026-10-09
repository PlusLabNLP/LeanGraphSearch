"""Lean REPL wrapper used by the prove task.

Defines a thin async `Verifier` interface and one implementation:

    LeanInteractVerifier — wraps the `lean-interact` Python package, which
    spawns a Lean 4 REPL subprocess and reuses it across queries. Add the
    `[lean]` extra (`pip install -e .[lean]`) to install it.

Users who already host an HTTP-based Lean REPL service can swap in their
own implementation against the same `Verifier` interface.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class VerifyResult:
    complete: bool
    error_msg: str
    has_sorry: bool
    raw: dict | None = None

    @property
    def success(self) -> bool:
        return self.complete and not self.has_sorry


class Verifier:
    async def verify(self, code: str, timeout_s: int = 600) -> VerifyResult:  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover
        pass


class LeanInteractVerifier(Verifier):
    """In-process Lean REPL via `lean-interact`.

    `project_dir` should point at a Lake project with Mathlib already
    fetched (`lake exe cache get`). The REPL is started lazily on the
    first `verify()` call and reused for the lifetime of this object.
    """

    def __init__(
        self,
        project_dir: str | None = None,
        lean_version: str | None = None,
        memory_hard_limit_mb: int | None = None,
        auto_build_project: bool = True,
    ) -> None:
        self.project_dir = project_dir
        self.lean_version = lean_version
        self.memory_hard_limit_mb = memory_hard_limit_mb
        self.auto_build_project = auto_build_project
        self._server = None
        self._lock = asyncio.Lock()

    async def _ensure_server(self):
        if self._server is not None:
            return
        try:
            from lean_interact import LeanREPLConfig, LeanServer, LocalProject, TempRequireProject
        except ImportError as e:
            raise RuntimeError(
                "lean-interact not installed. Run `pip install -e '.[lean]'` and ensure elan/lake are on PATH."
            ) from e
        if self.project_dir:
            project = LocalProject(
                directory=self.project_dir,
                auto_build=self.auto_build_project,
            )
        elif self.lean_version:
            project = TempRequireProject(lean_version=self.lean_version, require="mathlib")
        else:
            project = TempRequireProject(lean_version="v4.28.0-rc1", require="mathlib")
        kwargs = {"project": project}
        if self.memory_hard_limit_mb is not None:
            kwargs["memory_hard_limit_mb"] = self.memory_hard_limit_mb
        cfg = LeanREPLConfig(**kwargs)
        self._server = LeanServer(cfg)

    async def verify(self, code: str, timeout_s: int = 600) -> VerifyResult:
        from lean_interact import Command

        async with self._lock:
            await self._ensure_server()
            try:
                response = await asyncio.to_thread(
                    self._server.run, Command(cmd=code), timeout=timeout_s
                )
            except BaseException:
                # A timed-out or crashed REPL process cannot safely serve the
                # queued candidates. Reset it while holding the lock so the
                # next caller starts a fresh server instead of inheriting the
                # failure. Preserve cancellation/exception semantics.
                try:
                    if self._server is not None:
                        self._server.kill()
                except Exception:
                    pass
                self._server = None
                raise
        messages = getattr(response, "messages", []) or []
        sorries = getattr(response, "sorries", []) or []
        errors = [m for m in messages if getattr(m, "severity", "") == "error"]
        warnings = [m for m in messages if getattr(m, "severity", "") == "warning"]
        has_sorry = bool(sorries) or any("sorry" in (getattr(w, "data", "") or "").lower() for w in warnings)
        complete = not errors
        if complete and not has_sorry:
            return VerifyResult(complete=True, error_msg="", has_sorry=False, raw=getattr(response, "model_dump", lambda: None)())
        err_parts: list[str] = []
        for m in errors:
            pos = getattr(m, "start_pos", None) or getattr(m, "pos", None)
            line_number = pos.get("line") if isinstance(pos, dict) else getattr(pos, "line", None)
            line = f"line {line_number}" if line_number is not None else ""
            column = pos.get("column") if isinstance(pos, dict) else getattr(pos, "column", None)
            if line and column is not None:
                line += f", column {column}"
            err_parts.append(f"[{line}] {getattr(m, 'data', '')}".strip())
        if has_sorry and not errors:
            err_parts.append("Proof still contains `sorry`.")
        return VerifyResult(
            complete=complete,
            error_msg="\n".join(err_parts).strip(),
            has_sorry=has_sorry,
            raw=getattr(response, "model_dump", lambda: None)(),
        )

    async def close(self) -> None:
        if self._server is not None:
            try:
                self._server.kill()
            except Exception:
                pass
            self._server = None


def extract_lean_code(text: str, *, preserve_indentation: bool = False) -> str:
    """Pull the *last* lean4 code block out of an LLM response."""
    import re

    blocks = re.findall(r"```(?:lean4?|Lean4?)?[ \t]*\r?\n(.*?)```", text, flags=re.DOTALL)
    if not blocks:
        return text.strip("\r\n") if preserve_indentation else text.strip()
    return blocks[-1].strip("\r\n") if preserve_indentation else blocks[-1].strip()
