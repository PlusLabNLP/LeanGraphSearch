"""Benchmark problem record shared by the Olympiad loader."""


from __future__ import annotations

from dataclasses import asdict, dataclass

from typing import Any, Mapping, TypeVar

_T = TypeVar("_T", bound="JsonRecord")

class JsonRecord:
    """Small dataclass mixin with canonical plain-dict conversion."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: type[_T], payload: Mapping[str, Any]) -> _T:
        return cls(**dict(payload))

@dataclass(frozen=True)
class BenchmarkProblem(JsonRecord):
    problem_id: str
    split: str
    formal_statement: str
    header: str
    informal_statement: str
    source_index: int
    category: str | None = None
