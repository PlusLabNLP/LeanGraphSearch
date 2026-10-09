# SPDX-License-Identifier: MIT
"""Inspect LeanSearch declaration record files for graph-relevant fields."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

REFERENCE_FIELDS = ["dependencies", "deps", "references", "typeReferences", "valueReferences"]
INSPECTION_FIELDS = [
    *REFERENCE_FIELDS,
    "module",
    "module_name",
    "name",
    "kind",
    "type",
    "signature",
    "value",
    "informal_name",
    "informal_description",
    "isProp",
]
SUPPORTED_SUFFIXES = (".json", ".jsonl", ".json.gz", ".jsonl.gz", ".parquet")


def normalize_declaration_name(value: Any) -> str | None:
    if isinstance(value, list):
        return ".".join(str(part) for part in value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value not in (None, ""):
        return str(value)
    return None


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def is_supported_record_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def record_files(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(p for p in sorted(path.rglob("*")) if p.is_file() and is_supported_record_path(p))
        elif path.is_file() and is_supported_record_path(path):
            out.append(path)
    return out


def _iter_json_payload(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        else:
            yield payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL record: {exc}") from exc
            if isinstance(payload, dict):
                yield payload


def _iter_json(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as f:
        payload = json.load(f)
    yield from _iter_json_payload(payload)


def _iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        try:
            import pandas as pd  # type: ignore
        except Exception:
            raise RuntimeError(f"Cannot inspect parquet file {path}; install pyarrow or pandas") from exc
        frame = pd.read_parquet(path)
        for record in frame.to_dict(orient="records"):
            if isinstance(record, dict):
                yield record
        return

    table = pq.read_table(path)
    for record in table.to_pylist():
        if isinstance(record, dict):
            yield record


def iter_records_from_path(path: str | Path) -> Iterator[dict[str, Any]]:
    src = Path(path)
    if src.is_dir():
        for file_path in record_files([src]):
            yield from iter_records_from_path(file_path)
        return
    name = src.name.lower()
    if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
        yield from _iter_jsonl(src)
    elif name.endswith(".json") or name.endswith(".json.gz"):
        yield from _iter_json(src)
    elif name.endswith(".parquet"):
        yield from _iter_parquet(src)
    else:
        raise ValueError(f"Unsupported record file type: {src}")


def inspect_record_sources(paths: Iterable[str | Path]) -> dict[str, Any]:
    files = record_files(paths)
    field_presence = Counter()
    nonempty_counts = Counter({field: 0 for field in INSPECTION_FIELDS})
    declaration_names: set[str] = set()
    file_reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    total_rows = 0

    for file_path in files:
        file_rows = 0
        try:
            for record in iter_records_from_path(file_path):
                total_rows += 1
                file_rows += 1
                for field in record:
                    field_presence[str(field)] += 1
                for field in INSPECTION_FIELDS:
                    if is_nonempty(record.get(field)):
                        nonempty_counts[field] += 1
                name = normalize_declaration_name(record.get("name"))
                if name:
                    declaration_names.add(name)
        except Exception as exc:
            errors.append({"path": str(file_path), "error": str(exc)})
        file_reports.append({"path": str(file_path), "rows": file_rows})

    structured_refs = sum(nonempty_counts[field] for field in REFERENCE_FIELDS)
    official_has_refs = structured_refs > 0
    return {
        "files": file_reports,
        "errors": errors,
        "total_rows": total_rows,
        "distinct_declaration_names": len(declaration_names),
        "fields_present": sorted(field_presence),
        "field_presence_counts": dict(sorted(field_presence.items())),
        "nonempty_counts": {field: int(nonempty_counts[field]) for field in INSPECTION_FIELDS},
        "official_jsonl_has_structured_refs": official_has_refs,
        "released_final_jsonl_no_structured_refs": not official_has_refs,
    }


def write_schema_report(paths: Iterable[str | Path], output_path: str | Path) -> dict[str, Any]:
    report = inspect_record_sources(paths)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
