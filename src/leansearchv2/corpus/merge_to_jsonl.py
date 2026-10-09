"""Step 1b: merge per-file jixia raw output into a single declarations JSONL.

Reads `<workdir>/raw/Mathlib/**/*.{sym,decl}.json` and writes
`<workdir>/merged/mathlib_declarations.jsonl`, one JSON object per line.

Each output row joins the symbol entry with its declaration entry (when
present) and attaches the originating module name (derived from path).
Reference edges (`typeReferences`, `valueReferences`) and `isProp` are
carried through so the informalize step can build a dependency DAG.
When a decl entry exists, its `kind` (e.g. `instance`, `abbrev`) takes
precedence over the sym-level kind, which collapses everything into
`definition` / `theorem` / etc.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from ..config import get_path


log = logging.getLogger("leansearchv2.corpus.merge_to_jsonl")


def _module_name(raw_dir: Path, json_file: Path) -> str:
    rel = json_file.relative_to(raw_dir)
    return str(rel.with_suffix("").with_suffix("")).replace("/", ".")


def _merge_one(raw_dir: Path, sym_file: Path) -> list[dict]:
    decl_file = sym_file.with_name(sym_file.name.replace(".sym.json", ".decl.json"))
    with sym_file.open() as f:
        symbols = json.load(f)
    decl_map: dict[tuple, dict] = {}
    if decl_file.exists():
        with decl_file.open() as f:
            for d in json.load(f):
                decl_map[tuple(d.get("name", []))] = d
    module = _module_name(raw_dir, sym_file)
    out = []
    for sym in symbols:
        sym["module"] = module
        # symbol-level kind is preserved as `symbol_kind` so the dep-aware
        # informalizer can fall back to it when decl-level kind is absent.
        sym["symbol_kind"] = sym.get("kind") or ""
        # `type` mirrors load_raw_to_postgres.choose_type: prefer typeFull,
        # then typeReadable, then typeFallback. The informalize step needs
        # *some* non-empty type for sym-only entries (no decl signature).
        sym["type"] = sym.get("typeFull") or sym.get("typeReadable") or sym.get("typeFallback") or ""
        decl = decl_map.get(tuple(sym.get("name", [])))
        if decl is not None:
            mods = decl.get("modifiers", {}) or {}
            ds = mods.get("docString")
            sym["docstring"] = ds[0] if ds else None
            sym["sourceRange"] = (decl.get("ref") or {}).get("range")
            sym["visibility"] = mods.get("visibility")
            sym["namespace"] = (decl.get("scopeInfo") or {}).get("currNamespace", [])
            sym["auto_generated"] = False
            # decl-level kind (instance / abbrev / example / classInductive)
            # is the user-facing kind; prefer it over the sym-level kind
            # (which collapses these to `definition` / `inductive`).
            decl_kind = decl.get("kind")
            if decl_kind:
                sym["kind"] = decl_kind
            sig = decl.get("signature") or {}
            val = decl.get("value") or {}
            sym["signature"] = (sig.get("pp") if isinstance(sig, dict) else None) or ""
            sym["value"] = (val.get("pp") if isinstance(val, dict) else None) or ""
        else:
            sym["docstring"] = None
            sym["sourceRange"] = None
            sym["visibility"] = None
            sym["namespace"] = []
            sym["auto_generated"] = True
            sym["signature"] = ""
            sym["value"] = ""
        out.append(sym)
    return out


def merge(raw_dir: Path, output_jsonl: Path) -> int:
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw dir not found: {raw_dir}")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    sym_files = sorted(raw_dir.rglob("*.sym.json"))
    log.info(f"Merging {len(sym_files)} symbol files -> {output_jsonl}")
    total = 0
    with output_jsonl.open("w", encoding="utf-8") as out:
        for sym_file in tqdm(sym_files, desc="merge", unit="file"):
            try:
                rows = _merge_one(raw_dir, sym_file)
            except Exception as e:
                log.warning(f"FAIL {sym_file}: {e}")
                continue
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
    log.info(f"Wrote {total} rows.")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge jixia raw JSON into a single JSONL")
    parser.add_argument("--workdir", type=str, default=None,
                        help="Corpus workdir (default: paths.corpus_workdir)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: <workdir>/merged/mathlib_declarations.jsonl)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    workdir = Path(args.workdir) if args.workdir else Path(get_path("CORPUS_WORKDIR", "paths", "corpus_workdir"))
    raw = workdir / "raw"
    output = Path(args.output) if args.output else workdir / "merged" / "mathlib_declarations.jsonl"
    n = merge(raw, output)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
