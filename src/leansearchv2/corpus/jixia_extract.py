"""Step 1a: run jixia on every `Mathlib/**/*.lean`, producing one
`.sym.json` + `.decl.json` + `.mod.json` triple per file under
`<workdir>/raw/`.

Requires:
- `lake` + `lean` on PATH (point at the elan-installed binaries).
- `paths.mathlib4` (config.yaml) pointing at a Mathlib4 checkout with
  `lake exe cache get` already run.
- `paths.jixia_bin` pointing at a built jixia executable. By default
  this is `external/jixia/.lake/build/bin/jixia` (the submodule).

Output triples live under `<workdir>/raw/`, mirroring the Mathlib source
tree (e.g. `raw/Mathlib/Data/List/Basic.sym.json`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from tqdm import tqdm

from ..config import get_path


log = logging.getLogger("leansearchv2.corpus.jixia_extract")


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )


def _output_paths(raw_dir: Path, mathlib_dir: Path, lean_file: Path) -> dict[str, Path]:
    rel = lean_file.relative_to(mathlib_dir)
    base = (raw_dir / rel).with_suffix("")
    return {
        "sym": base.with_suffix(".sym.json"),
        "decl": base.with_suffix(".decl.json"),
        "mod": base.with_suffix(".mod.json"),
    }


def _run_jixia_on_file(
    lean_file: Path,
    mathlib_dir: Path,
    raw_dir: Path,
    jixia_bin: Path,
    timeout_s: int,
    resume: bool,
) -> Tuple[str, Path, Optional[str]]:
    outputs = _output_paths(raw_dir, mathlib_dir, lean_file)
    if resume and all(p.exists() for p in outputs.values()):
        return ("skip", lean_file, None)
    outputs["sym"].parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "lake", "env", str(jixia_bin),
            "-s", str(outputs["sym"]),
            "-d", str(outputs["decl"]),
            "-m", str(outputs["mod"]),
            "-i",
            str(lean_file),
        ]
        env = os.environ.copy()
        result = subprocess.run(
            cmd,
            cwd=mathlib_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        if result.returncode != 0:
            return ("error", lean_file, f"exit={result.returncode}\n{result.stderr.strip()[:2000]}")
        if not all(p.exists() for p in outputs.values()):
            return ("error", lean_file, "outputs incomplete")
        return ("success", lean_file, None)
    except subprocess.TimeoutExpired:
        return ("error", lean_file, f"timeout {timeout_s}s")
    except Exception as e:  # pragma: no cover
        return ("error", lean_file, f"{type(e).__name__}: {e}")


def extract(
    mathlib_dir: Path,
    raw_dir: Path,
    jixia_bin: Path,
    workers: int = 32,
    timeout_s: int = 300,
    resume: bool = True,
) -> dict[str, int]:
    mathlib_src = mathlib_dir / "Mathlib"
    if not mathlib_src.exists():
        raise FileNotFoundError(f"Mathlib source dir not found: {mathlib_src}")
    if not jixia_bin.exists():
        raise FileNotFoundError(f"jixia binary not found: {jixia_bin}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    log_file = raw_dir.parent / "logs" / "jixia_extract.log"
    err_file = raw_dir.parent / "logs" / "jixia_errors.txt"
    _setup_logging(log_file)

    lean_files = sorted(mathlib_src.rglob("*.lean"))
    log.info(f"Found {len(lean_files)} .lean files under {mathlib_src}")
    stats = {"total": len(lean_files), "success": 0, "skip": 0, "error": 0}

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                _run_jixia_on_file,
                f, mathlib_dir, raw_dir, jixia_bin, timeout_s, resume,
            ): f for f in lean_files
        }
        with tqdm(total=len(futures), desc="jixia", unit="file") as pbar:
            for fut in as_completed(futures):
                status, lean_file, err = fut.result()
                stats[status] += 1
                pbar.set_postfix(ok=stats["success"], skip=stats["skip"], err=stats["error"])
                pbar.update(1)
                if status == "error":
                    with err_file.open("a") as out:
                        out.write(f"[{datetime.now().isoformat()}] {lean_file}\n{err}\n\n")
                    log.warning(f"FAIL {lean_file.relative_to(mathlib_dir)}: {err.splitlines()[0] if err else ''}")

    log.info(f"Done: total={stats['total']} ok={stats['success']} skip={stats['skip']} err={stats['error']}")
    if stats["error"]:
        log.warning(f"See {err_file} for error details.")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Run jixia over Mathlib4")
    parser.add_argument("--mathlib-dir", type=str, default=None,
                        help="Mathlib4 checkout (default: paths.mathlib4)")
    parser.add_argument("--workdir", type=str, default=None,
                        help="Corpus workdir (default: paths.corpus_workdir)")
    parser.add_argument("--jixia-bin", type=str, default=None,
                        help="jixia executable (default: paths.jixia_bin)")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    mathlib = Path(args.mathlib_dir) if args.mathlib_dir else Path(get_path("MATHLIB4_DIR", "paths", "mathlib4"))
    workdir = Path(args.workdir) if args.workdir else Path(get_path("CORPUS_WORKDIR", "paths", "corpus_workdir"))
    jixia = Path(args.jixia_bin) if args.jixia_bin else Path(get_path("JIXIA_BIN", "paths", "jixia_bin"))
    raw = workdir / "raw"

    stats = extract(mathlib, raw, jixia, workers=args.workers, timeout_s=args.timeout, resume=args.resume)
    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
