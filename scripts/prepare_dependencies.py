#!/usr/bin/env python3
"""List or explicitly download public, revision-pinned reproduction dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the selected assets (many GB)")
    parser.add_argument("--only", nargs="+", choices=["embedder", "reranker", "cuvs", "mathlib", "jixia"])
    args = parser.parse_args()
    lock = json.loads((ROOT / "assets.lock.json").read_text())
    entries = {**lock["huggingface"], **lock["git"]}
    selected = args.only or list(entries)
    for name in selected:
        item = entries[name]
        print(f"{name}: {item.get('repo_id', item.get('url'))} @ {item['revision']} -> {item['local_dir']}", flush=True)
    if not args.download:
        print("Plan only. Add --download to fetch these external resources.")
        return
    for name in selected:
        item = entries[name]
        target = ROOT / item["local_dir"]
        if name in lock["huggingface"]:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=item["repo_id"], repo_type=item["repo_type"],
                              revision=item["revision"], local_dir=str(target))
        else:
            if target.exists():
                if not (target / ".git").exists():
                    raise RuntimeError(f"Refusing to modify an existing non-Git directory: {target}")
                head = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
                dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True).strip()
                if head != item["revision"] or dirty:
                    raise RuntimeError(f"Existing checkout differs or has changes: {target}")
                continue
            target.mkdir(parents=True)
            subprocess.run(["git", "init", str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "remote", "add", "origin", item["url"]], check=True)
            subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", item["revision"]], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True)
    print("Downloads complete. Follow REPRODUCING.md for Lean builds and graph extraction.")


if __name__ == "__main__":
    main()
