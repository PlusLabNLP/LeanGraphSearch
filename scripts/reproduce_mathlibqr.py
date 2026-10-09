#!/usr/bin/env python3
"""Evaluate the fixed MathlibQR fair subset with the shared retrieval backend."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from leansearchv2 import StandardClient
from leansearchv2.eval.search_metrics import ndcg_at_k, recall_at_k, summarize

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("q1a_lean", "q1b_latex", "q1c_natural", "q2_slogan", "q3_nickname", "q4_special_case")
KS = (1, 5, 10, 50, 100)


def queries() -> list[dict]:
    benchmark = json.loads((ROOT / "benchmark/MathlibQR.json").read_text())
    shared = set(json.loads((ROOT / "benchmark/MathlibQR_shared171.json").read_text())["shared_declarations"])
    rows = [
        {"id": f"{entry['id']}_{field}", "query": entry[field].strip(), "target": entry["full_name"]}
        for entry in benchmark if entry["full_name"] in shared
        for field in FIELDS if (entry.get(field) or "").strip()
    ]
    if len(shared) != 171 or len(rows) != 810:
        raise RuntimeError(f"Expected 171 shared declarations / 810 queries; found {len(shared)} / {len(rows)}")
    return rows


async def run(args: argparse.Namespace) -> int:
    rows = queries()
    if args.limit is not None:
        rows = rows[:args.limit]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("Use a new output directory to preserve previous runs")
    client = StandardClient(url=args.url, timeout=600)
    # Appendix B.3 is the MathlibQR-specific deep-ranking exception to K=50.
    request = dict(top_k=200, retrieve_k=200, rerank=True,
                   graph_initial_top_n=50, graph_expand_m=200,
                   graph_final_top_k=200, rank_fusion="qwen_ppr_zscore")
    manifest = {"benchmark": "MathlibQR fair810", "requested_queries": len(rows),
                "request": request, "server_health": await client.health(),
                "input_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                 for p in ("benchmark/MathlibQR.json", "benchmark/MathlibQR_shared171.json")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    with (output / "per_query.jsonl").open("w") as handle:
        for row in rows:
            for mode in ("standard", "graph"):
                record = {**row, "mode": mode}
                try:
                    results = await client.search(row["query"], graph_augment=mode == "graph", **request)
                    names = [".".join(result.result.name) for result in results]
                    record.update(retrieved=names,
                                  ndcg={str(k): ndcg_at_k(names, row["target"], k) for k in KS},
                                  recall={str(k): recall_at_k(names, row["target"], k) for k in KS})
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(record)
    errors = [record for record in records if "error" in record]
    summary = {"complete": not errors and len(rows) == 810,
               "evaluated_queries": len(rows), "failed_requests": len(errors)}
    if not errors:
        summary["systems"] = {mode: summarize([r for r in records if r["mode"] == mode], KS)
                              for mode in ("standard", "graph")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="outputs/mathlibqr")
    parser.add_argument("--limit", type=int, help="Run a preliminary subset instead of all 810 queries")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
