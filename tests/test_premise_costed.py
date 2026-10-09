from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_premise_costed.py"
SPEC = importlib.util.spec_from_file_location("reproduce_premise_costed", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
costed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(costed)


def test_json_fingerprint_is_canonical() -> None:
    left = {"b": [2, 3], "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": [2, 3]}
    changed = {"a": {"x": 1}, "b": [3, 2]}

    assert costed._json_fingerprint(left) == costed._json_fingerprint(right)
    assert costed._json_fingerprint(left) != costed._json_fingerprint(changed)


def test_sum_usage_aggregates_every_billed_token_class() -> None:
    total = costed._sum_usage(
        [
            {
                "requests": 2,
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_creation_input_tokens": 30,
                "cache_read_input_tokens": 40,
            },
            {
                "requests": 3,
                "input_tokens": 7,
                "output_tokens": 5,
                "cache_creation_input_tokens": 4,
                "cache_read_input_tokens": 2,
            },
        ]
    )

    assert total == {
        "requests": 5,
        "input_tokens": 107,
        "output_tokens": 25,
        "cache_creation_input_tokens": 34,
        "cache_read_input_tokens": 42,
    }


def test_cost_uses_separate_input_output_and_cache_prices() -> None:
    usage = {
        "requests": 1,
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
    }

    assert costed._cost(
        usage,
        input_price=2.0,
        output_price=10.0,
        cache_write_price=2.5,
        cache_read_price=0.2,
    ) == 14.7


def test_retriever_search_config_records_rank_fusion(monkeypatch) -> None:
    graph = {
        "graph_augment": True,
        "artifact_dir": "graph-artifact",
        "graph_edge_profile": "dep_only",
        "initial_top_n": 50,
        "graph_expand_m": 200,
        "final_top_k": 100,
        "rank_fusion": "qwen_ppr_zscore",
        "ppr_zscore_weight": 0.005,
    }
    monkeypatch.setattr(costed, "get", lambda *_args, **_kwargs: graph)

    kwargs, metadata = costed._retriever_search_config()

    assert kwargs == {
        "graph_augment": True,
        "return_metadata": True,
        "graph_initial_top_n": 50,
        "graph_expand_m": 200,
        "graph_final_top_k": 100,
        "rank_fusion": "qwen_ppr_zscore",
    }
    assert metadata["ppr_zscore_weight"] == 0.005
    assert metadata["graph_edge_profile"] == "dep_only"


def test_run_manifest_prevents_incompatible_resume(tmp_path, monkeypatch) -> None:
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "id": "p0",
                    "formal_statement": "theorem p0 : True",
                    "premise_group": [],
                }
            ]
        )
    )
    output = tmp_path / "out"

    class FakeRetriever:
        def __init__(self, **_kwargs):
            pass

        async def health(self):
            return {"status": "ok", "fake": True}

    async def fake_run_one(row, _retriever, *, branch_budget, **_kwargs):
        return {
            "id": row["id"],
            "branch_budget": branch_budget,
            "retrieved": [],
            "scores": {
                str(k): {"recall_group": 0.0, "covered": 0.0}
                for k in costed.KS
            },
            "api_usage": {key: 0 for key in costed.USAGE_KEYS},
            "api_cost_usd": 0.0,
        }

    monkeypatch.setattr(costed, "StandardClient", FakeRetriever)
    monkeypatch.setattr(costed, "_run_one", fake_run_one)
    monkeypatch.setattr(costed, "get", lambda *_args, **_kwargs: "sonnet5_vertex")

    args = argparse.Namespace(
        benchmark=str(benchmark),
        output=str(output),
        resume=False,
        offset=0,
        limit=1,
        url="http://fake",
        retriever_timeout=1.0,
        parallelism=1,
        branch_budget=2,
        search_top_k=30,
        filter_max_docs=20,
        big_loop=3,
        input_cost_per_mtok=2.0,
        output_cost_per_mtok=10.0,
        cache_write_cost_per_mtok=2.5,
        cache_read_cost_per_mtok=0.2,
        pricing_basis="test",
    )
    asyncio.run(costed._run(args))

    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["immutable"]["branch_budget"] == 2
    assert manifest["immutable"]["problem_parallelism"] == 1
    assert manifest["immutable"]["filter_max_docs"] == 20
    assert manifest["retriever_health"] == {"status": "ok", "fake": True}

    args.resume = True
    args.parallelism = 2
    with pytest.raises(ValueError, match="run manifest does not match"):
        asyncio.run(costed._run(args))
