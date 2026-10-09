from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_dataset_sizes_and_distinct_proving_ids():
    qr = load("reproduce_mathlibqr")
    proving = load("reproduce_proving")
    assert len(qr.queries()) == 810
    assert len(json.loads((ROOT / "benchmark/MathlibMPR.json").read_text())) == 69
    for problems, expected in [
        (proving._load_fate_h(), 100),
        (proving._load_mathlibmpr_prop(), 50),
        (proving._load_lean_imo("basic"), 30),
        (proving._load_lean_imo("advanced"), 30),
    ]:
        assert len(problems) == len({problem.problem_id for problem in problems}) == expected


def test_qr_pairs_share_query_and_budget_and_failures_are_not_scored(tmp_path, monkeypatch):
    qr = load("reproduce_mathlibqr")
    requests = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def health(self):
            return {"status": "synthetic-test"}

        async def search(self, query, **kwargs):
            requests.append((query, kwargs))
            if kwargs["graph_augment"]:
                raise RuntimeError("synthetic service failure")
            return [SimpleNamespace(result=SimpleNamespace(name=["Example"]))]

    monkeypatch.setattr(qr, "StandardClient", Client)
    monkeypatch.setattr(qr, "queries", lambda: [{"id": "fixture", "query": "example", "target": "Example"}])
    args = SimpleNamespace(limit=1, output=str(tmp_path), url="http://example.test")
    assert asyncio.run(qr.run(args)) == 1
    assert requests[0][0] == requests[1][0]
    baseline = dict(requests[0][1])
    graph = dict(requests[1][1])
    assert baseline.pop("graph_augment") is False
    assert graph.pop("graph_augment") is True
    assert baseline == graph
    assert baseline["retrieve_k"] == baseline["top_k"] == 200
    assert baseline["graph_initial_top_n"] == 50
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["failed_requests"] == 1
    assert summary["complete"] is False
    assert "systems" not in summary
