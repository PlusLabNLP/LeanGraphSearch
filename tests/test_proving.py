from __future__ import annotations


import importlib.util


from pathlib import Path


import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_draining_before_first_task_does_not_wait_on_empty_set(tmp_path):
    import asyncio
    from leansearchv2.prove.run import ProveProblem
    runner = _load_script("drain_runner", "scripts/reproduce_proving.py")
    drain = tmp_path / "drain"
    drain.touch()
    result = asyncio.run(runner._run_mode(
        "none", [ProveProblem(problem_id="demo", formal_statement="example : True := by")],
        output=tmp_path, retriever=None, parallelism=1, verifier=None,
        recorder=None, benchmark="MathlibMPR-Prop", resume=False,
        max_trajectory_retries=0, max_model_calls=63, max_query_calls=32,
        max_compiler_calls=32, strict_integrity=True, drain_file=drain,
        failure_memory=False,
    ))
    assert result["drained"] is True
    assert result["complete"] is False


def test_invalidated_output_is_rejected_before_manifest_write(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    runner = _load_script("invalid_runner", "scripts/reproduce_proving.py")
    monkeypatch.setattr(runner, "_load_mathlibmpr_prop", lambda: [])
    (tmp_path / "INVALIDATED.md").touch()
    args = SimpleNamespace(benchmark="mathlibmpr_prop", ids=None, offset=0, limit=None, output=str(tmp_path))
    with pytest.raises(RuntimeError, match="invalidated"):
        asyncio.run(runner.run(args))
    assert not (tmp_path / "run_manifest.json").exists()


def test_frozen_benchmark_loaders_have_exact_unique_sizes() -> None:
    runner = _load_script(
        "fate_runner", "scripts/reproduce_proving.py"
    )
    mpr = runner._load_mathlibmpr_prop()
    fate = runner._load_fate_h()
    assert len(mpr) == len({problem.problem_id for problem in mpr}) == 50
    assert len(fate) == len({problem.problem_id for problem in fate}) == 100


def test_query31_corrected_success_requires_integrity_and_no_escape() -> None:
    analysis = _load_script(
        "query31_analysis", "scripts/analyze_proving.py"
    )
    row = {
        "attempts": [{
            "round": 3,
            "success": True,
            "integrity_valid": False,
            "proof": "example : True := by trivial",
        }]
    }
    assert analysis._success_round(row, corrected=False) == 3
    assert analysis._success_round(row, corrected=True) is None
    row["attempts"][0]["integrity_valid"] = True
    assert analysis._success_round(row, corrected=True) == 3
    row["attempts"][0]["proof"] = "example : True := by sorry"
    assert analysis._success_round(row, corrected=True) is None

