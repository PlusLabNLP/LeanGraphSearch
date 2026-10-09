"""GPU placement and retrieval-source fingerprints."""


from __future__ import annotations

import hashlib

import json

from pathlib import Path

from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

def retrieval_source_fingerprint() -> dict[str, Any]:
    """Return a stable fingerprint of the production retrieval runtime.

    This lives in the deployment/runtime module so importing the production
    server does not pull in the MiniF2F evaluation package.  The schema stays
    compatible with existing runtime-attestation consumers.
    """

    runtime_files = {
        REPO_ROOT / "src/leansearchv2/pipeline.py",
        REPO_ROOT / "src/leansearchv2/server.py",
        REPO_ROOT / "src/leansearchv2/standard_client.py",
    }
    runtime_files.update((REPO_ROOT / "src/leansearchv2/graph").rglob("*.py"))
    files = {
        path.relative_to(REPO_ROOT).as_posix(): _sha256_file(path)
        for path in sorted(runtime_files)
        if path.is_file()
    }
    semantic_material = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "semantic_hash": hashlib.sha256(semantic_material).hexdigest(),
        "source_files": files,
    }

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def choose_pipeline_devices(
    *,
    total_visible_gpus: int,
    requested_num_gpus: int | None,
    device_policy: str = "auto",
) -> dict[str, Any]:
    """Return embedding/reranker device allocation for visible container GPUs."""
    if total_visible_gpus <= 0:
        raise RuntimeError("No GPU available. This pipeline requires GPU.")
    requested = requested_num_gpus if requested_num_gpus is not None else total_visible_gpus
    requested = max(1, min(int(requested), total_visible_gpus))
    if device_policy not in {"auto", "single_gpu", "multi_gpu"}:
        raise ValueError(f"unsupported device_policy: {device_policy}")
    use_single = device_policy == "single_gpu" or requested == 1 or total_visible_gpus == 1
    if use_single:
        return {
            "device_policy": "single_gpu",
            "num_gpus": 1,
            "embedding_device": "cuda:0",
            "reranker_devices": ["cuda:0"],
        }
    return {
        "device_policy": "multi_gpu",
        "num_gpus": requested,
        "embedding_device": "cuda:0",
        "reranker_devices": [f"cuda:{idx}" for idx in range(1, requested)],
    }
