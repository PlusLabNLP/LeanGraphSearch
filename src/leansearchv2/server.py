"""FastAPI server for LeanSearch v2 standard mode.

Run with::

    ./scripts/serve.sh

or directly with uvicorn::

    python -m uvicorn leansearchv2.server:app --host 0.0.0.0 --port 8000

Paths and tuning knobs are read from ``config.yaml`` at the repository root;
each can be overridden by an environment variable of the same name.
"""

from __future__ import annotations

import logging
import hashlib
import threading
import time
from contextlib import asynccontextmanager

import dotenv
from fastapi import Body, FastAPI, HTTPException

from .config import get, get_path
from .runtime import retrieval_source_fingerprint
from .graph.artifact import GraphArtifactError
from .graph.config import GraphConfig
from .pipeline import RetrievalPipeline, SearchResult


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
LOADED_RETRIEVAL_SOURCE_FINGERPRINT = retrieval_source_fingerprint()


def _as_bool(value: object, *, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _optional_positive_int(value: object, *, name: str) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _search_call(**kwargs):
    semaphore = getattr(app, "request_semaphore", None)
    if semaphore is None:
        return app.pipeline.search(**kwargs)
    with semaphore:
        return app.pipeline.search(**kwargs)




@asynccontextmanager
async def lifespan(app: FastAPI):
    dotenv.load_dotenv()
    log.info("Starting...")
    num_gpus = get("NUM_GPUS", "serve", "num_gpus")
    graph_config = GraphConfig.from_dict(get("GRAPH_CONFIG", "graph", default={}))
    device_policy = str(get("DEVICE_POLICY", "serve", "device_policy", default="auto"))
    single_gpu_deploy = device_policy == "single_gpu" or (
        num_gpus is not None and int(num_gpus) == 1
    )
    max_rerank_batch_size = _optional_positive_int(
        get(
            "MAX_RERANK_BATCH_SIZE",
            "serve",
            "max_rerank_batch_size",
            default=8 if single_gpu_deploy else None,
        ),
        name="max_rerank_batch_size",
    )
    max_concurrent_requests = _optional_positive_int(
        get("MAX_CONCURRENT_REQUESTS", "serve", "max_concurrent_requests", default=1),
        name="max_concurrent_requests",
    ) or 1
    queue_enabled = _as_bool(
        get(
            "REQUEST_QUEUE_ENABLED",
            "serve",
            "request_queue_enabled",
            default=single_gpu_deploy,
        ),
        name="request_queue_enabled",
    )
    app.pipeline = RetrievalPipeline(
        vectordb_dir=get_path("VECTORDB_DIR", "paths", "cuvs_db"),
        embedding_model_path=get_path("EMBEDDING_MODEL_PATH", "models", "embedder"),
        reranker_model_path=get_path("RERANKER_MODEL_PATH", "models", "reranker"),
        num_gpus=int(num_gpus) if num_gpus is not None else None,
        gpu_memory_utilization=float(
            get("GPU_MEMORY_UTILIZATION", "serve", "gpu_memory_utilization", default=0.9)
        ),
        graph_config=graph_config,
        device_policy=device_policy,
        embedding_dtype=get(
            "EMBEDDING_DTYPE",
            "serve",
            "embedding_dtype",
            default="bfloat16" if single_gpu_deploy else None,
        ),
        max_rerank_batch_size=max_rerank_batch_size,
        reranker_required=_as_bool(
            get(
                "RERANKER_REQUIRED",
                "graph",
                "reranker_required",
                default=True if single_gpu_deploy else graph_config.reranker_required,
            ),
            name="reranker_required",
        ),
    )
    app.request_queue_enabled = queue_enabled
    app.max_concurrent_requests = max_concurrent_requests
    app.request_semaphore = (
        threading.BoundedSemaphore(max_concurrent_requests) if queue_enabled else None
    )
    log.info("Ready")
    yield


app = FastAPI(lifespan=lifespan)


def _effective_graph_augment(requested: bool | None) -> bool:
    """Resolve an optional request override against the loaded config."""

    if requested is not None:
        return requested
    return bool(app.pipeline.graph_config.graph_augment)


@app.post("/search")
def search(
    query: list[str] = Body(...),
    num_results: int = Body(default=10),
    rerank: bool = Body(default=True),
    retrieve_k: int | None = Body(default=None),
    graph_augment: bool | None = Body(default=None),
    graph_initial_top_n: int | None = Body(default=None),
    graph_expand_m: int | None = Body(default=None),
    graph_final_top_k: int | None = Body(default=None),
    rank_fusion: str | None = Body(default=None),
    return_metadata: bool = Body(default=True),
) -> list[list[SearchResult]]:
    t0 = time.time()
    effective_graph_augment = _effective_graph_augment(graph_augment)
    try:
        results = [
            _search_call(
                query=q,
                top_k=num_results,
                rerank=rerank,
                retrieve_k=retrieve_k,
                graph_augment=effective_graph_augment,
                graph_initial_top_n=graph_initial_top_n,
                graph_expand_m=graph_expand_m,
                graph_final_top_k=(
                    graph_final_top_k if graph_final_top_k is not None else num_results
                ),
                rank_fusion=rank_fusion,
                return_metadata=return_metadata,
            )
            for q in query
        ]
    except GraphArtifactError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    log.info(
        f"/search | {time.time() - t0:.2f}s | n={len(query)} | rerank={rerank} | "
        f"graph_augment={effective_graph_augment} | {query}"
    )
    return results




@app.post("/search_with_profile")
def search_with_profile(
    query: list[str] = Body(...),
    num_results: int = Body(default=10),
    rerank: bool = Body(default=True),
    retrieve_k: int | None = Body(default=None),
    graph_augment: bool | None = Body(default=None),
    graph_initial_top_n: int | None = Body(default=None),
    graph_expand_m: int | None = Body(default=None),
    graph_final_top_k: int | None = Body(default=None),
    rank_fusion: str | None = Body(default=None),
    return_metadata: bool = Body(default=True),
) -> dict:
    t0 = time.time()
    effective_graph_augment = _effective_graph_augment(graph_augment)
    all_results = []
    all_profiles = []
    try:
        for q in query:
            results, profile = _search_call(
                query=q,
                top_k=num_results,
                return_profile=True,
                rerank=rerank,
                retrieve_k=retrieve_k,
                graph_augment=effective_graph_augment,
                graph_initial_top_n=graph_initial_top_n,
                graph_expand_m=graph_expand_m,
                graph_final_top_k=(
                    graph_final_top_k if graph_final_top_k is not None else num_results
                ),
                rank_fusion=rank_fusion,
                return_metadata=return_metadata,
            )
            all_results.append(results)
            all_profiles.append(profile)
    except GraphArtifactError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    total_time = time.time() - t0
    log.info(
        f"/search_with_profile | {total_time:.2f}s | n={len(query)} | rerank={rerank} | "
        f"graph_augment={effective_graph_augment} | {query}"
    )

    return {
        "results": all_results,
        "profiles": all_profiles,
        "total_request_time": total_time,
    }


@app.get("/health")
def health() -> dict:
    graph_config = app.pipeline.graph_config
    graph_artifact_dir = (
        str(graph_config.artifact_dir) if graph_config.artifact_dir is not None else None
    )
    edge_stats_sha256 = None
    if graph_config.artifact_dir is not None:
        edge_stats = graph_config.artifact_dir / "edge_stats.json"
        if edge_stats.is_file():
            edge_stats_sha256 = hashlib.sha256(edge_stats.read_bytes()).hexdigest()
    return {
        "status": "ok",
        "num_gpus": app.pipeline.num_gpus,
        "index_backend": "cuvs",
        "device_policy": getattr(app.pipeline, "device_policy", None),
        "startup_memory_log": getattr(app.pipeline, "startup_memory_log", {}),
        "embedding_dtype": getattr(app.pipeline, "embedding_dtype", None),
        "max_rerank_batch_size": getattr(app.pipeline, "max_rerank_batch_size", None),
        "reranker_required": getattr(app.pipeline, "reranker_required", False),
        "request_queue_enabled": getattr(app, "request_queue_enabled", False),
        "max_concurrent_requests": getattr(app, "max_concurrent_requests", None),
        "vectordb_dir": str(app.pipeline.vectordb_dir),
        "embedding_model_path": str(app.pipeline.embedding_model_path),
        "reranker_model_path": str(app.pipeline.reranker_model_path),
        "loaded_retrieval_source_fingerprint": LOADED_RETRIEVAL_SOURCE_FINGERPRINT,
        "graph": {
            "artifact_dir": graph_artifact_dir,
            "edge_stats_sha256": edge_stats_sha256,
            "graph_edge_profile": graph_config.graph_edge_profile,
            "initial_top_n": graph_config.initial_top_n,
            "graph_expand_m": graph_config.graph_expand_m,
            "final_top_k": graph_config.final_top_k,
            "rank_fusion": graph_config.rank_fusion,
            "rerank_graph_union": graph_config.rerank_graph_union,
            "reranker_required": graph_config.reranker_required,
            "edge_weights": graph_config.edge_weights.as_dict(),
        },
    }
