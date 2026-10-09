"""LeanSearch v2 standard-mode retrieval pipeline.

Qwen3-Embedding-8B for query encoding, cuVS CAGRA for vector search, and
Qwen3-Reranker-8B (HuggingFace, one replica per GPU) for cross-encoder rerank.
"""

from __future__ import annotations

import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import choose_pipeline_devices
from .graph.artifact import load_graph_artifact
from .graph.augment import CandidateScore, augment_candidates
from .graph.config import GraphConfig
from .standard_client import ResultData, SearchResult


_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = float(np.mean(values))
    stdev = float(np.std(values))
    if stdev <= 0.0:
        return [0.0 for _ in values]
    return [(value - mean) / stdev for value in values]


def apply_qwen_ppr_zscore_fusion(
    rows: list[dict[str, Any]],
    *,
    weight: float = 0.005,
) -> list[dict[str, Any]]:
    """Rank graph-union rows by ``qwen_score + weight * PPR_zscore``.

    This is the optional scoring-layer candidate from the fixed-budget
    diagnostics. It is disabled by default; configurations must opt into it
    explicitly with ``rank_fusion=qwen_ppr_zscore``.
    """
    scored = [dict(row) for row in rows]
    ppr_zscores = _zscore([float(row.get("graph_score") or 0.0) for row in scored])
    for row, ppr_zscore in zip(scored, ppr_zscores):
        row["ppr_score_zscore_within_query"] = ppr_zscore
        row["fusion_score"] = float(row.get("qwen_score") or 0.0) + weight * ppr_zscore
    return sorted(scored, key=lambda row: (-float(row["fusion_score"]), int(row.get("_position", 0))))


class RetrievalPipeline:
    INSTRUCTION = (
        "Retrieve the informal + formal representation of items in Mathlib4 "
        "that is mathematically relevant to the query. If the query ask for "
        "theorem or lemma, you shall try to find the entry that starts with "
        "theorem. If the query ask for a definition, you shall try to find "
        "the entry that starts with either definition or instance."
    )

    def __init__(
        self,
        vectordb_dir: str,
        embedding_model_path: str,
        reranker_model_path: str,
        num_gpus: int | None = None,
        gpu_memory_utilization: float = 0.9,
        graph_config: GraphConfig | None = None,
        device_policy: str = "auto",
        embedding_dtype: str | None = None,
        max_rerank_batch_size: int | None = None,
        reranker_required: bool | None = None,
    ) -> None:
        # Load GPU libraries only when constructing the actual service.
        # Pure ranking and request-contract tests use no GPU dependencies.
        global cp, torch, cagra
        import cupy as cp
        import torch
        from cuvs.neighbors import cagra

        if embedding_dtype not in {None, "float32", "float16", "bfloat16"}:
            raise ValueError(
                "embedding_dtype must be one of: float32, float16, bfloat16, or None"
            )
        if max_rerank_batch_size is not None and (
            type(max_rerank_batch_size) is not int or max_rerank_batch_size < 1
        ):
            raise ValueError("max_rerank_batch_size must be a positive integer or None")
        if reranker_required is not None and type(reranker_required) is not bool:
            raise ValueError("reranker_required must be a bool or None")
        self.vectordb_dir = Path(vectordb_dir)
        self.embedding_model_path = embedding_model_path
        self.reranker_model_path = reranker_model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.graph_config = graph_config or GraphConfig()
        self.device_policy = device_policy
        self.embedding_dtype = embedding_dtype
        self.max_rerank_batch_size = max_rerank_batch_size
        self.reranker_required = (
            self.graph_config.reranker_required
            if reranker_required is None
            else reranker_required
        )
        self._rerank_profile_local = threading.local()
        self._graph_artifact = None
        self._graph_artifact_path: Path | None = None
        self._graph_reverse_adjacency = None
        self.startup_memory_log: dict[str, Any] = {}

        print(f"Loading vector database from {self.vectordb_dir}")

        with open(self.vectordb_dir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        self.data = metadata["data"]
        self.embedding_dim = metadata["embedding_dim"]

        with open(self.vectordb_dir / "texts.pkl", "rb") as f:
            self.texts = pickle.load(f)

        total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        device_plan = choose_pipeline_devices(
            total_visible_gpus=total_gpus,
            requested_num_gpus=num_gpus,
            device_policy=device_policy,
        )
        self.num_gpus = int(device_plan["num_gpus"])
        self.embedding_device = str(device_plan["embedding_device"])
        self.reranker_devices = list(device_plan["reranker_devices"])
        self.startup_memory_log["device_policy"] = device_plan["device_policy"]
        self.startup_memory_log["visible_gpu_count"] = total_gpus
        if torch.cuda.is_available() and total_gpus:
            self.startup_memory_log["gpu_name"] = torch.cuda.get_device_name(0)

        self._load_cuvs_index()

        print(f"\n{'=' * 60}")
        print(f"GPU Allocation:")
        print(f"  Available: {total_gpus}, Using: {self.num_gpus}")
        print(f"  Device policy: {self.startup_memory_log['device_policy']}")
        print(f"  Embedding: {self.embedding_device}")
        print(f"  Vector index: cuvs")
        print(f"  Reranker: HuggingFace Qwen3-Reranker (yes/no 2-way log_softmax)")
        print(f"  Reranker devices: {self.reranker_devices}")
        print(f"{'=' * 60}\n")

        self._load_models()

    def _load_cuvs_index(self) -> None:
        cuvs_index_path = self.vectordb_dir / "cuvs_index.bin"
        if not cuvs_index_path.exists():
            raise RuntimeError(f"cuVS index not found at {cuvs_index_path}")

        print(f"Loading cuVS CAGRA index from {cuvs_index_path}...")
        self.cuvs_index = cagra.load(str(cuvs_index_path))
        # itopk_size must satisfy ceildiv(itopk_size, 32) * 32 >= topk in
        # multi-cta search mode. itopk=1024 with search_width=4 keeps
        # recall@200 close to 1.0 at negligible latency on H200.
        self.cuvs_search_params = cagra.SearchParams(
            itopk_size=1024, search_width=4, max_iterations=64
        )
        print("cuVS index loaded successfully")

    def _get_gpu_free_memory(self, gpu_idx: int) -> tuple[float, float]:
        free, total = torch.cuda.mem_get_info(gpu_idx)
        return free / (1024**3), total / (1024**3)

    def _record_memory_snapshot(self, label: str) -> None:
        if not torch.cuda.is_available() or self.num_gpus <= 0:
            return
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free, total = self._get_gpu_free_memory(0)
        self.startup_memory_log[label] = {
            "allocated_gib": allocated,
            "reserved_gib": reserved,
            "free_gib": free,
            "total_gib": total,
        }

    def _load_models(self) -> None:
        import sys

        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.cuda.empty_cache()
        free_gib, total_gib = self._get_gpu_free_memory(0)
        print(f"GPU memory before embedding: {free_gib:.1f}/{total_gib:.1f} GiB free")
        sys.stdout.flush()

        print(f"Loading embedding model from {self.embedding_model_path}...")
        sys.stdout.flush()
        embedding_kwargs: dict[str, Any] = {}
        if self.embedding_dtype is not None:
            embedding_kwargs["model_kwargs"] = {
                "torch_dtype": {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }[self.embedding_dtype]
            }
        self.embedding_model = SentenceTransformer(
            self.embedding_model_path,
            device=self.embedding_device,
            **embedding_kwargs,
        )
        print("Embedding model loaded")
        self._record_memory_snapshot("after_embedding_load")
        sys.stdout.flush()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        time.sleep(0.5)

        print(f"Loading reranker tokenizer from {self.reranker_model_path}...")
        sys.stdout.flush()
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            self.reranker_model_path, padding_side="left"
        )
        self.yes_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.reranker_tokenizer.convert_tokens_to_ids("no")
        self.prefix_tokens = self.reranker_tokenizer.encode(
            _PREFIX, add_special_tokens=False
        )
        self.suffix_tokens = self.reranker_tokenizer.encode(
            _SUFFIX, add_special_tokens=False
        )
        self.max_model_len = 8192
        self.body_max_len = (
            self.max_model_len - len(self.prefix_tokens) - len(self.suffix_tokens)
        )

        print(f"Loading reranker replicas on {self.reranker_devices}...")
        sys.stdout.flush()
        self.reranker_models: list = []
        for d in self.reranker_devices:
            print(f"  loading reranker on {d}")
            sys.stdout.flush()
            m = (
                AutoModelForCausalLM.from_pretrained(
                    self.reranker_model_path,
                    torch_dtype=torch.float16 if d.startswith("cuda") else torch.float32,
                )
                .to(d)
                .eval()
            )
            self.reranker_models.append(m)
        self._record_memory_snapshot("after_reranker_load")

        print(f"\n{'=' * 60}")
        print(f"All {len(self.reranker_models)} reranker replica(s) ready")
        print(f"{'=' * 60}")

    def _search_cuvs(self, query_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query_gpu = cp.asarray(query_embedding, dtype=cp.float32)
        distances, indices = cagra.search(
            self.cuvs_search_params, self.cuvs_index, query_gpu, k
        )
        return cp.asnumpy(distances), cp.asnumpy(indices)

    def _format_pair(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: list[str], device: str):
        """Tokenize pairs, prepend prefix + append suffix tokens, pad to a
        batch tensor on ``device``. Documents are truncated from the back so
        prompts fit within ``max_model_len``.
        """
        inputs = self.reranker_tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            add_special_tokens=False,
            max_length=self.body_max_len,
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.reranker_tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_model_len
        )
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        return inputs

    def _compute_scores(self, inputs, model) -> list[float]:
        """Score a batch by reading the assistant-position logits and computing
        ``P(yes) / (P(yes) + P(no))`` over the two token-id columns directly.

        Indexing the full vocab logits avoids the top-K logprobs truncation that
        affects ``generate``-style APIs when the reranker is highly confident
        and ``yes`` or ``no`` falls outside the returned top-K.
        """
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
            tv = logits[:, self.yes_id]
            fv = logits[:, self.no_id]
            stacked = torch.stack([fv, tv], dim=1)
            stacked = torch.nn.functional.log_softmax(stacked, dim=1)
            return stacked[:, 1].exp().tolist()

    def _build_result(
        self,
        i: int,
        distance: float,
        graph_metadata: dict | None = None,
    ) -> SearchResult:
        payload = {
            "result": ResultData(
                module_name=self.data[i].get("module_name", []),
                kind=self.data[i].get("kind", ""),
                name=self.data[i].get("name", []),
                signature=self.data[i].get("signature", ""),
                type=self.data[i].get("type", ""),
                value=self.data[i].get("value"),
                docstring=self.data[i].get("docstring"),
                informal_name=self.data[i].get("informal_name"),
                informal_description=self.data[i].get("informal_description"),
            ),
            "distance": float(distance),
        }
        if graph_metadata:
            payload.update(graph_metadata)
        return SearchResult(**payload)

    def _graph_request_config(
        self,
        *,
        top_k: int,
        graph_augment: bool,
        graph_initial_top_n: int | None,
        graph_expand_m: int | None,
        graph_final_top_k: int | None,
        rank_fusion: str | None = None,
    ) -> GraphConfig:
        updates = {
            "graph_augment": graph_augment,
            "final_top_k": graph_final_top_k or self.graph_config.final_top_k or top_k,
        }
        if graph_initial_top_n is not None:
            updates["initial_top_n"] = graph_initial_top_n
        if graph_expand_m is not None:
            updates["graph_expand_m"] = graph_expand_m
        if rank_fusion is not None:
            updates["rank_fusion"] = rank_fusion
        return replace(self.graph_config, **updates)

    def _get_graph_artifact(self, cfg: GraphConfig):
        artifact_dir = Path(cfg.artifact_dir) if cfg.artifact_dir else self.vectordb_dir / "graph"
        if self._graph_artifact is None or self._graph_artifact_path != artifact_dir:
            self._graph_artifact = load_graph_artifact(artifact_dir)
            self._graph_artifact_path = artifact_dir
            self._graph_reverse_adjacency = None
        return self._graph_artifact


    def _augment_with_graph(
        self,
        standard_candidates: list[CandidateScore],
        cfg: GraphConfig,
    ) -> list[CandidateScore]:
        artifact = self._get_graph_artifact(cfg)
        return augment_candidates(
            standard_candidates,
            artifact=artifact,
            config=cfg,
            final_top_k=cfg.final_top_k or len(standard_candidates),
        )

    def _graph_metadata(self, candidate: CandidateScore) -> dict:
        metadata = {
            "graph_score": candidate.graph_score,
            "seed_source": candidate.seed_source,
        }
        if candidate.graph_evidence:
            metadata["graph_evidence"] = candidate.graph_evidence
        if candidate.combined_score is not None:
            metadata["combined_score"] = candidate.combined_score
        return metadata

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_profile: bool = False,
        rerank: bool = True,
        retrieve_k: int | None = None,
        graph_augment: bool = False,
        graph_initial_top_n: int | None = None,
        graph_expand_m: int | None = None,
        graph_final_top_k: int | None = None,
        rank_fusion: str | None = None,
        return_metadata: bool = True,
    ) -> list[SearchResult] | tuple[list[SearchResult], dict]:
        """Retrieve the top `top_k` results for `query`.

        Graph augmentation is opt-in. With `graph_augment=False`, this method
        follows the existing embedding/reranker code path. With
        `graph_augment=True`, standard candidates seed query-personalized
        PageRank over the configured declaration graph, graph-only candidates
        are merged into the pool, and the merged pool is reranked when the
        reranker is enabled.
        """
        profile: dict = {}
        rerank_local = getattr(self, "_rerank_profile_local", None)
        if rerank_local is None:
            rerank_local = threading.local()
            self._rerank_profile_local = rerank_local
        rerank_local.events = []
        graph_cfg = self._graph_request_config(
            top_k=top_k,
            graph_augment=graph_augment,
            graph_initial_top_n=graph_initial_top_n,
            graph_expand_m=graph_expand_m,
            graph_final_top_k=graph_final_top_k,
            rank_fusion=rank_fusion,
        )

        t0 = time.time()
        q = f"Instruction: {self.INSTRUCTION}. Query: {query}"
        emb = self.embedding_model.encode([q], convert_to_numpy=True)
        emb = (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype("float32")
        profile["embedding_time"] = time.time() - t0

        if retrieve_k is None:
            retrieve_k = max(top_k * 2, 50) if rerank else top_k
        if graph_cfg.graph_augment:
            retrieve_k = max(retrieve_k, graph_cfg.initial_top_n)

        t0 = time.time()
        distances, indices = self._search_cuvs(emb, retrieve_k)
        candidates = [(int(i), float(d)) for i, d in zip(indices[0], distances[0]) if i != -1]
        profile["retrieval_time"] = time.time() - t0
        profile["retrieval_backend"] = "cuvs"
        profile["num_candidates"] = len(candidates)
        profile["reranked"] = bool(rerank)
        profile["graph_augmented"] = bool(graph_cfg.graph_augment)
        profile["rank_fusion"] = graph_cfg.rank_fusion
        profile["graph_expand_m"] = graph_cfg.graph_expand_m
        profile["candidate_top_k_before_rerank"] = retrieve_k
        profile["embedding_dtype"] = getattr(self, "embedding_dtype", None) or "model_default"
        profile["max_rerank_batch_size"] = getattr(
            self, "max_rerank_batch_size", None
        )
        profile["reranker_required"] = getattr(self, "reranker_required", False)

        if not candidates:
            if return_profile:
                profile["rerank_time"] = 0.0
                profile["total_time"] = profile["embedding_time"] + profile["retrieval_time"]
                return [], profile
            return []

        if not rerank:
            t0 = time.time()
            if graph_cfg.graph_augment:
                n = len(candidates)
                standard = [
                    CandidateScore(index=i, standard_score=(n - rank) / n, distance=d)
                    for rank, (i, d) in enumerate(candidates)
                ]
                augmented = self._augment_with_graph(standard, graph_cfg)
                results = [
                    self._build_result(c.index, c.distance, self._graph_metadata(c) if return_metadata else None)
                    for c in augmented
                ]
                profile["graph_candidates"] = len(augmented)
                profile["candidate_union_size"] = len(augmented)
            else:
                results = [self._build_result(i, d) for i, d in candidates[:top_k]]
            profile["postprocess_time"] = time.time() - t0
            profile["rerank_time"] = 0.0
            profile["total_time"] = (
                profile["embedding_time"] + profile["retrieval_time"] + profile["postprocess_time"]
            )
            if return_profile:
                return results, profile
            return results

        dist_map = {i: d for i, d in candidates}
        candidate_indices = [i for i, _ in candidates]
        n = len(candidate_indices)
        retriever_rank_scores = [(n - i) / n for i in range(n)]

        t0 = time.time()
        pairs = [self._format_pair(query, self.texts[i]) for i in candidate_indices]
        profile["prompt_build_time"] = time.time() - t0

        t0 = time.time()
        scores = self._rerank(pairs, fallback_scores=retriever_rank_scores)
        profile["rerank_time"] = time.time() - t0

        t0 = time.time()
        ranked = sorted(zip(candidate_indices, scores), key=lambda x: -x[1])
        if graph_cfg.graph_augment:
            standard = [
                CandidateScore(index=i, standard_score=float(score), distance=dist_map[i])
                for i, score in ranked
            ]
            augmented = self._augment_with_graph(standard, graph_cfg)
            graph_pairs = [self._format_pair(query, self.texts[c.index]) for c in augmented]
            fallback_scores = [c.combined_score or c.standard_score for c in augmented]
            graph_scores = self._rerank(graph_pairs, fallback_scores=fallback_scores)
            rows = [
                {
                    "_position": position,
                    "candidate": candidate,
                    "qwen_score": float(graph_score),
                    "graph_score": float(candidate.graph_score or 0.0),
                }
                for position, (candidate, graph_score) in enumerate(zip(augmented, graph_scores))
            ]
            if graph_cfg.rank_fusion == "qwen_ppr_zscore":
                fused_rows = apply_qwen_ppr_zscore_fusion(rows, weight=graph_cfg.ppr_zscore_weight)
            else:
                fused_rows = sorted(rows, key=lambda row: (-float(row["qwen_score"]), int(row["_position"])))
            results = []
            for row in fused_rows[: graph_cfg.final_top_k or top_k]:
                candidate = row["candidate"]
                metadata = self._graph_metadata(candidate) if return_metadata else None
                if metadata is not None:
                    metadata["graph_rerank_score"] = float(row["qwen_score"])
                    metadata["rank_fusion"] = graph_cfg.rank_fusion
                    if "fusion_score" in row:
                        metadata["fusion_score"] = float(row["fusion_score"])
                    if "ppr_score_zscore_within_query" in row:
                        metadata["ppr_score_zscore_within_query"] = float(row["ppr_score_zscore_within_query"])
                results.append(self._build_result(candidate.index, candidate.distance, metadata))
            profile["graph_candidates"] = len(augmented)
            profile["candidate_union_size"] = len(augmented)
        else:
            results = [self._build_result(i, dist_map[i]) for i, _ in ranked[:top_k]]
        profile["postprocess_time"] = time.time() - t0

        rerank_events = list(getattr(self._rerank_profile_local, "events", []))
        profile["rerank_attempted"] = bool(rerank_events)
        profile["rerank_succeeded"] = bool(rerank_events) and all(
            event["succeeded"] for event in rerank_events
        )
        profile["rerank_fallback_used"] = any(
            event["fallback_used"] for event in rerank_events
        )
        profile["rerank_batches"] = sum(event["batches"] for event in rerank_events)

        profile["total_time"] = (
            profile["embedding_time"]
            + profile["retrieval_time"]
            + profile["prompt_build_time"]
            + profile["rerank_time"]
            + profile["postprocess_time"]
        )

        if return_profile:
            return results, profile
        return results

    def _rerank(
        self,
        pairs: list[str],
        fallback_scores: list[float],
    ) -> list[float]:
        """Score every (query, doc) pair across the reranker replicas.

        Pairs are split into bounded microbatches, assigned round-robin to
        reranker replicas, and processed serially per device.  When
        ``reranker_required`` is true, any unavailable replica/OOM/model error
        propagates instead of silently producing a non-reranked benchmark row.
        """
        if not pairs:
            return []

        n = len(self.reranker_models)
        if n < 1 or len(self.reranker_devices) < n:
            error = RuntimeError("reranker is unavailable: no model replica is loaded")
            if self.reranker_required:
                raise error
            events = getattr(self._rerank_profile_local, "events", None)
            if events is not None:
                events.append({"succeeded": False, "fallback_used": True, "batches": 0})
            print(f"Warning: {error}; falling back to retriever order")
            return list(fallback_scores)

        batch_size = getattr(self, "max_rerank_batch_size", None) or max(
            1, (len(pairs) + n - 1) // n
        )
        batches = [
            (start, pairs[start : start + batch_size])
            for start in range(0, len(pairs), batch_size)
        ]
        assignments: list[list[tuple[int, list[str]]]] = [[] for _ in range(n)]
        for batch_index, batch in enumerate(batches):
            assignments[batch_index % n].append(batch)

        def _device_job(
            idx: int,
            payloads: list[tuple[int, list[str]]],
        ) -> list[tuple[int, list[float]]]:
            completed: list[tuple[int, list[float]]] = []
            for start, payload in payloads:
                inputs = self._process_inputs(payload, self.reranker_devices[idx])
                completed.append(
                    (start, self._compute_scores(inputs, self.reranker_models[idx]))
                )
            return completed

        try:
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = [
                    ex.submit(_device_job, idx, payloads)
                    for idx, payloads in enumerate(assignments)
                    if payloads
                ]
                completed = [item for future in futs for item in future.result()]
            completed.sort(key=lambda item: item[0])
            scores = [score for _, chunk_scores in completed for score in chunk_scores]
            if len(scores) != len(pairs):
                raise RuntimeError(
                    f"reranker returned {len(scores)} scores for {len(pairs)} pairs"
                )
            events = getattr(self._rerank_profile_local, "events", None)
            if events is not None:
                events.append(
                    {"succeeded": True, "fallback_used": False, "batches": len(batches)}
                )
            return scores
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.reranker_required:
                raise RuntimeError(f"required reranker failed: {e}") from e
            events = getattr(self._rerank_profile_local, "events", None)
            if events is not None:
                events.append(
                    {"succeeded": False, "fallback_used": True, "batches": len(batches)}
                )
            print(f"Warning: rerank failed ({e}); falling back to retriever order")
            return list(fallback_scores)
