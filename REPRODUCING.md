# Reproduction guide

Run all commands from the extracted package root. See `PAPER_SETTINGS.md` for
parameter definitions and the MathlibQR-specific deep-ranking interpretation.

## 1. Install

Use Linux, Python 3.11, a CUDA 12-compatible GPU environment, and Lean's
`elan`/`lake`. For full experiments:

```bash
python -m pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com
python -m pip install -e . --no-deps
# Needed for the Claude Sonnet 5 MathlibMPR configurations:
python -m pip install 'anthropic[vertex]>=0.75'
export PYTHONPATH="$PWD/src"
```

The pinned file records direct dependencies, not a complete OS/transitive lock.
Save `python -m pip freeze` and `nvidia-smi` with each new experiment. For CPU
checks only, use `requirements-test.txt` and the commands in `README.md`.

## 2. Prepare assets and graph

```bash
python scripts/prepare_dependencies.py             # list public dependencies
python scripts/prepare_dependencies.py --download  # explicitly download many GB
(cd external/Mathlib4 && lake exe cache get)
(cd external/jixia && lake build)
python -m leansearchv2.corpus.jixia_extract
python -m leansearchv2.corpus.merge_to_jsonl
python -m leansearchv2.graph.build \
  --records-source data/corpus/merged/mathlib_declarations.jsonl \
  --output-dir data/graph/dep_only \
  --dep-weight 1 --rev-dep-weight 0 --same-module-weight 0 \
  --co-premise-weight 0 --dependency-source jixia_merged_jsonl
```

The download plan in `assets.lock.json` includes two Qwen model checkpoints,
the public cuVS corpus/index (including declaration text), Mathlib, and Jixia. Downloading
supports `--only embedder reranker cuvs` or `--only mathlib jixia`.
Public model/data revisions are pinned when preparing this supplement; the
original experiment model snapshot hashes were not recorded in the configs.

Expected local assets:

```text
models/Qwen3-Embedding-8B/
models/Qwen3-Reranker-8B/
data/cuvs/mathlib-v4.28.0-rc1/{metadata.pkl,texts.pkl,cuvs_index.bin}
data/graph/dep_only/{nodes.json,edges.json,edge_stats.json,graph_adjacency.npz}
```

The index metadata is not a substitute for Jixia's structured dependency
references. Graph rebuilding uses the pinned Mathlib/Jixia source, no LLM and no
embedding recomputation. Check structured edge counts in `edge_stats.json`.
The paper describes 310,579 nodes and 4,528,330 forward edges; a differing build
must be investigated before treating it as the same graph. Mathlib caches and
the first LeanInteract invocation may require additional downloads.

## 3. Start the shared retrieval service

```bash
HOST=127.0.0.1 PORT=8000 bash scripts/serve.sh
```

The default uses one visible GPU. The source deployment recommends 80 GB GPU
memory for both 8B models; actual requirements depend on batching and lengths.
Choose GPUs with `CUDA_VISIBLE_DEVICES`; adjust `serve` settings for available
hardware. Keep the same server and assets for each paired comparison.

`config.yaml` configures this service; the presets below configure clients.
`CONFIG_PATH`, `LEANSEARCH_CONFIG`, and `LEANSEARCHV2_CONFIG` select a base YAML
in that precedence order. Root `config.local.yaml`, if present, is merged over
it. Avoid unintended local overrides when comparing presets. Environment
variables override only settings explicitly supported by the code.

Reasoning/proving uses reviewer-supplied Vertex credentials. Set
`GOOGLE_CLOUD_PROJECT` and Application Default Credentials, or point
`GOOGLE_APPLICATION_CREDENTIALS` to your credential file outside the package.
The presets preserve the source's provider model identifiers; access to those
models is required. The archive contains no credentials or project IDs.

## 4. MathlibQR

```bash
python scripts/reproduce_mathlibqr.py --limit 5 --output outputs/mathlibqr-smoke
python scripts/reproduce_mathlibqr.py --output outputs/mathlibqr
```

This evaluates matched standard/graph retrieval on the same fixed fair subset,
with no LLM-generated queries. Outputs are `manifest.json`, `per_query.jsonl`,
and `summary.json`. A failed request is retained as an error and prevents an
aggregate score; it is not silently counted as a retrieval miss. Use a new
output directory for each run. The full evaluation contains 810 paired queries.

## 5. MathlibMPR: Gemini and Claude

```bash
for backbone in gemini claude; do
  for mode in standard graph; do
    LEANSEARCH_CONFIG="configs/reasoning_${backbone}_${mode}.yaml" \
    python scripts/reproduce_premise_costed.py \
      --url http://127.0.0.1:8000 \
      --output "outputs/mathlibmpr/${backbone}_${mode}" \
      --limit 69 --parallelism 1 --branch-budget 2 \
      --search-top-k 50 --filter-max-docs 50 --big-loop 3
  done
 done
```

Use `--limit 1` for a preliminary check. Each model evaluates all 69 problems.
The runner writes per-problem records, metrics, and a manifest. Report group
Recall@k and Covered@k at 5, 10, 20, 30, 50; average corresponding summary
metrics over the two backbone runs to obtain the paper's two-model aggregate.
The concurrency of one above reduces service contention without changing the
per-problem branch/document budgets.

## 6. FATE-H, MathlibMPR-Prop, and Lean-IMO-Bench

All use Gemini 3.1 Pro, the same five conditions, and optional failure memory.
For a one-problem check:

```bash
LEANSEARCH_CONFIG=configs/proving.yaml python scripts/reproduce_proving.py \
  --benchmark mathlibmpr_prop --modes graph --limit 1 --parallelism 1 \
  --output outputs/proving-smoke --strict-integrity \
  --max-model-calls 63 --max-query-calls 31 --max-compiler-calls 32
```

Complete grid (Bash):

```bash
for benchmark in fate_h mathlibmpr_prop lean_imo_basic lean_imo_advanced; do
  for memory in off on; do
    memory_args=()
    if [ "$memory" = on ]; then memory_args=(--failure-memory); fi
    LEANSEARCH_CONFIG=configs/proving.yaml python scripts/reproduce_proving.py \
      --benchmark "$benchmark" --parallelism 1 \
      --output "outputs/proving/${benchmark}_${memory}" --strict-integrity \
      --max-model-calls 63 --max-query-calls 31 --max-compiler-calls 32 \
      "${memory_args[@]}"
    python scripts/analyze_proving.py --benchmark "$benchmark" \
      --input "outputs/proving/${benchmark}_${memory}" \
      --output "outputs/proving/${benchmark}_${memory}/analysis.json" \
      --max-model-calls 63 --max-query-calls 31 --max-compiler-calls 32
  done
 done
```

Default conditions are `none`, `standard`, `graph`, `reasoning_adaptive`, and
`graph_reasoning_adaptive`. Initial reasoning preparation has a separate budget.
The CLI controls action budgets and `--failure-memory` activates memory;
`--strict-integrity` enables the paper's immutable proof-hole contract.
The verifier uses Mathlib v4.28.0-rc1 and a 600-second timeout.

Inspect analyzer completeness and integrity checks before using a result. Keep
failed/partial runs and usage events. Price constants are inherited accounting
assumptions, not current price quotations; preserve raw token counts when
recomputing monetary costs. Relative cost is normalized within the same
benchmark/memory group by no retrieval. API cost excludes local retrieval and
Lean-verification hardware.

## 7. What was verified

CPU tests, import/CLI checks, the synthetic graph smoke check, and privacy/file
inventory checks are the preparation-time validations. Full GPU retrieval,
Lean graph reconstruction, and paid model experiments are separate operations
and were not rerun for this package. Record exact commands, configurations,
asset hashes, model IDs, environment, seeds, failures, and usage when rerunning.
