# Complete package contents

This source package contains **99 files**, totaling **1,262,545 bytes** before ZIP compression.

The list below is exhaustive. `SHA256SUMS` covers every file except itself.
No symlinks, saved experiment outputs, credentials, Git metadata, model weights,
or generated indexes/graphs are included. Benchmark data are public task inputs.

Verify after extraction with `sha256sum -c SHA256SUMS` from the package root.

## Root files

| File | Bytes |
|---|---:|
| [.gitignore](.gitignore) | 163 |
| [CONTENTS.md](CONTENTS.md) | 8,069 |
| [GUIDE.zh-CN.md](GUIDE.zh-CN.md) | 1,746 |
| [LICENSE](LICENSE) | 11,357 |
| [LICENSE-MIT](LICENSE-MIT) | 1,079 |
| [NOTICE](NOTICE) | 921 |
| [PAPER_SETTINGS.md](PAPER_SETTINGS.md) | 3,374 |
| [README.md](README.md) | 3,058 |
| [REPRODUCING.md](REPRODUCING.md) | 7,572 |
| [SHA256SUMS](SHA256SUMS) | 9,412 |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 1,624 |
| [assets.lock.json](assets.lock.json) | 2,587 |
| [config.example.yaml](config.example.yaml) | 421 |
| [config.yaml](config.yaml) | 1,068 |
| [pyproject.toml](pyproject.toml) | 1,245 |
| [requirements-test.txt](requirements-test.txt) | 259 |
| [requirements.txt](requirements.txt) | 737 |

## benchmark

| File | Bytes |
|---|---:|
| [benchmark/FATE-H.LICENSE](benchmark/FATE-H.LICENSE) | 663 |
| [benchmark/FATE-H.jsonl](benchmark/FATE-H.jsonl) | 168,074 |
| [benchmark/LeanIMO.LICENSE](benchmark/LeanIMO.LICENSE) | 536 |
| [benchmark/LeanIMO.csv](benchmark/LeanIMO.csv) | 260,418 |
| [benchmark/MathlibMPR.json](benchmark/MathlibMPR.json) | 117,997 |
| [benchmark/MathlibMPR_Prop_ids.txt](benchmark/MathlibMPR_Prop_ids.txt) | 1,124 |
| [benchmark/MathlibQR.json](benchmark/MathlibQR.json) | 115,281 |
| [benchmark/MathlibQR_shared171.json](benchmark/MathlibQR_shared171.json) | 7,233 |

## configs

| File | Bytes |
|---|---:|
| [configs/proving.yaml](configs/proving.yaml) | 1,452 |
| [configs/reasoning_claude_graph.yaml](configs/reasoning_claude_graph.yaml) | 1,100 |
| [configs/reasoning_claude_standard.yaml](configs/reasoning_claude_standard.yaml) | 759 |
| [configs/reasoning_gemini_graph.yaml](configs/reasoning_gemini_graph.yaml) | 1,126 |
| [configs/reasoning_gemini_standard.yaml](configs/reasoning_gemini_standard.yaml) | 785 |

## scripts

| File | Bytes |
|---|---:|
| [scripts/analyze_proving.py](scripts/analyze_proving.py) | 22,163 |
| [scripts/prepare_dependencies.py](scripts/prepare_dependencies.py) | 2,575 |
| [scripts/reproduce_mathlibqr.py](scripts/reproduce_mathlibqr.py) | 4,298 |
| [scripts/reproduce_premise_costed.py](scripts/reproduce_premise_costed.py) | 22,351 |
| [scripts/reproduce_proving.py](scripts/reproduce_proving.py) | 32,828 |
| [scripts/serve.sh](scripts/serve.sh) | 714 |
| [scripts/smoke_graph.py](scripts/smoke_graph.py) | 1,787 |

## src

| File | Bytes |
|---|---:|
| [src/leansearchv2/__init__.py](src/leansearchv2/__init__.py) | 393 |
| [src/leansearchv2/config.py](src/leansearchv2/config.py) | 2,722 |
| [src/leansearchv2/corpus/__init__.py](src/leansearchv2/corpus/__init__.py) | 64 |
| [src/leansearchv2/corpus/jixia_extract.py](src/leansearchv2/corpus/jixia_extract.py) | 6,261 |
| [src/leansearchv2/corpus/merge_to_jsonl.py](src/leansearchv2/corpus/merge_to_jsonl.py) | 5,245 |
| [src/leansearchv2/eval/__init__.py](src/leansearchv2/eval/__init__.py) | 59 |
| [src/leansearchv2/eval/premise_metrics.py](src/leansearchv2/eval/premise_metrics.py) | 2,990 |
| [src/leansearchv2/eval/search_metrics.py](src/leansearchv2/eval/search_metrics.py) | 1,421 |
| [src/leansearchv2/gemini_client.py](src/leansearchv2/gemini_client.py) | 20,240 |
| [src/leansearchv2/graph/__init__.py](src/leansearchv2/graph/__init__.py) | 278 |
| [src/leansearchv2/graph/artifact.py](src/leansearchv2/graph/artifact.py) | 33,249 |
| [src/leansearchv2/graph/augment.py](src/leansearchv2/graph/augment.py) | 6,178 |
| [src/leansearchv2/graph/build.py](src/leansearchv2/graph/build.py) | 7,674 |
| [src/leansearchv2/graph/config.py](src/leansearchv2/graph/config.py) | 4,731 |
| [src/leansearchv2/graph/ppr.py](src/leansearchv2/graph/ppr.py) | 5,176 |
| [src/leansearchv2/graph/schema_inspection.py](src/leansearchv2/graph/schema_inspection.py) | 6,324 |
| [src/leansearchv2/imo_eval/__init__.py](src/leansearchv2/imo_eval/__init__.py) | 156 |
| [src/leansearchv2/imo_eval/benchmark.py](src/leansearchv2/imo_eval/benchmark.py) | 4,300 |
| [src/leansearchv2/imo_eval/diagnostics.py](src/leansearchv2/imo_eval/diagnostics.py) | 6,085 |
| [src/leansearchv2/imo_eval/failure_memory.py](src/leansearchv2/imo_eval/failure_memory.py) | 15,377 |
| [src/leansearchv2/imo_eval/failure_state.py](src/leansearchv2/imo_eval/failure_state.py) | 4,157 |
| [src/leansearchv2/imo_eval/problem.py](src/leansearchv2/imo_eval/problem.py) | 725 |
| [src/leansearchv2/imo_eval/source_utils.py](src/leansearchv2/imo_eval/source_utils.py) | 2,001 |
| [src/leansearchv2/llm.py](src/leansearchv2/llm.py) | 12,923 |
| [src/leansearchv2/pipeline.py](src/leansearchv2/pipeline.py) | 27,439 |
| [src/leansearchv2/prove/__init__.py](src/leansearchv2/prove/__init__.py) | 770 |
| [src/leansearchv2/prove/budget.py](src/leansearchv2/prove/budget.py) | 3,269 |
| [src/leansearchv2/prove/integrity.py](src/leansearchv2/prove/integrity.py) | 11,248 |
| [src/leansearchv2/prove/prompts.py](src/leansearchv2/prove/prompts.py) | 18,893 |
| [src/leansearchv2/prove/run.py](src/leansearchv2/prove/run.py) | 41,175 |
| [src/leansearchv2/prove/telemetry.py](src/leansearchv2/prove/telemetry.py) | 6,207 |
| [src/leansearchv2/prove/verifier.py](src/leansearchv2/prove/verifier.py) | 5,849 |
| [src/leansearchv2/reasoning/__init__.py](src/leansearchv2/reasoning/__init__.py) | 645 |
| [src/leansearchv2/reasoning/decompose.py](src/leansearchv2/reasoning/decompose.py) | 3,452 |
| [src/leansearchv2/reasoning/filter.py](src/leansearchv2/reasoning/filter.py) | 4,677 |
| [src/leansearchv2/reasoning/judge.py](src/leansearchv2/reasoning/judge.py) | 2,542 |
| [src/leansearchv2/reasoning/prompts.py](src/leansearchv2/reasoning/prompts.py) | 15,797 |
| [src/leansearchv2/reasoning/run.py](src/leansearchv2/reasoning/run.py) | 18,079 |
| [src/leansearchv2/runtime.py](src/leansearchv2/runtime.py) | 2,695 |
| [src/leansearchv2/server.py](src/leansearchv2/server.py) | 10,064 |
| [src/leansearchv2/standard_client.py](src/leansearchv2/standard_client.py) | 4,630 |

## tests

| File | Bytes |
|---|---:|
| [tests/__init__.py](tests/__init__.py) | 162 |
| [tests/test_gemini_vertex_web_search.py](tests/test_gemini_vertex_web_search.py) | 6,698 |
| [tests/test_graph_build_cli.py](tests/test_graph_build_cli.py) | 1,943 |
| [tests/test_graph_core.py](tests/test_graph_core.py) | 24,873 |
| [tests/test_llm_usage.py](tests/test_llm_usage.py) | 6,206 |
| [tests/test_paper_entrypoints.py](tests/test_paper_entrypoints.py) | 2,379 |
| [tests/test_pipeline_graph_contract.py](tests/test_pipeline_graph_contract.py) | 2,915 |
| [tests/test_pipeline_graph_off_behavior.py](tests/test_pipeline_graph_off_behavior.py) | 1,942 |
| [tests/test_pipeline_missing_graph.py](tests/test_pipeline_missing_graph.py) | 1,635 |
| [tests/test_premise_costed.py](tests/test_premise_costed.py) | 5,117 |
| [tests/test_prove_a2_budget.py](tests/test_prove_a2_budget.py) | 7,112 |
| [tests/test_prove_body_transport.py](tests/test_prove_body_transport.py) | 5,336 |
| [tests/test_prove_failure_memory.py](tests/test_prove_failure_memory.py) | 7,572 |
| [tests/test_prove_integrity.py](tests/test_prove_integrity.py) | 13,068 |
| [tests/test_prove_usage_telemetry.py](tests/test_prove_usage_telemetry.py) | 7,189 |
| [tests/test_proving.py](tests/test_proving.py) | 2,950 |
| [tests/test_reasoning_budget.py](tests/test_reasoning_budget.py) | 5,581 |
| [tests/test_reasoning_filter.py](tests/test_reasoning_filter.py) | 2,076 |
| [tests/test_search_metrics.py](tests/test_search_metrics.py) | 2,828 |
| [tests/test_standard_client_graph.py](tests/test_standard_client_graph.py) | 2,693 |
| [tests/test_standard_graph_integration.py](tests/test_standard_graph_integration.py) | 8,154 |

