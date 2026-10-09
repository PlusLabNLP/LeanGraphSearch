# LeanGraphSearch

Anonymous supplementary code for **LeanGraphSearch for Formal Theorem Proving**.

This package contains only the shared retrieval/proving implementation and the
experiments covered by the paper:

| Experiment | Inputs | Model |
|---|---:|---|
| MathlibQR retrieval | 810 queries / 171 shared declarations | Qwen3 embedding and reranking |
| MathlibMPR reasoning retrieval | 69 problems | Gemini 3.1 Pro; Claude Sonnet 5 |
| FATE-H proving | 100 problems | Gemini 3.1 Pro |
| MathlibMPR-Prop proving | 50 problems | Gemini 3.1 Pro |
| Lean-IMO-Bench proving | 30 Basic + 30 Advanced problems | Gemini 3.1 Pro |

All three evaluation entry points use the same local retrieval backend.
Graph augmentation is selected per request; proving additionally compares no
retrieval and failure memory on/off. The `leansearchv2` import name is retained
for compatibility with the upstream baseline.

## Quick software check

From the extracted `LeanGraphSearch/` directory, using Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
export PYTHONPATH="$PWD/src"
python scripts/smoke_graph.py
python -m pytest tests -q -p no:cacheprovider
```

These checks require no GPU or credentials and make no model-service calls.
Full experiments require the external models/index, Lean toolchain, GPU, and
model access described in [REPRODUCING.md](REPRODUCING.md).

- [REPRODUCING.md](REPRODUCING.md): installation, asset preparation, and commands.
- [PAPER_SETTINGS.md](PAPER_SETTINGS.md): settings mapped to the final manuscript.
- [GUIDE.zh-CN.md](GUIDE.zh-CN.md): concise Chinese guide.
- [CONTENTS.md](CONTENTS.md): complete file inventory.

## Layout

```text
src/leansearchv2/   shared implementation, prompts, metrics, and Lean verification
scripts/           three evaluation entry points, proving analysis, setup and smoke check
configs/           four MathlibMPR presets and one proving preset
benchmark/         small benchmark inputs and third-party data notices
config.yaml        shared GPU retrieval service configuration
assets.lock.json   public external dependency revisions and destinations
tests/             offline checks for retained functionality
```

Original experiment outputs, generated proofs, logs, paper drafts, Git history,
private configurations, deployment tooling, optimization sweeps, the MiniF2F
evaluation framework, and model/index/graph binaries are excluded. A few shared
model and diagnostic components have been extracted from the larger original
framework so it is no longer an import dependency.

This is a self-contained source supplement with small benchmark inputs and
instructions, not an offline environment image: large public dependencies and
hosted model access must be prepared separately. No full model-based experiment
was rerun when preparing the supplement.

Upstream attribution and license terms are preserved in [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [LICENSE](LICENSE), and
[LICENSE-MIT](LICENSE-MIT).
