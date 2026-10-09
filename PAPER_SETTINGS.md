# Paper settings

Settings are taken from the final manuscript, especially Appendices B and D.
The table distinguishes benchmark size from per-query retrieval budgets.

| Component | Distributed setting | Manuscript |
|---|---|---|
| Mathlib / Lean | v4.28.0-rc1; pinned public source revisions | B.1, B.4 |
| Embedding / reranking | Qwen3-Embedding-8B / Qwen3-Reranker-8B | B.1 |
| Reranker input cap | 8,192 tokens | B.1 |
| Dependency graph | forward structured references, unit edge weight | B.1 |
| PPR | restart 0.15; 50 iterations; tolerance 1e-6 | B.1 |
| Graph retrieval | 50 seeds; 200 expansion candidates; normally 100 retained for reranking | B.1 |
| Prefilter | alpha=1, beta=1, gamma=0.05 | B.1 |
| Final graph score | reranker score + 0.005 * per-query PPR z-score | B.1 |
| MathlibQR | 810 fair queries; 200 dense candidates; retain up to 200 ranked entries | B.3 |
| MathlibMPR | all 69 problems; two branches; initial plan + at most 3 revisions | B.3, D.1 |
| MPR subquery / filter | at most 50 / 50 declarations; at most 100 in aggregated output | B.3, D.2 |
| MPR judge | at most 30 unique filtered declarations | B.3 |
| Gemini 3.1 Pro | HIGH thinking; top_p=1; max output 65,536; no Google Search | D.2 |
| Claude Sonnet 5 | adaptive thinking; high effort; max output 128,000; no supplied sampling seed | D.2 |
| Proving | one initial candidate + at most 31 compiler-tested revisions | D.2 |
| Proving budgets | at most 32 compiler submissions / 31 retrieval actions / 63 model calls | D.2 |
| Reflection retrieval | five queries per action, at most 50 results per query | B.2, D.2 |
| Reasoning preparation | separate from the proof-stage model-call budget | D.2 |
| Failure memory | on/off; repeat threshold 2; at most 3 recurrent-state summaries | B.4 |
| Proving cutoffs | round index <= 8, <= 16, <= 32, with initial submission at round 0 | D.3 |

## Explicit interpretation of manuscript details

- MathlibQR's specific deep-ranking protocol in B.3 is used as the exception to
  the general top-50 output statement in B.1/D.2. Its runner requests 200 dense
  candidates and up to 200 retained/ranked candidates while using 50 PPR seeds
  and 200 expansion candidates. The normal internal pool of 100 is overridden
  for this benchmark so ranks beyond 50 can be assessed.
- B.4 describes temperature-zero requests at the controller interface. D.2
  describes provider-default temperatures. The existing model adapter omits
  those controller temperature arguments for the Gemini/Claude providers, so
  the outgoing request follows D.2. The inherited Gemini seed is recorded in
  its YAML; Claude does not send a seed.
- Some historical MathlibMPR working configurations used 30 retrieved and 30
  filtered documents. This supplement uses the final manuscript's 50/50
  setting. The earlier standalone MathlibQR script also had different default
  budgets and a separate ranking implementation; the supplied entry point uses
  the shared backend with the manuscript settings above. Consequently, this
  package does not claim that the new entry points reproduce every historical
  numeric result without rerunning the evaluations.

The retired 32-compiler/8-query proving protocol, fixed-context proving CLI
conditions, MiniF2F experiments, alternative graph sweeps, and historical report
renderers are not supplied. No saved scores are included.
