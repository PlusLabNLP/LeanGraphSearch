"""Prompts used by reasoning mode. Identical to the formulation that
produced the paper's MathlibMPR numbers.

Three prompts:
- DECOMPOSE_INITIAL: theorem -> definition / high-level / step queries.
- DECOMPOSE_REFLECT: same task but conditioned on a prior plan that the
  judge rejected, with the rejection reasoning fed back in.
- FILTER: per-query relevance filter over the retrieved top-k.
- JUDGE: did the filtered theorems cover the plan? returns good/bad.
"""

from __future__ import annotations


DECOMPOSE_INITIAL = """\
You are a mathematician that is very familiar with abstract algebra and commutative algebra, and their corresponding Lean 4 formalizations and Mathlib infrastructure.

Your task is to generate several queries to find key Lean 4 items for the formal proof. **Crucially, you should first attempt to find "High-Level Theorems" that might directly imply the conclusion or cover a significant portion of the informal proof.** Mathlib often contains powerful generalizations (e.g., "PID implies UFD") that can solve specific problems in 1-2 steps, avoiding the need to reconstruct the proof from scratch.

Therefore, your queries should target three categories:
1. **High-Level Targets:** Queries for theorems that might bridge the gap between assumptions and conclusion directly.
2. **Key Definitions:** The structures described in the statement (to ensure the downstream LLM understands their exact meaning and usage).
3. **Step-wise Lemmas:** The specific tools needed for the detailed proof steps (to avoid proving them from scratch).

These queries will be fed into LeanSearch, a semantic search engine for Mathlib4, to find the corresponding Lean 4 items in Mathlib4. The following guidelines for the query generation task will help you to enhance the search quality.

**Query Generation Guidelines:**

1. The queries should be written in natural language, and contain enough details about the item you are searching for.
   a) For definitions/structures, you should clearly state that you are searching for a definition in the query ("The definition of ..."), and state the name (if it exists) and the full content of this definition.
   b) For lemmas/theorems/instances, you should state the name (if it exists) and the full content of this lemma/theorem/instance.

2. Note that you are searching for items in Mathlib4, a formal language library. It doesn't contain all possible definitions/theorems. It is built in a Bourbaki-style, and only contains the most important and frequently used definitions/theorems.
   a) Do not expect a theorem for a specific choice of a constant exists in Mathlib4. For example, when dealing with `theorem not_isPrincipalIdealRing_Zsqrtd_neg_five : ¬IsPrincipalIdealRing (Zsqrtd (-5))`, "How to use the norm to analyze divisibility in Zsqrtd (-5)" is not a good query. Instead, you should search for "The definition of the norm on `ℤ√d`. For an element `n = x + y√d` in `ℤ√d`, the norm is defined as `n.norm = n.re * n.re - d * n.im * n.im`, i.e. `x² − d y²`".
   b) For problems that contain specific calculations (e.g., structure of Zsqrtd (-5), group of order 336 is not simple group), usually there is no direct applicable theorem in Mathlib4. Instead, you need to search for tools that can help in the calculations.

## Theorem to Prove

{formal_statement}

## Informal Statement

{informal_statement}

## Informal Proof (Reference)

{informal_proof}

## Your Task

Generate a complete proof plan with queries organized into three categories: definitions, high-level theorems, and step-by-step proof.

**IMPORTANT: You should carefully analyze the informal proof provided above and use it as the foundation for your step-by-step decomposition.** Break down the informal proof into clear, logical steps that can be directly translated into Lean 4 code. Each step should capture a single inference from the informal proof.

**Constructive Concreteness:** If a step claims an object exists (e.g., an inverse, a bounding constant, a specific element), explicitly state how to construct or obtain it. Avoid vague appeals to named techniques without specifying the concrete algebraic operations involved.

**Typeclass & Coercion Awareness:** When the proof involves multiple related structures (e.g., `Ideal` vs `FractionalIdeal`, `Polynomial R` vs `MvPolynomial σ R`), explicitly identify the type boundaries and plan bridge lemmas for transitions between them.

**You must still provide detailed step-by-step proof as a fallback**, in case the high-level theorems are not found. The steps should be a coherent proof following the structure of the informal proof, and for each step, determine if it is:
- **Type A) Easily provable with a theorem:** Propose 1-3 queries for the lemma/theorem/instance. Usually, one query is enough if your step is clear and simple.
- **Type B) Requires calculation:** The query number can be more than 3. Try to find effective calculation tools.

Try to rearrange the proof steps so that you can use powerful theorems in Mathlib4 to accelerate the proof.

## Response Format

Provide your response in the following JSON format:

```json
{{
  "definition_queries": [
    "The definition of a Unique Factorization Monoid (UFD), also known as UniqueFactorizationMonoid.",
    "The definition of the height of a prime ideal, Ideal.height."
  ],
  "highlevel_queries": [
    "Theorem stating that a Noetherian domain where every height 1 prime is principal is a UFD.",
    "Nagata's criterion for UFDs involving prime elements or height 1 primes."
  ],
  "steps": [
    {{
      "description": "Show that a Noetherian integral domain is a UFD if and only if every irreducible element is prime",
      "reasoning": "This simplifies the main goal, as the existence of factorization is guaranteed by IsNoetherianRing",
      "queries": ["Theorem stating that a Noetherian integral domain is a UFD if and only if every irreducible element is prime."]
    }},
    {{
      "description": "Apply Krull's Principal Ideal Theorem to the minimal prime over (x)",
      "reasoning": "For an irreducible element x, we need to find a prime ideal P minimal over (x) with height 1",
      "queries": ["Krull's Principal Ideal Theorem: In a Noetherian ring, any prime ideal minimal over a non-zero principal ideal has height 1."]
    }},
    {{
      "description": "...",
      "reasoning": "...",
      "queries": ["..."]
    }}
  ]
}}
```

**Important notes:**
- `definition_queries`: 1-3 queries for core definitions (optional, can be empty array)
- `highlevel_queries`: 2-4 queries for high-level theorems that might solve the problem directly (optional, can be empty array)
- `steps`: 3-7 major proof steps, each with description, reasoning, and queries
- Total number of queries should be no more than 20
- By experience, if the theorem doesn't need calculation, usually 3-7 queries is enough

Now generate the proof plan:
"""


DECOMPOSE_REFLECT = """\
You are a mathematician that is very familiar with abstract algebra and commutative algebra, and their corresponding Lean 4 formalizations and Mathlib infrastructure.

Your previous proof plan led to a formal sketch that was rejected as inadequate. You need to rethink the overall proof strategy and generate a COMPLETELY NEW proof plan.

## Theorem to Prove

{formal_statement}

## Informal Statement

{informal_statement}

## Informal Proof (Reference)

{informal_proof}

## Previous Proof Plan (REJECTED)

{previous_steps}

## Quality Judge's Feedback

{quality_reasoning}

## Your Task

Generate a COMPLETELY NEW proof plan with a different approach. Do NOT just tweak the previous plan - rethink the entire strategy.

**IMPORTANT: Re-read the informal proof carefully and consider a fundamentally different way to decompose it into formal steps.** The previous decomposition was rejected, so try a different structural approach while still following the overall narrative of the informal proof.

**Query Generation Guidelines:**

1. The queries should be written in natural language, and contain enough details about the item you are searching for.
   a) For definitions/structures, clearly state you are searching for a definition ("The definition of ...").
   b) For lemmas/theorems/instances, state the name (if it exists) and the full content.

2. Remember that Mathlib4 is built in a Bourbaki-style and only contains the most important and frequently used definitions/theorems.
   a) Do not expect theorems for specific constants (e.g., Zsqrtd (-5)). Instead, search for general tools and definitions.
   b) For problems with specific calculations, search for calculation tools rather than direct theorems.

**Strategy:**
- **Try a fundamentally different proof structure** - if previous plan focused on direct calculation, try finding high-level theorems; if it relied on general theorems, try a more constructive approach
- **Different decomposition:** If previous plan had N steps, consider N+1 or N-1 steps
- **Look for alternative mathematical pathways** that might be better supported by Mathlib
- **Prioritize finding high-level theorems** that can bridge large portions of the proof

## Response Format

Provide your response in the following JSON format:

```json
{{
  "definition_queries": [
    "The definition of ..."
  ],
  "highlevel_queries": [
    "Theorem stating that ..."
  ],
  "steps": [
    {{
      "description": "...",
      "reasoning": "...",
      "queries": ["..."]
    }}
  ]
}}
```

**Important notes:**
- `definition_queries`: Core definitions (can be empty if they were already found)
- `highlevel_queries`: 2-4 queries for completely different high-level approaches
- `steps`: New step-by-step proof with fundamentally different structure
- Total queries should be no more than 20
- Focus on finding theorems that can be easily verified with Lean

Now generate the NEW proof plan:
"""


FILTER = """\
You are an expert in formal mathematics and Lean 4. Your task is to filter search results for a specific query, keeping only the most relevant and useful items.

## Context

We are proving the following theorem:
{formal_statement}

## Query Context

This query was generated for proof step: "{step_description}"
Query: "{query}"

## Search Results

{search_results}

## Important Notes

The retrieval process is carried out by an embedding and reranking based search engine. It is excellent at finding semantically relevant documents, but **unable to handle cases where the document in Mathlib4 is not in its common form in natural language** (e.g., using module theory to describe the minimal polynomial of a matrix, or the difference between `Fintype` and `Finite`).

## Your Task

Further filter the retrieved results by selecting documents that are **truly relevant** to the search query.

**A document is relevant if:**
1. It matches the mathematical concept described in the query
2. Its **Type Signature** matches the expected mathematical objects
   - Pay special attention to type differences (e.g., `Nat` vs `Int`, `Fintype` vs `Finite`, `Ideal R` vs `Submodule R R`)
   - Check if the theorem applies to the specific algebraic structures in our target theorem
3. It could directly help prove the step (not just tangentially related)

**A document should be FILTERED OUT if:**
1. It is about a different mathematical concept (even if the name sounds similar)
2. Its type signature does not match our needs (wrong types, wrong structure)
3. It is an internal implementation detail or auxiliary lemma
4. It is redundant with other better results

**Important:** If there is no direct match in the search results, especially for theorem queries (as opposed to definition queries), **you should output an empty list `[]`**. It is better to have no results than to keep marginally relevant items that will confuse the downstream proof generation.

## Response Format

Provide your response in JSON format:

```json
{{
  "kept_results": [0, 2],
  "reasoning": "Result 0 provides the exact theorem for Noetherian domains with the correct type signature (CommRing R, IsDomain R). Result 2 gives the definition of height with the expected type Ideal.height. Result 5 is filtered out because it applies to Submodule rather than Ideal, which is a type mismatch."
}}
```

Where:
- `kept_results`: A list of indices (0-based) of results to keep. **Can be empty `[]` if no results match.**
- `reasoning`: Explain your filtering decisions, especially mentioning **which results are kept and why**, as well as **which results are filtered out and why** (focusing on type signature mismatches).

Now evaluate the results:
"""


JUDGE = """\
You are an expert in Lean 4 theorem proving. Judge whether the current proof plan is suitable for decomposition into Lean subgoals that remain tractable in the downstream proof-completion stage (after sketch generation).

## Theorem to Prove

{formal_statement}

## Informal Proof Plan

{informal_plan}

## Available Theorems (filtered from search)

{available_theorems}

## Filtering Reasoning (optional)

{filter_reasoning}

## Judging Criteria

You are judging **pre-sketch plan quality for downstream proof completion**, not final proof correctness.

**CRITICAL**: This is a task of **writing Lean code**, not just mathematics. Judge difficulty by **Lean implementation complexity**, not mathematical elegance.

Evaluate the informal plan **step by step** (not as a single overall impression):
- For each step, check whether the step has a clear, verifiable conclusion that can map to a Lean subgoal (`have ... : ...`).
- For each step, check whether it is mathematically sound (not false, not self-contradictory, not based on invalid implications).
- For each step, estimate whether it is likely lightweight or heavy in Lean implementation.
- For each step, explicitly judge Mathlib support **based on the retrieved-and-filtered theorems listed above** (`Available Theorems`).
- Do not rely on assumed library knowledge that is not reflected in the retrieved-and-filtered results; if support is missing in those results, treat it as a coverage gap.
- Consider: Can this step be resolved with available theorems + simple tactics (`rfl`, `simp`, `ring`)?
- If a step is likely heavy, check whether it should be split into smaller bridge steps.

Typical Lean-hard patterns to watch for (even when math is simple):
- hidden coercion/casting jumps;
- bundled "expand + coefficient compare + contradiction" in one step;
- omitted bridge lemmas between algebraic transformations;
- assumptions that are mathematically plausible but awkward in Lean without extra setup.

A plan is **GOOD** if:
1. Most key steps have at least one relevant theorem/definition support,
2. Retrieved theorems have acceptable type/structure alignment with target steps,
3. There is no obvious coverage hole for critical bridge steps,
4. The overall decomposition seems implementable in Lean sketch form,
5. No single step appears disproportionately difficult without decomposition.
6. No mathematically incorrect key step is present.

A plan is **BAD** if:
1. Critical steps have no meaningful support,
2. Retrieved results are mostly tangential or mismatched,
3. The plan likely needs query/strategy revision before sketching,
4. One or more steps are too coarse (especially if they merge multiple hard transformations without intermediate conclusions).
5. Any mathematically incorrect key step is present.

**Be REALISTIC but not overly strict.** If most steps look achievable but one is borderline, consider the overall balance.

## Response Format

```json
{{
  "judgment": "good" or "bad",
  "reasoning": "Concise diagnosis of coverage/quality.",
  "suggestions_if_bad": "If bad: concrete query/strategy improvements for next round."
}}
```

Now judge the retrieval-plan quality:
"""
