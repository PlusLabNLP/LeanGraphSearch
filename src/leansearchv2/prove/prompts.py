"""Prove prompts: paper-compatible templates and revised strict templates.

Four prompts:
- PROVER_INIT: theorem + retrieved docs -> initial proof attempt.
- PROVER_REFLECT: error + retrieved docs -> revised proof attempt.
- GET_QUERY_INIT: theorem -> retrieval queries for the initial attempt.
- GET_QUERY_REFLECT: theorem + failed proof + errors -> retrieval queries
  for the next reflection round.
"""

from __future__ import annotations


PROVER_INIT = """\
You are an expert at using Lean for formal proofs. Your task is to complete the following Lean 4 code. You have access to the following relevant theorems and definitions found through search.

## Lean 4 Code to Complete

```lean4
{lean_code}
```
## Relevant Search Results

{search_results}

## Critical Guidance

- Pay special attention to the search results above, as they contain relevant lemmas and theorems that may be directly applicable to this proof. Consider how to incorporate these results into your proof strategy.

## Final Requirement on the Formal Statement

Note that the given formal statement has been verified by Lean 4 and professional mathematicians, so you SHOULD NOT CHANGE the formal statement. Instead, use the exact code in the formal statement as your prefix.

### Your Response Format

Before producing the Lean 4 code to formally prove the given theorem, provide:
1. **Detailed proof plan:** Outline the main proof steps and strategies. The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof. Analyze how the relevant search results can be utilized in your proof.
2. **Final Lean 4 Code:** Provide the complete Lean 4 code block that includes the formal statement and your proof.

Your final code **MUST** begin with the exact, character-for-character original formal statement and follow the format:
```lean4
{lean_code}
```
"""


PROVER_REFLECT = """\
You are an expert at using Lean for formal proofs. Your task is to revise the following Lean 4 code. You will be provided with an incorrect proof, its error messages, and search results that contain relevant theorems and definitions.

## Incorrect Proof

```lean4
{proof}
```

## Error Messages

{error_msg}

## Relevant Search Results

{search_results}

## Critical Guidance on using External Information

**When the Compiler Reports Errors:**
If the `Error messages` section contains errors like:
- `module '...' does not exist`: You are trying to import a module (.lean file) that is not available in the current Mathlib version. Please use the **most safe import** `import Mathlib` to replace all your incorrect customized imports, or just use the original `import` in the formal statement (`import Mathlib` for most cases).
- `unknown tactic '...'`: This means the tactic you are trying to use is not recognized by Lean 4, likely because it does not exist in the current library or version you are using.
- `unknown identifier '...'` (or `Unknown constant`, `invalid field`): This mostly indicates that the identifier (which could be a theorem, definition, or variable) is not defined or imported in your current context. Please refer to the search results to find an appropriate replacement. If you cannot find any proper replacement in the search results, it indicates that such identifier is non-existent in Mathlib, please do not guess and **prove it from scratch** by `have`, `obtain`, or `show`, `suffices` tactics using known theorems from search results.
- `no goals to be solved`: it's possible that the current line of Tactic is redundant and can be deleted.

**Your Action Plan (MANDATORY):**
- You **MUST NOT** attempt to use the same non-existent item again.
- You **MUST** treat the `Relevant search results` as your **sole source of truth**.
- Your entire revised strategy **MUST** be built using **only** the tools and theorems found in the search results.

## Final Requirement on the Formal Statement

This is the most important rule. The formal statement of the theorem is **ABSOLUTELY IMMUTABLE**. It is a fixed prefix for your code.
- **DO NOT** change variable names, reorder hypotheses, change types, or change definitions.
- **All errors are guaranteed to be in the proof, never in the statement itself.** Your task is to fix the proof *within the confines* of the given statement.

## Your Response Format

Before producing the final code, you MUST provide a step-by-step thinking process:
1.  **Error Analysis:** Briefly explain what the error messages mean.
2.  **Revised Strategy:** Outline your new proof plan. Explain how you will use the search results to fix the errors.
3.  **Final Code:** Provide the complete, corrected Lean 4 code block.

Your final code **MUST** begin with the exact, character-for-character original formal statement and follow the format:
```lean4
{lean_code}
```
"""


PROOF_INTEGRITY_RULES = """\
## Non-negotiable Proof-Integrity Rules

These rules are part of the task, not optional style guidance:

1. Prove the supplied target using its hypotheses and existing library results.
   Do not add an unproved assumption or use the theorem being defined as its
   own proof. Introducing a binder or hypothesis required by the goal, case
   analysis, induction, and proof by contradiction are legitimate proof steps.
2. Never use or introduce `axiom`, `constant`, `opaque`, `unsafe`, `unsound`,
   `sorry`, or `admit`. Do not hide these constructs in another declaration.
3. The host owns every top-level declaration. Generate only the tactic body;
   no new top-level declarations, imports, namespaces, options, or commands.
4. The supplied prefix, including the target theorem header ending in `:= by`,
   and the supplied suffix are immutable. The host inserts your generated
   tactic body between them; you must not copy, rewrite, or close either one.
5. Output only tactic commands that belong inside the existing `by` proof body,
   without another outer `by`. Local `have`, `let`, `letI`, `haveI`,
   `suffices`, and ordinary tactic steps are allowed. If you need a helper
   result, state and prove it locally with `have` or `suffices`, including a
   stronger or universally quantified helper when useful. Every helper needs
   a proof. The `classical` tactic and existing library automation are allowed.
6. Existing declarations from the supplied Mathlib environment are the only
   admissible external facts. Search results are hints to existing declarations,
   not assumptions and not authorization to invent similarly named facts.
7. Aim to close every goal. If unsuccessful, submit a concrete proof attempt
   for compiler feedback, without placeholders or trusted-code escape hatches.

The host checks source boundaries and forbidden constructs before compilation;
Lean checks the proof. A rejected candidate returns compiler-tool feedback to
the next reflection round and does not count as a successful proof.
"""


FAILURE_MEMORY_GUIDANCE = """\
## Failure Memory

The host maintains one problem-level memory of deterministic Lean failures in
this same proof trajectory. Compiler feedback may contain a structured
`failure_memory` summary. Do not repeat a listed tactic, lemma, declaration,
exact candidate, or proof modification in the same formal state. If the same
dead end appears twice, make a materially different local repair, rewrite the
proof, change the mathematical strategy, or use the available retrieval
evidence differently. The memory is advisory: it never selects a proof route
for you and does not authorize changing the theorem or violating proof
integrity.
"""


FAILURE_MEMORY_GUIDANCE_STRICT = """\
## Failure Memory

The `failure_memory` feedback records earlier failures in this trajectory.
Treat a listed tactic or declaration as evidence about its failed application,
not a global ban on that tool. Corrected arguments, goal shape, instances, or
local hypotheses can make a new application valid. An unchanged full candidate
with a cached failure is blocked; submit a revised candidate.

When a failure repeats, identify what has changed that could resolve it. Keep
useful proof steps; change the local application or choose another strategy
when the feedback shows a mathematical obstacle. A repeated error alone does
not establish that the whole strategy is impossible. Memory summaries guide
repairs; Lean checks the submitted proof.
"""


PROVER_INIT_STRICT = """\
You are an expert at writing sound Lean 4 proofs. The host will insert exactly
one generated tactic body into the proof hole below. Solve the stated theorem;
do not reproduce or edit the surrounding Lean source.

## Exact Immutable Prefix

```lean4
{proof_prefix}
```

This prefix already ends in `:= by`. It is context for understanding the goal,
not text to repeat in your response.

## Exact Immutable Suffix

```lean4
{proof_suffix}
```

This suffix is host-owned context and must not be repeated or closed by your
response.

## Relevant Search Results

{search_results}

Use the search results as hints to declarations that actually exist in the
supplied Mathlib environment. You may also use other existing Mathlib
declarations you know, but never invent a declaration or add an assumption.
Check a result's type, hypotheses, argument order, and conclusion before using
it. Search rank and informal similarity do not establish applicability.
Empty search results do not mean the needed fact is absent from Mathlib.
Any supplied reasoning sketch is a candidate route: adapt it if its steps
do not fit the formal target or available declarations.

## Proof Construction

Read the exact target and local hypotheses, then choose a proof that fits them.
Use a direct library application when applicable; otherwise reduce the goal
with explicit intermediate claims. Match coercions and typeclass assumptions
at each application. Use local `have` blocks for auxiliary results and ordinary
Lean tactics for the resulting goals.

{integrity_rules}

{failure_memory_guidance}

## Response Format

First give a concise proof plan. Then output exactly one Lean 4 code block that
contains only the honest tactic commands to place after the existing `:= by`.
Return the entire replacement body, including all branches and helper proofs.
Start outermost tactics at column zero; indent nested blocks consistently.
Do not include the theorem statement, imports, namespace commands, suffix, or
an additional outer `by`. Put every helper inside the tactic body with `have`,
`suffices`, or `let`.
"""


PROVER_REFLECT_STRICT = """\
You are an expert at repairing sound Lean 4 proofs. The host owns the immutable
source contract and will insert your revised tactic body into its existing
`:= by` proof hole.

## Exact Immutable Prefix

```lean4
{proof_prefix}
```

## Exact Immutable Suffix

```lean4
{proof_suffix}
```

## Previous Failed Tactic Body

```lean4
{proof}
```

## Compiler-Tool Feedback

{error_msg}

## Relevant Search Results

{search_results}

Use compiler feedback to choose the next concrete repair. Reported line and
column numbers refer to the complete assembled file (prefix, body, suffix),
not line numbers within the body shown above.

Prioritize an early syntax or scope error that could cause later errors.
For indentation or parser errors, repair the block structure first. For a
type mismatch, compare the expected and actual types, explicit arguments,
coercions, and required instances. For remaining goals, prove the displayed
obligation. For `no goals to be solved`, remove the redundant tactic.

For an unknown identifier, first determine whether it is a local name, a
namespace-resolution issue, or a library declaration. A failed name does not
establish that the mathematical fact is absent. Use an applicable known or
retrieved declaration, or prove the needed fact locally. Library facts outside
the search results remain available. Check retrieved statements against the
current goal and hypotheses; a search ranking is not evidence of applicability.

Preserve coherent parts of the proof while repairing the identified obstacle.
Change strategy if the needed step cannot be justified or the chosen approach
has stopped making progress. A supplied reasoning sketch is revisable.

{integrity_rules}

{failure_memory_guidance}

## Mandatory Repair of the Previous Attempt

If feedback reports a proof-integrity violation, remove the offending
construction. Prove any useful helper locally using `have` or `suffices`,
then use it to finish the original target.

## Response Format

First briefly identify the obstacle and the change that addresses it. Then output exactly
one Lean 4 code block containing only the repaired tactic commands to place
after the existing `:= by`. Return the entire replacement body, not a patch
or only the last failing step. Start outermost tactics at column zero and
indent nested blocks consistently. Do not repeat the prefix, suffix, theorem header,
imports, namespace commands, or an additional outer `by`.
"""


GET_QUERY_REFLECT_STRICT = """\
Generate exactly {num_queries} complementary Mathlib search queries for the
next proof attempt. These queries supply candidate declarations to a Lean
prover; they do not execute Lean code.

## Formal Target and Context
```lean4
{lean_code}
```

## Informal Description
{informal_statement}

## Previous Failed Tactic Body
```lean4
{proof}
```

## Compiler Feedback
{error_msg}

Find the mathematical obligation or missing API behind the current obstacle.
For a library-name error, describe the intended mathematical fact in natural
language, including relevant hypotheses and the type of objects. Check whether
the unknown name is instead a local variable, a namespace issue, or a result
of an earlier syntax error. A local-name or indentation error itself does not
require a missing library lemma; search for facts supporting the intended next
proof step in that case.

For a type mismatch, seek a version of the fact with the actual domains,
codomains, assumptions, and coercions. For an unsolved goal, describe that goal
and the hypotheses available to discharge it. Include a definition or bridge
lemma when it would connect the available facts to the goal.

Each query should express a distinct useful fact or alternative approach,
avoiding repeated paraphrases of one guessed name. Give natural-language descriptions; add
an identifier only when it helps disambiguate a known declaration. Explicitly
say when searching for a definition. Use only the supplied context, not an
assumed benchmark answer. A previous failed application does not make every
use of that lemma invalid.

Return exactly one JSON object in a json code block, with no other text:
{{"queries": ["natural-language query", "..."]}}
The queries array must contain exactly {num_queries} nonempty strings.
"""


GET_QUERY_INIT = """\
You are an expert in formal mathematics and Lean 4. Given a theorem statement and its proof context, your task is to generate {num_queries} precise search queries that would help find relevant lemmas, theorems, or definitions in a mathematical knowledge base.

**Theorem Statement:**
```lean4
{lean_code}
```

**Informal Description:**
{informal_statement}

Please first analyze the theorem and provide a natural language proof step by step, then generate exactly {num_queries} search queries that would be most helpful for proving this theorem. Avoid Proposing simple theorem names like "MonoidHom.map_one" alone. LeanSearch works best with natural language matching. Therefore, you should propose queries that include both the theorem name and a natural language description of its content.
Tips: when you need to search for definitions, you should clearly state that you are searching for a definition of something in the query, because LeanSearch are not sensitive to definitions.

Focus on:
1. Key mathematical concepts and objects involved in the proof
2. Main techniques or lemmas that might be needed
3. Specific properties or structures that appear in the statement

Return your response in the following JSON format. Please wrap the JSON block with ```json\\n ... \\n```:
```json
{{
    "queries": [
        "query 1 description",
        "query 2 description",
        "query 3 description",
        ...,
        "query {num_queries} description"
    ]
}}
```

Each query should be a clear, specific mathematical concept or theorem name that would likely appear in a mathematical database.
"""


GET_QUERY_REFLECT = """\
You are an expert in formal mathematics and Lean 4. Given a theorem statement, its proof context, and error information from a previous proof attempt, your task is to generate {num_queries} precise search queries that would help find relevant lemmas, theorems, or definitions to fix the errors.

**Theorem Statement:**
```lean4
{lean_code}
```

**Informal Description:**
{informal_statement}

**Previous Proof Attempt:**
```lean4
{proof}
```

**Error Messages:**
{error_msg}

### IMPORTANT: How to Handle "Not Found" Errors
If the error messages contain phrases like `Unknown constant ...`, `unknown identifier '...'` (or `Unknown constant`, `invalid field`), this is a critical signal. It means that an import path or a specific theorem name used in the previous proof is **invalid or non-existent in the current library**.

In this situation, your primary task is to find an equivalent or an alternative. One of your new search queries **MUST** be a **pure natural language description** of the *mathematical concept* you were trying to use.

**Example:**
- **IF** the error is: `Unknown constant 'Nat.prime_factor_unique'`
- **DO NOT** generate a query like: `"Nat.prime_factor_unique"` (this will fail again)
- **INSTEAD**, generate a query like: `"unique prime factorization theorem for natural numbers"` (this will help find the correct, existing theorem)

Based on the errors encountered, please analyze what went wrong and then generate exactly {num_queries} **New** search queries that would help find the missing pieces to fix these specific errors. Tips: when you need to search for definitions, you should clearly state that you are searching for a definition of something in the query, because LeanSearch are not sensitive to definitions.

Focus on:
1. Lemmas or theorems that could resolve the specific error messages, especially those related to incorrect theorem names or terms (`unknown identifier '...'` (or `Unknown constant '...'`, `invalid field '...'`)).
2. Alternative proof techniques that might avoid the current issues

Return your response in the following JSON format. Please wrap the JSON block with ```json\\n ... \\n```:
```json
{{
    "queries": [
        "query 1 description",
        "query 2 description",
        "query 3 description",
        ...,
        "query {num_queries} description"
    ]
}}
```

Each query should target specific mathematical concepts that could directly address the encountered errors.
"""
