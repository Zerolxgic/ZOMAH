# T3c Live Results — Candidate Union and Deterministic Ambiguity Audit

Date: 2026-09-27

Commit under test: `d97e168`

Input artifact:

`~/.local/share/zomah/experiments/qwen3-embedding/run-1/results.json`

Audit configuration:

- candidate set: `union(lexical top-3, semantic top-3)`
- deduplicated by document path
- labels used only for evaluation, never candidate construction or runtime classification
- agreement: lexical #1 and semantic #1 are the same document
- dominance: exactly one candidate outranks every other candidate on both rankings and is strictly better on at least one
- ambiguous: neither agreement nor unique dominance
- no model load, no re-embedding, no retrieval rerun, no Jev/Laya call

## Results

Candidate recall:

- lexical Top-3: `19 / 30` (`63.3%`)
- semantic Top-3: `27 / 30` (`90.0%`)
- union Top-3: `30 / 30` (`100.0%`)

Union recall by category:

- exact: `7 / 7` (`100.0%`)
- paraphrase: `10 / 10` (`100.0%`)
- conceptual: `5 / 5` (`100.0%`)
- distractor: `8 / 8` (`100.0%`)

Candidate-set size:

- mean: `4.77`
- median: `5`
- maximum: `6`
- distribution: `{3: 2, 4: 6, 5: 19, 6: 3}`

Deterministic resolution:

- agreement cases: `11`
- agreement correct: `10 / 11`
- unique dominance cases: `0`
- ambiguous cases: `19`
- ambiguous cases with expected document in candidate set: `19 / 19`
- retrieval failures: `0`

For the 19 ambiguous cases, the expected document was:

- lexical #1: `8`
- semantic #1: `8`
- another candidate: `3`
- not a candidate: `0`

Diagnostic selector performance on ambiguous cases:

- lexical #1: `8 / 19` (`42.1%`)
- semantic #1: `8 / 19` (`42.1%`)
- RRF #1: `9 / 19` (`47.4%`)

Reference ceilings / totals:

- deterministic agreement + perfect judge on ambiguous cases: `29 / 30`
- lexical Top-1: `18 / 30`
- semantic Top-1: `18 / 30`
- RRF Top-1: `19 / 30`

All T3a agreement buckets were represented in the candidate union:

- both-correct: `10 / 10`
- lexical-only: `8 / 8`
- semantic-only: `8 / 8`
- both-wrong: `4 / 4`

## Interpretation

This audit cleanly separates retrieval failure from ranking/judgment failure.

The combined Top-3 candidate generator achieved `100%` recall on this 30-case benchmark. Every expected document entered the deterministic candidate set, including all exact, paraphrase, conceptual, distractor, lexical-only, semantic-only, and both-wrong cases. Therefore the observed remaining problem is not candidate generation on this benchmark; it is selecting the correct source from a small candidate set.

The candidate set is genuinely bounded: typically five documents, with a maximum of six. This is a much narrower role than retrieval itself and is the first measured retrieval-stage job that could plausibly justify a bounded probabilistic judge.

The deterministic agreement rule resolved 11 cases but made one wrong decision, so agreement is useful but is not a correctness guarantee. The specified Pareto-dominance rule resolved no additional cases. This follows from the geometry of a union that always contains both retrievers' first-ranked documents: absent an empty ranking, a candidate that dominates every other candidate must also be #1 in both rankings, which is already agreement.

Nineteen cases remained ambiguous, and the expected source was inside the candidate set in all nineteen. Eight favored lexical Top-1, eight favored semantic Top-1, and three required selecting a document that neither retriever ranked first. Lexical, semantic, and equal-weight RRF each perform poorly as a final selector on this ambiguous subset (`42.1%`, `42.1%`, and `47.4%` respectively).

The `29 / 30` deterministic-plus-perfect-judge reference is not a realizable system score. It assumes the eleven deterministic agreement decisions remain fixed, including the one incorrect agreement, while a hypothetical perfect judge resolves every ambiguous case. Because the expected document is present in the candidate union for all thirty cases, a hypothetical selector allowed to revisit agreement cases has an oracle ceiling of `30 / 30`; that is also not a system result.

## Architectural consequence

T3c provides concrete evidence for a possible bounded judgment experiment while keeping responsibilities separate:

1. **Lexical retrieval (FTS5/BM25)** — candidate generation with strong exact-term precision.
2. **Qwen3-Embedding-0.6B** — complementary semantic candidate generation.
3. **Deterministic ZOMAH layer** — Top-3 union, deduplication, scope, provenance, budgets, and classification of obvious agreement versus unresolved ambiguity.
4. **Jev OR Laya (next experiment candidate)** — alternative implementations of one bounded selection role over a small explicit candidate set, only where deterministic retrieval/ranking leaves ambiguity.
5. **read_file / localization** — source-grounded evidence retrieval after a source is selected.
6. **LLM worker** — interpretation and synthesis from retrieved evidence.
7. **Human/policy gates** — authority for consequential actions.

A future Jev/Laya benchmark should not ask either model to search, embed documents, enforce permissions, make lifecycle decisions, or perform action policy. The experimentally justified input/output shape is narrower:

```text
input:
  query
  bounded candidate set (typically 3–6 documents)
  deterministic retrieval features/evidence

output:
  candidate choice / score / no-selection
```

Jev and Laya should be compared as alternative implementations of this same role, not stacked sequentially by default.

## Decision

Candidate recall is sufficiently strong on this benchmark to justify the next isolated experiment: compare Jev and Laya on the 19 ambiguous bounded-choice cases, while retaining the 11 agreement cases as diagnostics rather than treating deterministic agreement as infallible.

No production integration decision is made by T3c alone.
