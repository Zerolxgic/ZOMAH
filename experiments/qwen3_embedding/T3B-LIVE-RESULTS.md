# T3b Live Results — Deterministic Reciprocal Rank Fusion

Date: 2026-09-27

Commit under test: `5f2736b`

Input artifact:

`~/.local/share/zomah/experiments/qwen3-embedding/run-1/results.json`

Fusion configuration:

- method: equal-weight Reciprocal Rank Fusion (RRF)
- `k = 60`
- lexical source: frozen T3a FTS5/BM25 rankings
- semantic source: frozen T3a Qwen/Qwen3-Embedding-0.6B rankings
- labels/categories used only for evaluation
- no model load, re-embedding, score normalization, learned weights, or production integration

## Results

| Slice | n | Lexical Top-1 | Lexical Top-3 | Lexical MRR | Semantic Top-1 | Semantic Top-3 | Semantic MRR | RRF Top-1 | RRF Top-3 | RRF MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 30 | 60.0% | 63.3% | 0.675 | 60.0% | 90.0% | 0.765 | 63.3% | 83.3% | 0.767 |
| Exact | 7 | 100.0% | 100.0% | 1.000 | 42.9% | 100.0% | 0.714 | 100.0% | 100.0% | 1.000 |
| Paraphrase | 10 | 50.0% | 50.0% | 0.571 | 70.0% | 100.0% | 0.833 | 50.0% | 70.0% | 0.667 |
| Conceptual | 5 | 20.0% | 20.0% | 0.334 | 40.0% | 80.0% | 0.650 | 20.0% | 60.0% | 0.500 |
| Distractor | 8 | 62.5% | 75.0% | 0.734 | 75.0% | 75.0% | 0.796 | 75.0% | 100.0% | 0.854 |

Top-1 preservation / recovery:

- both-correct cases preserved: `10 / 10`
- lexical-only wins preserved: `8 / 8`
- semantic-only wins preserved: `0 / 8`
- both-wrong cases recovered: `1 / 4`
- RRF Top-1 total: `19 / 30` (`63.3%`)
- oracle union reference: `26 / 30` (`86.7%`), not a realizable system result
- expected document ranked better than both inputs: `di-trace-location`
- expected document ranked worse than both inputs: none
- Top-1 decided by deterministic path tie-break: `di-reranker-condition`

## Interpretation

Equal-weight RRF improves overall Top-1 only modestly over either retriever alone (`63.3%` vs `60.0%`). It preserves all lexical-only wins and all both-correct cases, but preserves none of the eight semantic-only Top-1 wins.

This is an important structural result, not just a weak aggregate score. The two retrieval systems remain strongly complementary, but this particular deterministic fusion rule does not exploit that complementarity symmetrically.

### Correction after T3c audit

The original write-up attributed the lost semantic-only wins mainly to lexical retrieval omitting the correct document. That was not the explanation for these eight live cases. The T3a artifact shows the correct documents were present in the lexical rankings, generally at weaker ranks (roughly `#4` through `#13`). Equal-weight RRF at `k = 60` behaves approximately like combining rank advantages: the wrong lexical #1 often also had enough semantic support to outrank the semantically correct document, whose lexical position was much weaker. The observed loss is therefore primarily a **relative-rank geometry** effect, not an absence-from-lexical effect. Lexical retrieval can still omit unmatched documents in principle, but that was not the cause of these eight semantic-only losses.

RRF still demonstrates useful deterministic behavior:

- preserves lexical exact-term strength (`100%` Top-1 on exact queries)
- improves overall Top-1 to `63.3%`
- improves distractor Top-1 to `75.0%` and Top-3 to `100.0%`
- never ranks the expected document worse than both inputs in this run
- recovers one case both input systems missed at Top-1

But it gives back much of semantic retrieval's gain on paraphrase and conceptual queries:

- paraphrase Top-1: lexical `50%`, semantic `70%`, RRF `50%`
- conceptual Top-1: lexical `20%`, semantic `40%`, RRF `20%`

## Architectural consequence

Do not replace either retriever with equal-weight RRF as a final production ranking policy.

The current evidence supports keeping responsibilities separate:

1. **Deterministic ZOMAH substrate** — permissions, scope, schemas, lifecycle, tracing, budgets, tool execution, candidate-set construction.
2. **Lexical retrieval** — high precision when query wording overlaps source terminology.
3. **Qwen3-Embedding-0.6B** — semantic candidate generation that rescues paraphrase/conceptual cases lexical search misses.
4. **Deterministic fusion** — useful for constructing a bounded shared candidate set, but equal-weight RRF is not sufficient as the final Top-1 selector on this benchmark.
5. **Jev / Laya (future experiment)** — possible alternative bounded judgment/reranking layer over a small explicit candidate set only if deterministic ranking remains insufficient. They are not candidate generators, permission authorities, or action-policy engines.
6. **LLM worker** — open-ended reasoning and interpretation after evidence is retrieved.
7. **Human / policy gates** — authorization for consequential actions.

The next experiment should preserve a deterministic candidate-generation boundary, then test whether a bounded judge can select among the small candidate set without duplicating the embedding model's retrieval job.

No production integration decision is made by T3b alone.
