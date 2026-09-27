# T3a Live Results — Qwen3-Embedding-0.6B

Date: 2026-09-27

Status: **LIVE BENCHMARK COMPLETED**

This file records the first real-machine CPU benchmark for the isolated T3a Qwen3-Embedding-0.6B document-ranking experiment. It is a summary record, not a replacement for the full machine-generated `results.json` / `report.md` output.

## Configuration

- Model: `Qwen/Qwen3-Embedding-0.6B`
- Device: CPU
- Precision: float32
- Embedding dimension: 1024
- Corpus: 13 ZOMAH documentation files
- Benchmark cases: 30
- Drifted cases: 0
- Retrieval unit: whole-document ranking
- Semantic scoring: normalized embeddings + dot product / cosine-equivalent similarity
- Lexical baseline: unchanged ZOMAH `search_knowledge` / FTS5-BM25 path

## Timing

- Model load: **0.76 s**
- Corpus embedding: **44.98 s**
- Warm semantic query median: **216.9 ms**
- Lexical query median: **18.5 ms**

## Retrieval Metrics

| Category | n | Lexical Top-1 | Lexical Top-3 | Lexical MRR | Semantic Top-1 | Semantic Top-3 | Semantic MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 30 | 60.0% | 63.3% | 0.675 | 60.0% | 90.0% | 0.765 |
| Exact | 7 | 100.0% | 100.0% | 1.000 | 42.9% | 100.0% | 0.714 |
| Paraphrase | 10 | 50.0% | 50.0% | 0.571 | 70.0% | 100.0% | 0.833 |
| Conceptual | 5 | 20.0% | 20.0% | 0.334 | 40.0% | 80.0% | 0.650 |
| Distractor | 8 | 62.5% | 75.0% | 0.734 | 75.0% | 75.0% | 0.796 |

## Top-1 Agreement

- both correct: **10**
- lexical correct / semantic wrong: **8**
- semantic correct / lexical wrong: **8**
- both wrong: **4**

The important result is complementarity rather than dominance. Lexical retrieval is much stronger on exact terminology; semantic retrieval recovers many paraphrase and conceptual cases that lexical ranking misses badly. Semantic Top-3 recall is especially strong at **90% overall**.

## Observed Complementarity

Examples where semantic retrieval rescued a lexical miss include:

- `pa-validation-echo`: lexical #13 → semantic #1
- `pa-fingerprint`: lexical #9 → semantic #1
- `pa-protected-executable`: lexical #8 → semantic #1
- `pa-archive-not-delete`: lexical #7 → semantic #1
- `co-config-file-state`: lexical #9 → semantic #1
- `co-decision-window-order`: lexical #9 → semantic #1
- `di-stdout-budget`: lexical #4 → semantic #1
- `di-refresh-scaling`: lexical #8 → semantic #1

Examples where lexical remained stronger include exact-term and nearby-topic cases such as `ex-primitives`, `ex-migrations`, `ex-clipboard-limits`, and `ex-character-bound`.

## Current Interpretation

The benchmark does **not** justify replacing lexical retrieval with semantic retrieval.

It does justify the next isolated experiment: deterministic hybrid fusion over the frozen lexical and semantic rankings.

Candidate direction:

```text
query
  ├─ FTS5 / BM25 lexical ranking
  └─ Qwen3-Embedding semantic ranking
           ↓
     deterministic fusion
           ↓
localized evidence hint
           ↓
read_file
```

Reciprocal Rank Fusion (RRF) is preferred for the first hybrid test because it combines ranks without pretending BM25 relevance and embedding cosine scores share a meaningful scale.

## Role Separation Guardrail

ZOMAH should keep each component's job narrow so probabilistic systems do not compete for the same responsibility unnecessarily.

### Deterministic substrate

Owns machine truth and control:

- capability exposure and scope enforcement
- schema validation
- lifecycle / authority rules
- tracing and provenance
- tool-result budgets and loop bounds
- action gating
- lexical localization
- deterministic fusion such as RRF

### Lexical retrieval

Job: high-precision retrieval when source terminology overlaps the query.

### Qwen3-Embedding

Job: semantic candidate generation / ranking when wording differs from the source.

It should not become an authority or policy component.

### Jev / Laya

Reserved for bounded probabilistic judgment or classification over explicit choices when deterministic rules are insufficient.

Examples that may eventually justify them:

- bounded relevance judgment over an already-generated candidate set
- routing among explicitly defined workflow choices
- scoring a small set of structured alternatives

They should **not** automatically duplicate semantic retrieval, replace deterministic policy, or decide consequential authorization.

A reranking role for Jev/Laya should only be tested if lexical + semantic + deterministic fusion still leaves a demonstrated relevance problem.

### Generative LLM worker

Owns open-ended reasoning, tool choice, evidence interpretation, and synthesis.

### Human operator

Owns authorization for consequential decisions where policy requires approval.

## Least-Probabilistic-Capable Principle

Prefer the least probabilistic component that can do the job well:

1. deterministic rule / invariant
2. deterministic retrieval or fusion where sufficient
3. bounded learned classifier / judge (Jev, Laya, specialist ranker) when evidence shows a need
4. generative LLM for open-ended reasoning and synthesis
5. human authorization for consequential decisions

The next test should preserve this separation rather than adding another learned component simply because one is available.
