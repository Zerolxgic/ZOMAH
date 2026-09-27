# T3a: Qwen3-Embedding-0.6B document-ranking experiment

An isolated experiment, not production retrieval. It asks one question:

> Can Qwen3-Embedding-0.6B rank the correct document better than ZOMAH's current lexical search when the query's wording differs from the source's?

It compares document ranking only. Within-document localization, chunking, rerankers, hybrid fusion, vector stores and GPU work are out of scope. Nothing here changes `search_knowledge`, the model's tools, or the Operator Console, and none of these dependencies belong in ZOMAH's `pyproject.toml`.

## Layout

```text
benchmark.json            committed evaluation manifest (30 cases, 4 categories)
run_benchmark.py          T3a CLI: download, run, write results.json + report.md
run_fusion.py             T3b CLI: RRF over a T3a results.json (standard library only)
T3A-LIVE-RESULTS.md       summary of the first real-machine T3a run
requirements.txt          experiment-only dependencies (verified versions)
embedding_bench/
  manifest.py             manifest validation + evidence-anchor drift checks
  corpus.py               corpus = exactly what ZOMAH's KnowledgeIndex holds
  retrieval.py            lexical (real search_knowledge) + brute-force cosine ranking
  metrics.py              top-1 / top-3 / MRR, per category, top-1 agreement
  runner.py               runs both sides, timings, memory -> JSON-ready dict
  report.py               Markdown report + terminal summary
  qwen.py                 the real embedder (the only module importing torch)
  fusion.py               T3b: T3a input validation, RRF, fused metrics
  fusion_report.py        T3b Markdown report + terminal summary
tests/                    pure-logic tests (no torch, no model download)
```

## Setup (once)

From the ZOMAH checkout. The experiment gets its own Python 3.13 environment (verified here; ZOMAH itself stays on its own interpreter). [uv](https://docs.astral.sh/uv/) fetches Python 3.13 if it is not installed:

```bash
uv venv --python 3.13 experiments/qwen3_embedding/.venv
uv pip install --python experiments/qwen3_embedding/.venv/bin/python \
  torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python experiments/qwen3_embedding/.venv/bin/python \
  -r experiments/qwen3_embedding/requirements.txt -e .
```

The CPU-only torch wheel keeps the environment small and keeps the first run independent of the GPU runtime used by the conversational model. `.venv/` is git-ignored and skipped by the knowledge index.

Download the model into the normal Hugging Face cache (about 1.2 GB; never committed):

```bash
experiments/qwen3_embedding/.venv/bin/python experiments/qwen3_embedding/run_benchmark.py --download-only
```

## Run

```bash
experiments/qwen3_embedding/.venv/bin/python experiments/qwen3_embedding/run_benchmark.py \
  --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1
```

Without `--output-dir`, results go to `$XDG_DATA_HOME/zomah/experiments/qwen3-embedding/<UTC time>/`. Keep outputs outside any `--read-root`: `report.md` would otherwise be indexed by `search_knowledge`.

The run prints a summary and writes:

- `results.json`: configuration, environment and package versions, corpus with per-document token counts and embed times, timings, peak RSS, metrics, top-1 agreement buckets, and every case with both full rankings.
- `report.md`: metrics by category, every disagreement with both top-3 lists and scores, per-query results, timing, corpus, drift and configuration.

`--fake-embedder` runs the whole pipeline with a hashing bag-of-words stand-in (no torch needed). Its "semantic" numbers are meaningless and the reports say so.

Tests (in ZOMAH's own environment or this one):

```bash
python -m pytest experiments/qwen3_embedding/tests
```

## Method

**Corpus.** A fresh temporary `KnowledgeIndex` over `--root` (default: this checkout) with ZOMAH's own rules (`.md`/`.txt`/`.rst`, UTF-8, ≤ 2 MiB, no symlinks, tooling directories skipped). The semantic side embeds the documents read back from that index, so both sides see identical files and text. The `experiments/` directory is additionally excluded because this README describes the benchmark queries.

**Lexical baseline.** The unchanged model-facing `search_knowledge` capability (FTS5/BM25, refresh before each search). A document that matches no query term is not returned (rank "—", reciprocal rank 0).

**Semantic.** `Qwen/Qwen3-Embedding-0.6B` through sentence-transformers on CPU in float32, one text per forward pass, full 1024-dimension output, L2-normalized. Documents are embedded whole with no instruction. Queries use the model's documented format, `Instruct: {task}\nQuery:{query}`, with the task:

> Given a question about a local software project and its documentation, retrieve the document that contains the evidence needed to answer it.

Every query is scored against every document by dot product of unit vectors (cosine) in an in-memory matrix. Ties break by path, like the lexical `ORDER BY bm25, path`. At load the embedder checks that sentence-transformers passes text through unchanged (sentence-transformers 5 can wrap text in a chat template for some model configurations) and refuses to run if it does not; the decoded probe is recorded in `results.json`. Documents longer than the model's maximum tokens are reported, not chunked.

**Metrics.** Top-1 accuracy, top-3 recall and MRR for each side, overall and per category. "Correct" in the agreement buckets means the expected document ranked first.

**Timing.** Library import, model load (cached weights only; the run sets `HF_HUB_OFFLINE=1`), corpus embedding (sum of per-document forward passes), and warm query latency (query embedding plus brute-force scoring, after one discarded warm-up query) are measured separately. Peak RSS comes from `resource.getrusage`.

## The manifest

Each case has `id`, `category`, `query`, `expected_path` (relative to the root), `evidence_anchor`, and a `rationale` saying why only that document holds the evidence. Before scoring, the expected file must still contain the anchor (whitespace-normalized, otherwise verbatim). If it does not, the case is reported as drift and not scored. Anchors that also appear in another document are flagged as weaker labels.

Categories: `exact` (source terms), `paraphrase` (the same fact in everyday wording), `conceptual` (reasons or kinds of things in synonyms), `distractor` (vocabulary shared with nearby documents, evidence in one). Several exact and paraphrase cases share the same evidence, so wording is the only difference between them.

## T3b: rank fusion over the frozen T3a rankings

T3b asks how much of the lexical/semantic complementarity T3a measured an unsupervised, deterministic fusion rule recovers. It reads a T3a `results.json` and never loads a model or re-embeds anything; any Python 3.11+ with only the standard library runs it:

```bash
python experiments/qwen3_embedding/run_fusion.py \
  --input ~/.local/share/zomah/experiments/qwen3-embedding/run-1/results.json \
  --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1-fusion
```

It writes `results.json` and `report.md` (refusing to write into the T3a run's own directory) and prints lexical, semantic and RRF metrics overall and by category.

**Fusion.** Standard equal-weight Reciprocal Rank Fusion with the conventional k = 60, fixed and not tuned: `rrf(d) = 1/(60 + lexical_rank) + 1/(60 + semantic_rank)`, where a document absent from one ranking gets nothing from that side. Only rank positions are used; BM25 relevance and cosine similarity are never combined. Scores are exact fractions, so rank pairs such as (1, 3) and (3, 1) tie exactly; ties go to path ascending, and every top-1 decided that way is reported. Labels and categories are used only to score the fused ranking, never to produce it.

**Frozen input.** Before fusing, the T3a artifact must contain every case's full lexical and semantic rankings (the semantic one covering the whole corpus), and its stored ranks, metrics and agreement buckets must follow from those rankings. The cases must carry the manifest's labels unchanged, with any unscored manifest case listed as T3a drift. Missing or inconsistent data is reported and nothing is fused or regenerated.

**Report.** Top-1 / top-3 / MRR for all three systems overall and per category; which T3a lexical-only, semantic-only and both-correct top-1 wins RRF keeps; which both-wrong cases it recovers; where the expected document's fused rank is better or worse than both inputs; top-3 losses against each input; ties decided by path; the RRF top-3 for every RRF miss; and every query's three ranks. The top-1 union of the two inputs is shown as an oracle reference, not a system result.

