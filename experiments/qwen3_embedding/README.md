# T3a: Qwen3-Embedding-0.6B document-ranking experiment

An isolated experiment, not production retrieval. It asks one question:

> Can Qwen3-Embedding-0.6B rank the correct document better than ZOMAH's current lexical search when the query's wording differs from the source's?

It compares document ranking only. Within-document localization, chunking, rerankers, hybrid fusion, vector stores and GPU work are out of scope. Nothing here changes `search_knowledge`, the model's tools, or the Operator Console, and none of these dependencies belong in ZOMAH's `pyproject.toml`.

## Layout

```text
benchmark.json            committed evaluation manifest (30 cases, 4 categories)
run_benchmark.py          T3a CLI: download, run, write results.json + report.md
run_fusion.py             T3b CLI: RRF over a T3a results.json (standard library only)
run_candidate_audit.py    T3c CLI: candidate union + ambiguity audit (standard library only)
prepare_cases.py          T3d step 1: freeze judge cases (ZOMAH environment)
run_jev_judge.py          T3d step 2a: Jev on the frozen cases (typesafe-sdk environment)
run_laya_judge.py         T3d step 2b: Laya on the frozen cases (laya environment)
compare_judges.py         T3d step 3: score and compare the judges (standard library only)
run_localization.py       T3e: lexical vs semantic within-document localization (embedding environment)
run_localization_fusion.py T3f: projection + packet fusion over a T3e results.json (no model)
requirements-jev.txt      T3d Jev judge dependency (verified version)
requirements-laya.txt     T3d Laya judge dependency (verified version)
T3A-LIVE-RESULTS.md       summary of the first real-machine T3a run
T3B-LIVE-RESULTS.md       summary of the real-machine T3b fusion run
T3C-LIVE-RESULTS.md       summary of the real-machine T3c candidate audit
T3E-LIVE-RESULTS.md       summary of the real-machine T3e localization run
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
  candidates.py           T3c: candidate union, resolution classes, audit
  candidates_report.py    T3c Markdown report + terminal summary
  judge_contract.py       T3d: judge input, the one rendered request, answer normalization, run loop
  judge_prepare.py        T3d: candidate evidence packs from T3a + T3c + lexical localization
  judge_adapters.py       T3d: Jev (typesafe-sdk) and Laya adapters
  judge_runner.py         T3d: shared runner command flow
  judge_compare.py        T3d: metrics, baselines, pairwise and agreement diagnostics
  judge_report.py         T3d Markdown report + terminal summary
  passages.py             T3e: 3-line windows, bounded excerpts, semantic passage ranking
  evidence.py             T3e: anchor spans and post-localization coverage scoring
  localization_runner.py  T3e: label-free localization, then evaluation, timings, candidate packs
  localization_report.py  T3e Markdown report + terminal summary
  projection.py           T3f: query-aware projection of a selected passage (ZOMAH's lexical matcher)
  localization_fusion.py  T3f: T3e input validation, label-free decisions, packets, evaluation
  localization_fusion_report.py  T3f Markdown report + terminal summary
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

## T3c: candidate union and ambiguity audit

T3c asks how often the correct document is in a small candidate set built from both retrievers, and how many cases simple deterministic rules cannot resolve. It reads the same T3a `results.json`, uses the standard library only, and never loads a model:

```bash
python experiments/qwen3_embedding/run_candidate_audit.py \
  --input ~/.local/share/zomah/experiments/qwen3-embedding/run-1/results.json \
  --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1-candidate-audit
```

**Candidate set.** `union(lexical top 3, semantic top 3)`, deduplicated by path. Each candidate keeps its rank in both full rankings; a document a ranking never returned counts as worse than any rank.

**Resolution classes** (from the rankings alone, never the labels): AGREEMENT when both retrievers put the same document first; DOMINANCE when, otherwise, exactly one candidate is ranked at least as well as every other candidate on both rankings and strictly better on one; AMBIGUOUS when neither applies. Because the union always holds both first places, unique dominance can only differ from agreement when one ranking is empty. The report says so, and shows the non-dominated candidates (the Pareto front) as the smallest set a judge would need.

**Evaluation** (labels read only here): candidate recall for lexical top 3, semantic top 3 and the union, overall and by category; candidate-set sizes; accuracy of deterministic resolutions; every ambiguous case with its candidates' ranks; and every retrieval failure, meaning the expected document is not a candidate. Retrieval failures are kept separate from ambiguous cases whose answer is a candidate, which are the only scope a bounded judge could address. Lexical #1, semantic #1 and RRF (k = 60) are reported on the ambiguous cases as labelled diagnostics, never as the resolver.

## T3d: Jev vs Laya on bounded retrieval selection

T3d asks whether a bounded judge can pick the right document among the small candidate set T3c builds, on the cases deterministic agreement does not resolve. Jev and Laya are two alternative implementations of that one job. They run independently on identical input and are compared with each other and with deterministic baselines; they are never stacked or combined.

**The job.** Input: one query and the T3c candidates (`union(lexical top 3, semantic top 3)`), each with its path, lexical rank, semantic rank, and an excerpt from the unchanged `zomah.knowledge.localize` (at most 160 characters, with its line range). Output: one candidate, or `none`. Candidates are listed by path, not by retrieval rank, so option position carries no retrieval signal; a "first listed option" baseline measures position bias. The judge never sees the case id, category, expected document, evidence anchor, rationale, T3a/T3c outcomes, or RRF, and it cannot search or read files.

**One request, both judges.** Each case is rendered once into a plain-text state (`Question: …`) and a single choice question whose options are the candidates (evidence first, then path, lines and ranks) plus `none`. Jev's System One API and Laya both accept exactly this question shape, so both receive byte-identical requests, stored in the cases file with a SHA-256 per case. Answers keep each system's own probabilities and confidence; nothing is thresholded. A choice outside the supplied ids, a malformed answer, or an error is a protocol failure.

**Workflow.** Four steps, each in its own environment:

```bash
# 1. ZOMAH environment: freeze the cases (checks the corpus still matches T3a;
#    --t3c-results also checks candidate sets and baselines against T3c)
python experiments/qwen3_embedding/prepare_cases.py \
  --input ~/.local/share/zomah/experiments/qwen3-embedding/run-1/results.json \
  --t3c-results ~/.local/share/zomah/experiments/qwen3-embedding/run-1-candidate-audit/results.json \
  --output ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/cases.json

# 2a. Jev environment (typesafe-sdk; TYPESAFE_API_KEY set)
python experiments/qwen3_embedding/run_jev_judge.py \
  --cases ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/cases.json \
  --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/jev

# 2b. Laya environment (laya)
python experiments/qwen3_embedding/run_laya_judge.py \
  --cases ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/cases.json \
  --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/laya

# 3. Any Python 3.11+, standard library only
python experiments/qwen3_embedding/compare_judges.py \
  --cases ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/cases.json \
  --jev   ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/jev/judgments.json \
  --laya  ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/laya/judgments.json \
  --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/comparison
```

Neither the cases file nor a judgments file is replaced without `--overwrite`. Each runner does one warm-up call on a fixed non-benchmark request, then sends every case once; it records the raw provider output, the normalized selection, latency, model load time, and versions. The Jev runner uses `typesafe_sdk` directly (the `jev` decorator package discards probabilities) with the SDK's default model (`jev-latest`) unless `--model` is given. The Laya runner loads `convaiinnovations/laya` with the checkpoint's own token budgets unless `--max-len` / `--head-max-len` are given, and records, using Laya's own tokenizer, how much of each option and of the state it actually read.

**Report.** On the ambiguous cases, for each judge: accuracy, abstentions, accuracy when selecting, protocol failures, latency, results by category and by whether the answer was lexical #1, semantic #1 or neither, and the cases it wins or loses against RRF. Baselines (lexical #1, semantic #1, RRF #1, T3c order, first option) are recomputed from the frozen cases. Also: pairwise judge agreement, agreement-case diagnostics (does the judge ever overturn a correct agreement, or fix the wrong one), and evidence-pack limits: whether the expected document's excerpt actually covers the benchmark evidence, and Laya's truncation.

**Known evidence-pack limit.** The excerpt comes from lexical localization, so for reworded queries it often lands on the wrong lines of the right document. `prepare_cases.py` prints how many ambiguous cases show the evidence in the expected candidate's excerpt, and the report separates those cases from judgment failures.

## T3e: semantic within-document evidence localization

T3e asks whether Qwen3-Embedding can find the evidence *inside* a document where the lexical `localize()` cannot. Each benchmark query is localized in its benchmark document (an evaluation setup that separates localization from retrieval; production would not know that document), by both the unchanged `zomah.knowledge.localize()` and a semantic localizer. Nothing here changes production localization.

```bash
experiments/qwen3_embedding/.venv/bin/python experiments/qwen3_embedding/run_localization.py \
  --t3a-results ~/.local/share/zomah/experiments/qwen3-embedding/run-1/results.json \
  --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3e
```

It writes `results.json`, `report.md` and, with `--t3a-results`, `candidate_packs.json`, and prints a summary. `--fake-embedder` checks the pipeline without the model.

**Semantic localizer.** Every contiguous 3-line window of the document (stride 1, lines split on newline and numbered like `read_file`; windows without a letter or digit are skipped) is embedded as-is with the T3a embedder unchanged (CPU, float32, 1024 dimensions, normalized). The query is embedded once with the T3a instruction. The window with the highest cosine similarity is selected; ties go to the earliest start line. Windows are embedded once per document and reused across queries; per-document embedding time is reported as the cost of localizing in a document not seen before.

**Excerpts.** From each localizer's selected passage, the same rule cuts 80-, 120- and 160-character excerpts: the passage's leading characters, backing up to whitespace rather than splitting a word when that keeps half the budget. The lexical localizer's own production excerpt (≤160 characters, centred on its first match) is reported separately.

**Evaluation, after localization.** Localizers receive only the query and the document; the benchmark anchor, category and labels are read afterwards. Coverage is measured by source position: selected lines overlapping the anchor's lines, the passage holding the whole anchor, and, per excerpt budget, whether the evidence survives (the excerpt holds the whole anchor, or the anchor is longer and the excerpt is entirely anchor text), plus the share of the anchor shown. Results are reported overall and by category, with lexical-vs-semantic case buckets, every disagreement with both selections, and the semantic top-3 windows per case with the rank of the first window that reaches the evidence.

**Drift guards.** Cases whose anchor is gone are reported and not scored. With `--t3a-results` the corpus must still match T3a. The recomputed lexical coverage must match the recorded baseline (line overlap: exact 7/7, paraphrase 0/10, conceptual 0/5, distractor 3/8); otherwise the run stops before loading the model.

**Candidate-pack simulation.** With `--t3a-results`, the semantic localizer is also applied to every document in each case's T3c candidate union, producing one passage per candidate in `candidate_packs.json`. It is not scored and is sent to no judge.

## T3f: deterministic evidence projection + bounded localization fusion

T3e showed that selection has little headroom (lexical-or-semantic top-1 reaches 21/30 against 20/30 for semantic alone) while projection loses evidence (the semantic passage holds it in 20/30 cases, its 160-character leading excerpt in 12/30). T3f works on projection first. It reads a T3e `results.json` and nothing else: no model, no corpus, no re-embedding, and T3e's window size is unchanged. Run it in the ZOMAH environment:

```bash
.venv/bin/python experiments/qwen3_embedding/run_localization_fusion.py \
  --t3e-results ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3e/results.json \
  --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3f
```

It writes `results.json` and `report.md` and prints a summary.

**Order of work.** The artifact's structure is checked and its recorded leading excerpts must reproduce from its passages. Label-free cases (id, query, path, both passages, lexical native excerpt, semantic similarity and top windows) are built, and every projection and packet is decided and frozen (`decisions_sha256`). Only then are categories and anchors read: the recorded T3e evaluation, metrics, pairwise buckets and top-3 diagnostics must reproduce, the recomputed baselines must equal the accepted live run in `T3E-LIVE-RESULTS.md`, and the frozen decisions are scored.

**Query-aware projection.** Deterministic, from the selected passage and the query alone. If the trimmed passage fits the budget it is returned whole. Otherwise query phrases are matched with ZOMAH's own lexical matcher (`zomah.knowledge._query_phrases` and `_line_matches`, unchanged: case- and diacritic-insensitive, whole words, `tool-result` as the words tool, result). The strongest line has the most distinct phrases, then the most occurrences, then comes first. The window starts a quarter of the budget before the earliest match on that line (localize() keeps 40 of 160). With no match it is centred on the passage, never on the leading characters. A window that reaches a passage edge is shifted back inside it. A cut that would split a word (a run of letters or digits) moves to the word boundary when that keeps at least half the budget. Spans are exact source positions; "…" marks omitted passage text and does not count against the budget.

**Projection evaluation.** Both frozen T3e passages (lexical and semantic) are projected by the T3e leading rule and by the query-aware rule at 80, 120 and 160 characters. The report gives, per method and budget, evidence survival, whole-anchor containment and mean anchor coverage, overall and by category, plus leading-vs-query-aware case buckets, and how many of the semantic passage's line hits keep their evidence after compression.

**Packet fusion.** `semantic_primary_dual_on_disagreement`: the semantic passage comes first. When the lexical and semantic line ranges overlap, only the semantic projection is emitted, with the whole total budget. When they are disjoint, the semantic projection is followed by the lexical projection, each with half the total. Totals are 160, 240 and 320 characters, and the split never depends on correctness. A control spends the same total on the semantic passage alone, so gains from dual evidence are separated from gains from a larger budget. Packets are scored on the union of their emitted source spans, with emitted source characters (mean, median, max) and the share of cases that needed dual evidence reported alongside.

**Headroom diagnostics.** After scoring: the top-1 passage oracle (lexical or semantic lines overlap the evidence), semantic top-1 and semantic top-3.

**Drift guards.** A malformed, incompatible or internally inconsistent artifact stops the run (exit 2). Recomputed baselines that differ from the accepted live run stop it too (exit 3), naming every differing value. `--skip-t3e-reference` is for fixture runs only and is flagged in every output.
