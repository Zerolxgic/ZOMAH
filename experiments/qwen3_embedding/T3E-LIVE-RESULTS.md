# T3e Live Results — Semantic Within-Document Evidence Localization

Date recorded: 2026-09-28

Commit under test: `2306148`

Output artifact:

`~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3e/results.json`

Configuration:

- lexical localizer: `zomah.knowledge.localize()` (unchanged)
- semantic localizer: Qwen/Qwen3-Embedding-0.6B (T3a embedder unchanged, CPU, float32) over 3-line windows, stride 1
- selection: highest cosine similarity; ties go to the earliest start line
- excerpts: leading characters of the selected passage at 80 / 120 / 160 characters
- document under test: the benchmark's expected document (isolates localization from retrieval)
- labels read only after both localizers had run
- lexical reference check passed before the model loaded

## Results

Overall (30 cases):

| localizer | line overlap | passage | excerpt 80 | excerpt 120 | excerpt 160 |
|---|---:|---:|---:|---:|---:|
| lexical | 10/30 | 10/30 | 2/30 | 3/30 | 6/30 |
| semantic | 20/30 | 20/30 | 6/30 | 8/30 | 12/30 |

"Passage" means the selected passage holds the whole evidence anchor. "Excerpt N" means the evidence survives the N-character excerpt.

Line overlap by category:

| category | n | lexical | semantic |
|---|---:|---:|---:|
| exact | 7 | 7/7 | 6/7 |
| paraphrase | 10 | 0/10 | 6/10 |
| conceptual | 5 | 0/5 | 2/5 |
| distractor | 8 | 3/8 | 6/8 |

Pairwise line overlap:

- both hit: `9`
- lexical only: `1`
- semantic only: `11`
- both miss: `9`

Pairwise excerpt-160 survival:

- both hit: `5`
- lexical only: `1`
- semantic only: `7`
- both miss: `17`

Semantic first-hit rank (best window overlapping the evidence):

- #1: `20`
- #2: `1`
- #3: `1`
- >3: `8`

Semantic evidence in the top 3 windows: `22/30`.

## Corpus and runtime

- documents: `13`
- passage windows: `2,293`
- model load: `3.51 s`
- passage embedding: `361.95 s`
- warm localization median (query embedding + scoring one document): `~192.7 ms`
- cold localization per document median (embed its windows + warm): `~35,667.1 ms`
- lexical `localize()` median: `~1.5 ms`

## Interpretation

- Semantic localization is complementary, not a lexical replacement. Lexical keeps one unique line hit; semantic adds eleven.
- The top-1 passage union (lexical or semantic line overlap) is `21/30`. Semantic top-1 alone already reaches `20/30`, so a perfect lexical-vs-semantic selector could recover at most one more case. Selector optimization has very little upside.
- Evidence projection is a separate bottleneck: the semantic passage holds the evidence in `20/30` cases, but the evidence survives the 160-character leading excerpt in only `12/30`.
- Semantic top-3 windows reach the evidence in `22/30`, only two more than top-1.
- Brute-force cold passage embedding (~36 s per document) is not production-viable.
- T3e authorized no production cache or index design.
