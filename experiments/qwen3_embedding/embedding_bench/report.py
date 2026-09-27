"""Render a benchmark result dict as Markdown and as a short terminal summary."""

from __future__ import annotations

from typing import Any

_BUCKET_TITLES = {
    "both_correct": "Both correct",
    "lexical_correct_semantic_wrong": "Lexical correct / semantic wrong",
    "semantic_correct_lexical_wrong": "Semantic correct / lexical wrong",
    "both_wrong": "Both wrong",
}


def render_markdown(result: dict[str, Any]) -> str:
    config = result["config"]
    embedder = config["embedder"]
    timings = result["timings"]
    corpus = result["corpus"]
    lines: list[str] = [
        "# Qwen3-Embedding document-ranking experiment (T3a)",
        "",
        f"{result['experiment']}. Run {result['started_at']} to {result['finished_at']}.",
        "",
    ]
    if embedder.get("embedder") == "fake-hashing":
        lines += [
            "> **FAKE EMBEDDER.** This run used a hashing bag-of-words stand-in to check the "
            "pipeline. Its \"semantic\" numbers say nothing about Qwen3-Embedding.",
            "",
        ]

    lines += ["## Metrics", "", *_metrics_table(result), ""]
    lines += [
        "Top-1 = expected document ranked first. Top-3 = expected document in the first three. "
        "MRR = mean of 1/rank (0 when the document is not returned; lexical search omits documents "
        "with no matching term).",
        "",
    ]

    lines += ["## Top-1 agreement", ""]
    for bucket, title in _BUCKET_TITLES.items():
        ids = result["agreement"][bucket]
        lines.append(f"- **{title}** ({len(ids)}): {', '.join(ids) if ids else '—'}")
    lines.append("")

    cases = {case["id"]: case for case in result["cases"]}
    for bucket in ("lexical_correct_semantic_wrong", "semantic_correct_lexical_wrong", "both_wrong"):
        ids = result["agreement"][bucket]
        if not ids:
            continue
        lines += [f"## {_BUCKET_TITLES[bucket]}", ""]
        for case_id in ids:
            lines += _case_detail(cases[case_id])

    lines += ["## Per-query results", ""]
    lines += [
        "| id | category | lexical rank | semantic rank | lexical top-1 | semantic top-1 | agreement |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for case in result["cases"]:
        lines.append(
            f"| {case['id']} | {case['category']} | {_rank(case['lexical'])} | "
            f"{_rank(case['semantic'])} | {_top1_path(case['lexical'])} | "
            f"{_top1_path(case['semantic'])} | {case['agreement']} |"
        )
    lines.append("")
    lines += ["<details><summary>Queries and evidence anchors</summary>", ""]
    for case in result["cases"]:
        lines += [
            f"- **{case['id']}** ({case['category']}) — {case['query']}",
            f"  - expected `{case['expected_path']}`; anchor: “{case['evidence_anchor']}”",
        ]
    lines += ["", "</details>", ""]

    lines += ["## Timing and resources", ""]
    query = timings["semantic_query_ms"]
    if timings.get("library_import_s") is not None:
        lines.append(f"- Library import (torch, sentence-transformers; excluded from model load): {_s(timings['library_import_s'])}")
    lines += [
        f"- Model load (cached weights, no download): {_s(timings['model_load_s'])}",
        f"- Corpus embedding: {_s(timings['corpus_embed_s'])} for {corpus['documents']} documents "
        f"(wall {_s(timings['corpus_build_wall_s'])} including token counting)",
        f"- Warm-up query (excluded): {_ms(timings['warmup_query_ms'])}",
        f"- Warm semantic query (embed + score): median {_ms(query['median'])}, "
        f"mean {_ms(query['mean'])}, min {_ms(query['min'])}, max {_ms(query['max'])} "
        f"(n={query['n']})",
        f"  - of which query embedding: median {_ms(timings['semantic_query_embed_ms']['median'])}; "
        f"brute-force scoring (pure Python, no numpy): median {_ms(timings['semantic_query_score_ms']['median'])}",
        f"- Lexical query (search_knowledge incl. incremental refresh): median "
        f"{_ms(timings['lexical_query_ms']['median'])}, max {_ms(timings['lexical_query_ms']['max'])}",
        "- Peak RSS (MB, monotonic): "
        + ", ".join(f"{stage} {_mb(value)}" for stage, value in result["memory_peak_rss_mb"].items()),
        "",
    ]

    lines += ["## Corpus", ""]
    lines += [
        f"{corpus['documents']} documents, {corpus['total_bytes']:,} bytes, embedding dimension "
        f"{corpus['embedding_dimension']}, model max tokens {corpus['max_tokens'] or 'n/a'}.",
        "",
    ]
    if corpus["over_max_tokens"]:
        lines += [
            "**Documents longer than the model's max tokens (truncated when embedded):** "
            + ", ".join(f"`{p}`" for p in corpus["over_max_tokens"]),
            "",
        ]
    lines += ["| document | bytes | tokens | embed time |", "|---|---:|---:|---:|"]
    for row in corpus["files"]:
        tokens = "n/a" if row["tokens"] is None else f"{row['tokens']:,}"
        lines.append(f"| {row['path']} | {row['size_bytes']:,} | {tokens} | {_s(row['embed_s'])} |")
    lines.append("")

    lines += ["## Benchmark integrity", ""]
    manifest = result["manifest"]
    lines.append(
        f"Manifest `{manifest['path']}` (sha256 `{manifest['sha256'][:12]}`): {manifest['cases']} cases, "
        f"{manifest['scored']} scored, {manifest['drift']} drifted."
    )
    lines.append("")
    if result["drift"]:
        lines += ["**Drifted cases (not scored):**", ""]
        for item in result["drift"]:
            lines.append(f"- {item['id']}: `{item['expected_path']}` — {item['reason']}")
        lines.append("")
    shared = [case for case in result["cases"] if case["anchor_also_in"]]
    if shared:
        lines += ["**Anchors that also appear in other documents (weaker labels):**", ""]
        for case in shared:
            lines.append(f"- {case['id']}: also in {', '.join(case['anchor_also_in'])}")
        lines.append("")

    lines += ["## Configuration", ""]
    packages = embedder.get("packages") or {}
    lines += [
        f"- Embedder: {_kv({k: v for k, v in embedder.items() if k != 'packages'})}",
        f"- Packages: {_kv(packages) if packages else 'n/a'}",
        f"- Query instruction: “{config['query_instruction']}”",
        f"- Query format: `{config['query_format_example']!r}`",
        "- Documents: embedded whole, no instruction",
        f"- Similarity: {config['similarity']}; ties: {config['tie_break']}",
        f"- Lexical: {config['lexical']['implementation']}, max_results {config['lexical']['max_results']}",
        f"- Corpus: {_kv(config['corpus'])}",
        f"- Environment: {_kv(result['environment'])}",
        "",
    ]
    return "\n".join(lines)


def render_summary(result: dict[str, Any], *, json_path: str | None = None, markdown_path: str | None = None) -> str:
    timings = result["timings"]
    embedder = result["config"]["embedder"]
    lines = [
        f"T3a document ranking — embedder: {embedder.get('model_id', embedder.get('embedder'))}",
    ]
    if embedder.get("embedder") == "fake-hashing":
        lines.append("  FAKE EMBEDDER: pipeline check only, semantic numbers are meaningless")
    corpus = result["corpus"]
    manifest = result["manifest"]
    lines += [
        f"  corpus {corpus['documents']} docs, dim {corpus['embedding_dimension']}; "
        f"cases {manifest['scored']}/{manifest['cases']} scored ({manifest['drift']} drifted)",
        f"  model load {_s(timings['model_load_s'])}, corpus embed {_s(timings['corpus_embed_s'])}, "
        f"warm query median {_ms(timings['semantic_query_ms']['median'])} "
        f"(lexical {_ms(timings['lexical_query_ms']['median'])})",
        "",
        *_metrics_table(result, markdown=False),
        "",
        "  top-1 agreement: "
        + ", ".join(f"{bucket}={len(ids)}" for bucket, ids in result["agreement"].items()),
    ]
    cases = {case["id"]: case for case in result["cases"]}
    for bucket in ("lexical_correct_semantic_wrong", "semantic_correct_lexical_wrong", "both_wrong"):
        for case_id in result["agreement"][bucket]:
            case = cases[case_id]
            lines.append(
                f"    {bucket:32} {case_id:24} lex #{_rank(case['lexical'])} "
                f"sem #{_rank(case['semantic'])}  expected {case['expected_path']}"
            )
    if corpus["over_max_tokens"]:
        lines.append(f"  WARNING truncated documents: {', '.join(corpus['over_max_tokens'])}")
    for item in result["drift"]:
        lines.append(f"  DRIFT {item['id']}: {item['reason']}")
    if json_path:
        lines.append(f"  json:     {json_path}")
    if markdown_path:
        lines.append(f"  markdown: {markdown_path}")
    return "\n".join(lines)


def _metrics_table(result: dict[str, Any], *, markdown: bool = True) -> list[str]:
    metrics = result["metrics"]
    rows = [("overall", metrics["lexical"]["overall"], metrics["semantic"]["overall"])]
    for category in result["manifest"]["categories"]:
        rows.append(
            (category, metrics["lexical"]["by_category"][category], metrics["semantic"]["by_category"][category])
        )
    if markdown:
        lines = [
            "| slice | n | lexical top-1 | lexical top-3 | lexical MRR | semantic top-1 | semantic top-3 | semantic MRR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, lex, sem in rows:
            lines.append(
                f"| {name} | {lex['n']} | {_pct(lex['top1'])} | {_pct(lex['top3'])} | {_f(lex['mrr'])} | "
                f"{_pct(sem['top1'])} | {_pct(sem['top3'])} | {_f(sem['mrr'])} |"
            )
        return lines
    lines = [f"  {'slice':12} {'n':>3}   {'lexical top1/top3/MRR':>24}   {'semantic top1/top3/MRR':>24}"]
    for name, lex, sem in rows:
        lines.append(
            f"  {name:12} {lex['n']:>3}   "
            f"{_pct(lex['top1']):>6} {_pct(lex['top3']):>6} {_f(lex['mrr']):>6}         "
            f"{_pct(sem['top1']):>6} {_pct(sem['top3']):>6} {_f(sem['mrr']):>6}"
        )
    return lines


def _case_detail(case: dict[str, Any]) -> list[str]:
    lines = [
        f"### {case['id']} ({case['category']})",
        "",
        f"- Query: {case['query']}",
        f"- Expected: `{case['expected_path']}` — anchor: “{case['evidence_anchor']}”",
        "- Anchor content words in query: "
        + (", ".join(case["anchor_terms_in_query"]) if case["anchor_terms_in_query"] else "none"),
    ]
    if case.get("rationale"):
        lines.append(f"- Label rationale: {case['rationale']}")
    for side in ("lexical", "semantic"):
        data = case[side]
        hits = "; ".join(f"{i}. `{hit['path']}` ({hit['score']:.4f})" for i, hit in enumerate(data["top3_hits"], 1))
        expected = "not returned" if data["expected_score"] is None else f"{data['expected_score']:.4f}"
        lines.append(f"- {side.capitalize()}: rank {_rank(data)}, expected score {expected}; top-3: {hits or 'none'}")
    lines.append("")
    return lines


def _rank(side: dict[str, Any]) -> str:
    return "—" if side["rank"] is None else str(side["rank"])


def _top1_path(side: dict[str, Any]) -> str:
    return side["top3_hits"][0]["path"] if side["top3_hits"] else "—"


def _kv(mapping: dict[str, Any]) -> str:
    return "; ".join(f"{key}={value}" for key, value in mapping.items())


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} s"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def _mb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"
