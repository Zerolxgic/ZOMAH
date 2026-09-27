"""Render a T3e localization result as Markdown and as a paste-able terminal summary."""

from __future__ import annotations

from typing import Any

MEASURES = (
    ("line_overlap", "selected lines overlap the evidence"),
    ("passage_contains_anchor", "selected passage holds the whole anchor"),
    ("excerpt_80_survives", "evidence survives the 80-char excerpt"),
    ("excerpt_120_survives", "evidence survives the 120-char excerpt"),
    ("excerpt_160_survives", "evidence survives the 160-char excerpt"),
)
NATIVE = ("native_excerpt_survives", "evidence survives localize()'s own excerpt (≤160)")


def render_markdown(result: dict[str, Any]) -> str:
    lexical, semantic = result["metrics"]["lexical"], result["metrics"]["semantic"]
    n = lexical["overall"]["n"]
    embedder = result["config"]["embedder"]
    lines = [
        "# T3e: semantic within-document evidence localization",
        "",
        f"{result['experiment']}. Run {result['started_at']} to {result['finished_at']}. Each case localizes the "
        "query inside its benchmark document (an evaluation setup that isolates localization from retrieval); "
        "benchmark labels were read only after both localizers had run.",
        "",
    ]
    if embedder.get("embedder") == "fake-hashing":
        lines += ["> **FAKE EMBEDDER.** Pipeline check only; the semantic numbers say nothing about Qwen3-Embedding.", ""]
    ref = result["lexical_reference"]
    lines += [
        "Lexical reference check: "
        + ("recomputed coverage matches the recorded baseline" if ref.get("checked") else "not checked")
        + f" ({_categories(ref['recomputed'])}).",
        "",
        "## Evidence coverage",
        "",
        "| measure | lexical | semantic |",
        "|---|---:|---:|",
    ]
    for key, label in MEASURES:
        lines.append(f"| {label} | {_frac(lexical['overall']['hits'][key], n)} | {_frac(semantic['overall']['hits'][key], n)} |")
    lines.append(f"| {NATIVE[1]} | {_frac(lexical['overall']['hits'][NATIVE[0]], n)} | — |")
    lines += [
        "",
        "\"Survives\": the excerpt holds the whole anchor, or the anchor is longer than the excerpt and the excerpt is "
        f"entirely anchor text. Anchors longer than 80/120/160 characters: {_anchor_lengths(result)}.",
        "",
    ]

    categories = list(lexical["by_category"])
    short = ("lines overlap", "whole anchor in passage", "80 survives", "120 survives", "160 survives")
    for name, metrics in (("Lexical", lexical), ("Semantic", semantic)):
        lines += [f"### {name} by category", "", "| category | n | " + " | ".join(short) + " |",
                  "|---|---:|" + "---:|" * len(MEASURES)]
        for category in categories:
            block = metrics["by_category"][category]
            lines.append(
                f"| {category} | {block['n']} | " + " | ".join(f"{block['hits'][k]}" for k, _ in MEASURES) + " |"
            )
        lines.append("")

    lines += ["### Mean share of the anchor's characters shown", "", "| | passage | 80 | 120 | 160 |", "|---|---:|---:|---:|---:|"]
    for name, metrics in (("lexical", lexical), ("semantic", semantic)):
        cov = metrics["overall"]["mean_coverage"]
        lines.append(
            f"| {name} | {_pct(cov['passage'])} | {_pct(cov['excerpt_80'])} | {_pct(cov['excerpt_120'])} | {_pct(cov['excerpt_160'])} |"
        )
    lines.append("")

    rows = {row["id"]: row for row in result["cases"]}
    lines += ["## Lexical vs semantic, case by case", ""]
    for measure, buckets in result["pairwise"].items():
        lines.append(
            f"- **{measure.replace('_', ' ')}:** both hit {len(buckets['both_hit'])}, lexical only "
            f"{len(buckets['lexical_only'])}, semantic only {len(buckets['semantic_only'])}, both miss {len(buckets['both_miss'])}."
        )
    lines.append("")
    overlap = result["pairwise"]["line_overlap"]
    for bucket, title in (("semantic_only", "Semantic only"), ("lexical_only", "Lexical only"), ("both_miss", "Both miss")):
        if overlap[bucket]:
            lines += [f"### {title} (line overlap)", ""]
            for case_id in overlap[bucket]:
                lines += _case_detail(rows[case_id])

    lines += [
        "## Semantic top-3 windows (diagnostic)",
        "",
        f"Rank of the best window that overlaps the evidence: {_histogram(result['top3']['first_hit_rank_histogram'])}. "
        f"An overlapping window is in the top 3 for {result['top3']['hit_in_top3']}/{n} cases. Windows overlap by "
        "construction (stride 1), so neighbouring ranks often share lines.",
        "",
        "| id | category | selected | top-3 (lines, similarity, evidence) | first hit rank |",
        "|---|---|---|---|---:|",
    ]
    for row in result["cases"]:
        sem = row["semantic"]
        top = "; ".join(
            f"{w['start_line']}-{w['end_line']} {w['similarity']:.3f}{' ✓' if w['overlaps_anchor'] else ''}" for w in sem["top"]
        )
        lines.append(
            f"| {row['id']} | {row['category']} | {sem['passage']['start_line']}-{sem['passage']['end_line']} | {top} | "
            f"{sem['first_hit_rank'] if sem['first_hit_rank'] is not None else '—'} |"
        )
    lines.append("")

    t = result["timings"]
    lines += [
        "## Performance",
        "",
        f"- Model load: {_s(t['model_load_s'])}; library import {_s(t.get('library_import_s'))}.",
        f"- Passages: {t['passages_total']} windows over {result['corpus']['embedded_documents']} documents, embedded in "
        f"{_s(t['passage_embed_total_s'])} ({t['passage_embed_ms_per_passage']:.1f} ms per window; per document median "
        f"{_ms(t['passage_embed_ms_per_document']['median'])}, max {_ms(t['passage_embed_ms_per_document']['max'])}).",
        f"- Warm localization (query embedding + scoring one document's windows): median "
        f"{_ms(t['warm_localization_ms']['median'])}, mean {_ms(t['warm_localization_ms']['mean'])}; query embedding median "
        f"{_ms(t['query_embed_ms']['median'])}.",
        f"- Cold localization of one document (embed its windows + warm): median {_ms(t['cold_document_localization_ms']['median'])}, "
        f"max {_ms(t['cold_document_localization_ms']['max'])}.",
        f"- Lexical localize(): median {_ms(t['lexical_localization_ms']['median'])}.",
        "- Peak RSS (MB, monotonic): " + ", ".join(f"{k} {_mb(v)}" for k, v in result["memory_peak_rss_mb"].items()),
        "",
        "## Method",
        "",
    ]
    config = result["config"]
    lines += [
        f"- Windows: {config['window_lines']} lines, stride {config['stride']}, {config['line_semantics']}; {config['blank_window_rule']}.",
        f"- Scoring: {config['similarity']}; ties: {config['tie_break']}. Query format: `{config['query_format_example']!r}`; passages embedded as-is.",
        f"- Excerpts ({', '.join(str(b) for b in config['excerpt_budgets'])} characters): {config['excerpt_rule']}. "
        f"The same rule is applied to both localizers' passages; lexical's native excerpt is reported separately.",
        f"- Document under test: {config['document_under_test']}.",
        f"- Embedder: {_kv({k: v for k, v in config['embedder'].items() if k != 'packages'})}",
        f"- Packages: {_kv(config['embedder'].get('packages') or {})}",
        f"- Environment: {_kv(result['environment'])}",
    ]
    if result["drift"]:
        lines.append("- Drifted cases (not scored): " + ", ".join(f"{d['id']} ({d['reason']})" for d in result["drift"]))
    lines.append("")
    return "\n".join(lines)


def render_summary(result: dict[str, Any], *, paths: dict[str, str] | None = None) -> str:
    lexical, semantic = result["metrics"]["lexical"], result["metrics"]["semantic"]
    n = lexical["overall"]["n"]
    embedder = result["config"]["embedder"]
    t = result["timings"]
    lines = [
        f"T3e evidence localization — {n} cases, benchmark document given; embedder "
        f"{embedder.get('model_id', embedder.get('embedder'))}",
    ]
    if embedder.get("embedder") == "fake-hashing":
        lines.append("  FAKE EMBEDDER: pipeline check only")
    ref = result["lexical_reference"]
    lines.append(
        "  lexical reference: " + ("matches" if ref.get("checked") else "not checked") + f" ({_categories(ref['recomputed'])})"
    )
    columns = (("line_overlap", "overlap", 8), ("passage_contains_anchor", "passage", 9), ("excerpt_80_survives", "80", 7),
               ("excerpt_120_survives", "120", 7), ("excerpt_160_survives", "160", 7))
    lines += ["", f"  {'':<19}" + "".join(f"{title:>{width}}" for _, title, width in columns) + f"{'native':>8}"]
    for name, metrics in (("lexical", lexical), ("semantic", semantic)):
        for scope, block in [("overall", metrics["overall"]), *metrics["by_category"].items()]:
            hits, count = block["hits"], block["n"]
            native = f"{hits['native_excerpt_survives']}/{count}" if "native_excerpt_survives" in hits else "—"
            label = f"{name} overall" if scope == "overall" else f"  {scope}"
            lines.append(
                f"  {label:<19}" + "".join(f"{f'{hits[key]}/{count}':>{width}}" for key, _, width in columns) + f"{native:>8}"
            )
    for measure, buckets in result["pairwise"].items():
        lines.append(
            f"  pairwise {measure}: both {len(buckets['both_hit'])}, lexical only {len(buckets['lexical_only'])}, "
            f"semantic only {len(buckets['semantic_only'])}, both miss {len(buckets['both_miss'])}"
        )
    lines.append(
        f"  semantic first-hit rank: {_histogram(result['top3']['first_hit_rank_histogram'])}; "
        f"evidence in top-3 windows {result['top3']['hit_in_top3']}/{n}"
    )
    lines.append(
        f"  timing: model load {_s(t['model_load_s'])}, {t['passages_total']} windows embedded in {_s(t['passage_embed_total_s'])}, "
        f"warm localization median {_ms(t['warm_localization_ms']['median'])}, cold per document median "
        f"{_ms(t['cold_document_localization_ms']['median'])}, lexical median {_ms(t['lexical_localization_ms']['median'])}"
    )
    for label, path in (paths or {}).items():
        lines.append(f"  {label + ':':9} {path}")
    return "\n".join(lines)


def _case_detail(row: dict[str, Any]) -> list[str]:
    lex, sem = row["lexical"], row["semantic"]
    return [
        f"#### {row['id']} ({row['category']})",
        "",
        f"- Query: {row['query']}",
        f"- Document: `{row['expected_path']}`; evidence lines {row['anchor_lines'][0]}-{row['anchor_lines'][1]}",
        f"- Lexical lines {lex['passage']['start_line']}-{lex['passage']['end_line']}: {_flat(lex['native_excerpt']['text'])}",
        f"- Semantic lines {sem['passage']['start_line']}-{sem['passage']['end_line']} (similarity {sem['similarity']:.3f}, "
        f"first hit rank {sem['first_hit_rank'] if sem['first_hit_rank'] is not None else '—'}): {_flat(sem['excerpts']['160']['text'])}",
        "",
    ]


def _anchor_lengths(result: dict[str, Any]) -> str:
    lengths = [row["anchor_length"] for row in result["cases"]]
    return ", ".join(f"{sum(1 for x in lengths if x > b)} over {b}" for b in result["config"]["excerpt_budgets"])


def _categories(recomputed: dict[str, Any]) -> str:
    return ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in recomputed.items())


def _histogram(histogram: dict[str, int]) -> str:
    order = {"1": 0, "2": 1, "3": 2}
    return ", ".join(f"{k}: {v}" for k, v in sorted(histogram.items(), key=lambda kv: order.get(kv[0], 9))) or "none"


def _flat(text: str) -> str:
    return " ".join(text.split())


def _frac(count: int, n: int) -> str:
    return f"{count}/{n}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def _s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} s"


def _mb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def _kv(mapping: dict[str, Any]) -> str:
    return "; ".join(f"{k}={v}" for k, v in mapping.items())
