"""Render a T3f projection/fusion result as Markdown and as a paste-able terminal summary."""

from __future__ import annotations

from typing import Any

from embedding_bench.localization_fusion import CONTROL, FUSION, METHODS, PACKET_TOTALS, PROJECTION_BUDGETS, SOURCES

METHOD_LABEL = {"leading": "leading (T3e)", "query_aware": "query-aware"}
STRATEGY_LABEL = {FUSION: "fusion (semantic primary, dual on disagreement)", CONTROL: "semantic only (control)"}


def render_markdown(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    n = result["input"]["cases"]
    source = result["input"]
    lines = [
        "# T3f: deterministic evidence projection + bounded localization fusion",
        "",
        f"{result['experiment']}. Created {result['created_at']} from the T3e artifact `{source['t3e_results']}` "
        f"(sha256 `{source['t3e_sha256']}`, T3e run started {source['t3e_started_at']}, embedder "
        f"{_embedder(source['t3e_embedder'])}). No model was loaded and nothing was re-embedded. Every projection and "
        f"packet was frozen (decisions sha256 `{result['decisions_sha256'][:16]}…`) before any category or anchor was read.",
        "",
    ]
    if _is_fake(source["t3e_embedder"]):
        lines += ["> **FAKE EMBEDDER IN THE T3e INPUT.** Pipeline check only; the semantic passages are not Qwen3-Embedding's.", ""]
    reference = result["t3e_reference"]
    if reference.get("checked"):
        lines += ["T3e reference check: the baselines recomputed from the artifact match the accepted live run.", ""]
    else:
        lines += ["> **T3e REFERENCE NOT CHECKED.** The recomputed baselines were not compared with the accepted live run; "
                  "these numbers may not describe it.", ""]

    headroom = metrics["headroom"]
    lines += [
        "## Selection headroom (from the frozen T3e selections)",
        "",
        f"- Top-1 passage oracle (lexical or semantic lines overlap the evidence): {headroom['top1_oracle_line_overlap']}/{n}",
        f"- Semantic top-1: {headroom['semantic_top1_line_overlap']}/{n}; lexical top-1: {headroom['lexical_top1_line_overlap']}/{n}",
        f"- Semantic top-3 windows: {headroom['semantic_top3_line_overlap']}/{n}",
        "",
        "## Projection of the same selected passage",
        "",
        "Each localizer's frozen T3e passage is projected twice: the T3e leading-character excerpt and the query-aware "
        "projection. Only the projection differs.",
        "",
    ]
    for title, key, fmt in (("Evidence survives", "survives", _count(n)), ("Whole anchor contained", "contains_anchor", _count(n)),
                            ("Mean share of the anchor shown", "mean_coverage", _pct)):
        lines += [f"### {title}", "", "| passage | projection | " + " | ".join(str(b) for b in PROJECTION_BUDGETS) + " |",
                  "|---|---|" + "---:|" * len(PROJECTION_BUDGETS)]
        for src in SOURCES:
            for method in METHODS:
                cells = [fmt(metrics["projection"][src][method][str(b)]["overall"][key]) for b in PROJECTION_BUDGETS]
                lines.append(f"| {src} | {METHOD_LABEL[method]} | " + " | ".join(cells) + " |")
        native = metrics["native_excerpt"]["overall"][key]
        lines += [f"| lexical | localize() native excerpt (≤160) | — | — | {fmt(native)} |", ""]

    retention = metrics["retention"]
    sem = retention["semantic"]
    qa160 = metrics["projection"]["semantic"]["query_aware"]["160"]["overall"]["survives"]
    lines += [
        "### How much of the selected-passage success survives compression",
        "",
        f"- Semantic passage line hit: {sem['line_overlap']}/{n} (passage holds the whole anchor: {sem['passage_contains_anchor']}/{n}).",
        f"- Semantic query-aware 160 excerpt, evidence survives: {qa160}/{n}.",
    ]
    for src in SOURCES:
        block = retention[src]
        for method in METHODS:
            counts = ", ".join(f"{b}: {block['survives_among_line_hits'][method][str(b)]}" for b in PROJECTION_BUDGETS)
            lines.append(f"- {src} {METHOD_LABEL[method]}, evidence survives among its {block['line_overlap']} line hits: {counts}.")
    lines.append("")

    for src in SOURCES:
        lines += [f"### {src.capitalize()} passage, evidence survives by category", "",
                  "| category | n | " + " | ".join(f"{'lead' if m == 'leading' else 'query'} {b}" for b in PROJECTION_BUDGETS for m in METHODS) + " |",
                  "|---|---:|" + "---:|" * (len(PROJECTION_BUDGETS) * len(METHODS))]
        for category in result["categories"]:
            blocks = [metrics["projection"][src][m][str(b)]["by_category"][category] for b in PROJECTION_BUDGETS for m in METHODS]
            lines.append(f"| {category} | {blocks[0]['n']} | " + " | ".join(str(b["survives"]) for b in blocks) + " |")
        lines.append("")

    lines += ["### Leading vs query-aware, case by case (evidence survives)", ""]
    for src in SOURCES:
        for b in PROJECTION_BUDGETS:
            buckets = metrics["projection_pairwise"][src][str(b)]
            lines.append(
                f"- **{src} {b}:** both hit {len(buckets['both_hit'])}, leading only {len(buckets['leading_only'])}"
                f"{_ids(buckets['leading_only'])}, query-aware only {len(buckets['query_aware_only'])}"
                f"{_ids(buckets['query_aware_only'])}, both miss {len(buckets['both_miss'])}."
            )
    anchors = metrics["anchors"]
    lines += [
        "",
        "No query phrase in the passage, so the query-aware projection is centred on it: "
        + "; ".join(
            f"{src} {anchors[src]['center_fallback']}/{n} ("
            + ", ".join(f"{c} {v}" for c, v in anchors[src]["center_fallback_by_category"].items()) + ")"
            for src in SOURCES
        )
        + ".",
        "",
        "Mean source characters shown: "
        + "; ".join(
            f"{src} {METHOD_LABEL[m]} "
            + "/".join(f"{metrics['projection'][src][m][str(b)]['source_chars']['mean']:.1f}" for b in PROJECTION_BUDGETS)
            for src in SOURCES for m in METHODS
        )
        + f" (budgets {'/'.join(str(b) for b in PROJECTION_BUDGETS)}).",
        "",
        "## Evidence packets",
        "",
        "Fusion keeps the semantic passage first and adds the lexical passage only when the two selected line ranges are "
        "disjoint, splitting the total budget evenly; otherwise the semantic passage gets the whole budget. The control "
        "spends the same total budget on the semantic passage alone. Packet contents are query-aware projections; "
        "\"survives\" is scored on the union of emitted source spans.",
        "",
        "| packet | total | survives | whole anchor | mean coverage | lines overlap | dual | source chars mean / median / max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in (FUSION, CONTROL):
        for total in PACKET_TOTALS:
            block = metrics["packets"][strategy][str(total)]
            overall, chars = block["overall"], block["source_chars"]
            lines.append(
                f"| {STRATEGY_LABEL[strategy]} | {total} | {overall['survives']}/{n} | {overall['contains_anchor']}/{n} | "
                f"{_pct(overall['mean_coverage'])} | {block['line_overlap']}/{n} | {block['dual_cases']}/{n} | "
                f"{chars['mean']:.1f} / {chars['median']:.1f} / {chars['max']} |"
            )
    lines += ["", "Same total budget, fusion vs control (evidence survives):", ""]
    for total in PACKET_TOTALS:
        buckets = metrics["packet_pairwise"][str(total)]
        fusion, control = metrics["packets"][FUSION][str(total)], metrics["packets"][CONTROL][str(total)]
        lines.append(
            f"- **{total}:** both hit {len(buckets['both_hit'])}, fusion only {len(buckets['fusion_only'])}"
            f"{_ids(buckets['fusion_only'])}, control only {len(buckets['semantic_only_only'])}"
            f"{_ids(buckets['semantic_only_only'])}, both miss {len(buckets['both_miss'])}. Dual evidence in "
            f"{fusion['dual_cases']}/{n} cases ({_pct(fusion['dual_cases'] / n if n else None)}); mean source characters "
            f"{fusion['source_chars']['mean']:.1f} vs {control['source_chars']['mean']:.1f}."
        )
    lines += ["", "### Packets, evidence survives by category", "",
              "| category | n | " + " | ".join(f"{'fusion' if s == FUSION else 'control'} {t}" for t in PACKET_TOTALS for s in (FUSION, CONTROL)) + " |",
              "|---|---:|" + "---:|" * (len(PACKET_TOTALS) * 2)]
    for category in result["categories"]:
        blocks = [metrics["packets"][s][str(t)]["by_category"][category] for t in PACKET_TOTALS for s in (FUSION, CONTROL)]
        lines.append(f"| {category} | {blocks[0]['n']} | " + " | ".join(str(b["survives"]) for b in blocks) + " |")

    lines += [
        "",
        "## Cases",
        "",
        "| id | category | semantic anchor | sem lead 160 | sem query 160 | lex lead 160 | lex query 160 | dual | fusion 320 | control 320 |",
        "|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|",
    ]
    for row in result["cases"]:
        ev = row["evaluation"]
        anchor = row["decision"]["anchors"]["semantic"]
        marks = [
            ev["projection"]["semantic"]["leading"]["160"], ev["projection"]["semantic"]["query_aware"]["160"],
            ev["projection"]["lexical"]["leading"]["160"], ev["projection"]["lexical"]["query_aware"]["160"],
        ]
        lines.append(
            f"| {row['id']} | {row['category']} | {'match, line ' + str(anchor['line']) if anchor['mode'] == 'match' else 'centre'} | "
            + " | ".join(_mark(m["survives"]) for m in marks)
            + f" | {'yes' if ev['packets'][FUSION]['320']['dual'] else ''} | {_mark(ev['packets'][FUSION]['320']['survives'])} | "
            f"{_mark(ev['packets'][CONTROL]['320']['survives'])} |"
        )
    lines.append("")

    changed = [
        row for row in result["cases"]
        if row["evaluation"]["projection"]["semantic"]["leading"]["160"]["survives"]
        != row["evaluation"]["projection"]["semantic"]["query_aware"]["160"]["survives"]
    ]
    if changed:
        lines += ["### Semantic 160: cases where the projection changes the outcome", ""]
        for row in changed:
            projections = row["decision"]["projections"]["semantic"]
            lines += [
                f"#### {row['id']} ({row['category']})",
                "",
                f"- Query: {row['query']}",
                f"- Evidence lines {row['anchor_lines'][0]}-{row['anchor_lines'][1]}; semantic passage lines "
                f"{row['decision']['passages']['semantic']['start_line']}-{row['decision']['passages']['semantic']['end_line']}",
                f"- Leading ({_mark(row['evaluation']['projection']['semantic']['leading']['160']['survives'])}): "
                f"{_flat(projections['leading']['160']['text'])}",
                f"- Query-aware ({_mark(row['evaluation']['projection']['semantic']['query_aware']['160']['survives'])}): "
                f"{_flat(projections['query_aware']['160']['text'])}",
                "",
            ]

    config = result["config"]
    lines += [
        "## Method",
        "",
        f"- Leading projection: {config['leading_projection']}.",
        f"- Query-aware projection: {config['query_aware_projection']}. Leading context: {config['lead_share']}.",
        f"- Query matching: {config['query_matcher']}.",
        f"- Fusion: {config['fusion'][FUSION]}. Control: {config['fusion'][CONTROL]}.",
        f"- Survives: {config['survives_definition']}.",
        f"- Labels: {config['labels']}.",
        "- Document under test: the benchmark's expected document, as in T3e (isolates localization from retrieval).",
        "",
    ]
    return "\n".join(lines)


def render_summary(result: dict[str, Any], *, paths: dict[str, str] | None = None) -> str:
    metrics = result["metrics"]
    n = result["input"]["cases"]
    headroom = metrics["headroom"]
    reference = result["t3e_reference"]
    lines = [f"T3f evidence projection + packet fusion — {n} cases from {result['input']['t3e_results']}"]
    if _is_fake(result["input"]["t3e_embedder"]):
        lines.append("  FAKE EMBEDDER IN THE T3e INPUT: pipeline check only")
    lines.append("  T3e reference: " + ("matches the accepted live run" if reference.get("checked") else "NOT CHECKED"))
    lines += [
        f"  headroom: top-1 oracle {headroom['top1_oracle_line_overlap']}/{n}, semantic top-1 "
        f"{headroom['semantic_top1_line_overlap']}/{n}, semantic top-3 {headroom['semantic_top3_line_overlap']}/{n}",
        "",
        f"  {'evidence survives':<30}" + "".join(f"{b:>8}" for b in PROJECTION_BUDGETS),
    ]
    for src in SOURCES:
        for method in METHODS:
            cells = "".join(
                f"{str(metrics['projection'][src][method][str(b)]['overall']['survives']) + '/' + str(n):>8}"
                for b in PROJECTION_BUDGETS
            )
            lines.append(f"  {src + ' ' + METHOD_LABEL[method]:<30}{cells}")
    sem = metrics["retention"]["semantic"]
    lines += [
        f"  semantic passage line hit {sem['line_overlap']}/{n}; among them survives at 160: leading "
        f"{sem['survives_among_line_hits']['leading']['160']}, query-aware {sem['survives_among_line_hits']['query_aware']['160']}",
        "",
        f"  {'packet':<24}{'total':>6}{'survives':>10}{'dual':>7}{'chars mean/med/max':>22}",
    ]
    for strategy in (FUSION, CONTROL):
        for total in PACKET_TOTALS:
            block = metrics["packets"][strategy][str(total)]
            chars = block["source_chars"]
            cost = f"{chars['mean']:.1f}/{chars['median']:.1f}/{chars['max']}"
            lines.append(
                f"  {'fusion' if strategy == FUSION else 'semantic only (control)':<24}{total:>6}"
                f"{str(block['overall']['survives']) + '/' + str(n):>10}{str(block['dual_cases']) + '/' + str(n):>7}{cost:>22}"
            )
    for label, path in (paths or {}).items():
        lines.append(f"  {label + ':':9} {path}")
    return "\n".join(lines)


def _embedder(embedder: Any) -> str:
    if not isinstance(embedder, dict):
        return "unknown"
    return str(embedder.get("model_id") or embedder.get("embedder") or "unknown")


def _is_fake(embedder: Any) -> bool:
    return isinstance(embedder, dict) and embedder.get("embedder") == "fake-hashing"


def _count(n: int):
    return lambda value: f"{value}/{n}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _mark(hit: bool) -> str:
    return "✓" if hit else "·"


def _ids(ids: list[str]) -> str:
    return f" ({', '.join(ids)})" if ids else ""


def _flat(text: str) -> str:
    return " ".join(text.split())
