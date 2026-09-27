"""Render a T3b fusion result dict as Markdown and as a compact terminal summary."""

from __future__ import annotations

from typing import Any

SYSTEM_LABELS = {"lexical": "lexical", "semantic": "semantic", "rrf": "RRF"}
_TRANSITIONS = (  # (bucket, report title, terminal label, verb)
    ("both_correct", "Both correct in T3a", "both-correct", "kept"),
    ("lexical_correct_semantic_wrong", "Lexical-only top-1 wins", "lexical-only", "kept"),
    ("semantic_correct_lexical_wrong", "Semantic-only top-1 wins", "semantic-only", "kept"),
    ("both_wrong", "Both wrong in T3a", "both-wrong", "recovered"),
)


def render_markdown(result: dict[str, Any]) -> str:
    source = result["input"]
    check = result["manifest_check"]
    config = result["config"]
    embedder = source["t3a_embedder"]
    lines = [
        "# T3b: Reciprocal Rank Fusion over frozen T3a rankings",
        "",
        f"{result['experiment']}. Generated {result['created_at']} from recorded rankings only; "
        "no model was loaded and nothing was re-embedded.",
        "",
    ]
    if source["fake_embedder"]:
        lines += [
            "> **FAKE EMBEDDER INPUT.** The T3a run used the hashing stand-in, so the semantic "
            "and RRF numbers below say nothing about Qwen3-Embedding.",
            "",
        ]

    lines += ["## Metrics", "", *_metrics_table(result), ""]
    oracle = result["oracle_top1_union"]
    rrf_top1 = sum(1 for case in result["cases"] if case["top1"]["rrf"])
    lines += [
        f"RRF ranks the expected document first in **{rrf_top1}/{oracle['n']}** cases. "
        f"Reference only: the top-1 union of the two inputs (either one first) is "
        f"{oracle['correct']}/{oracle['n']} ({_pct(oracle['rate'])}); that is an oracle, not a system.",
        "",
    ]

    lines += ["## Top-1 against the T3a agreement buckets", ""]
    for bucket, title, _, verb in _TRANSITIONS:
        data = result["top1_transitions"][bucket]
        line = f"- **{title}:** {verb} {len(data['rrf_correct'])}/{data['n']}"
        if verb == "recovered" and data["rrf_correct"]:
            line += f" ({_ids(data['rrf_correct'])})"
        if data["rrf_wrong"]:
            line += f"; {'lost' if verb == 'kept' else 'not recovered'}: {_ids(data['rrf_wrong'])}"
        lines.append(line)
    lines.append("")

    rank_vs = result["rank_vs_inputs"]
    top3 = result["top3_lost"]
    ties = result["ties"]
    lines += [
        "## Fused rank of the expected document",
        "",
        f"- **Better than both inputs** ({len(rank_vs['better_than_both'])}): {_ids(rank_vs['better_than_both'])}",
        f"- **Worse than both inputs** ({len(rank_vs['worse_than_both'])}): {_ids(rank_vs['worse_than_both'])}",
        f"- Matches the better input: {len(rank_vs['matches_best'])}; between the inputs: {len(rank_vs['between'])}",
        f"- Top-3 lost relative to lexical: {_ids(top3['vs_lexical'])}",
        f"- Top-3 lost relative to semantic: {_ids(top3['vs_semantic'])}",
        f"- **RRF top-1 decided by the path tie-break** ({len(ties['rrf_top1_decided_by_path'])}): "
        f"{_ids(ties['rrf_top1_decided_by_path'])}",
        f"- Expected document's fused rank decided by the path tie-break: {_ids(ties['expected_rank_decided_by_path'])}",
        "",
    ]

    misses = [case for case in result["cases"] if not case["top1"]["rrf"]]
    if misses:
        lines += ["## Cases where RRF does not rank the expected document first", ""]
        for case in misses:
            lines += _case_detail(case)

    lines += ["## Per-query ranks", ""]
    lines += [
        "| id | category | T3a agreement | lexical | semantic | RRF | RRF top-1 | note |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for case in result["cases"]:
        ranks = case["ranks"]
        notes = []
        if case["rrf_vs_inputs"] in ("better_than_both", "worse_than_both"):
            notes.append(case["rrf_vs_inputs"].replace("_", " "))
        if case["rrf_top1_decided_by_path"]:
            notes.append("top-1 tie")
        lines.append(
            f"| {case['id']} | {case['category']} | {case['t3a_agreement']} | {_rank(ranks['lexical'])} | "
            f"{_rank(ranks['semantic'])} | {_rank(ranks['rrf'])} | {case['rrf_ranking'][0]['path']} | "
            f"{'; '.join(notes) or ''} |"
        )
    lines.append("")

    lines += [
        "## Method",
        "",
        f"- {config['formula']}; a document absent from one ranking gets {config['missing_rank_contribution']} "
        "from that side.",
        f"- k = {config['k']} (standard, fixed, not tuned); equal weights; {config['inputs']}.",
        f"- Scores use {config['arithmetic']}; ties: {config['tie_break']}.",
        f"- Labels and categories: {config['labels_used_for']}.",
        f"- Input: `{source['path']}` (sha256 `{(source['sha256'] or 'n/a')[:12]}`), T3a run {source['t3a_started_at']}, "
        f"embedder {embedder.get('model_id', embedder.get('embedder'))}, ZOMAH commit "
        f"{(source['t3a_zomah_commit'] or 'n/a')[:10]}, {source['corpus_documents']} documents.",
        f"- Manifest `{check['path']}`: labels match the T3a cases; sha256 "
        + ("matches the T3a run." if check["sha256_matches"] else "differs from the T3a run (non-label text changed).")
        + (f" Not scored in T3a (drift): {', '.join(check['not_scored_in_t3a'])}." if check["not_scored_in_t3a"] else ""),
        "",
    ]
    return "\n".join(lines)


def render_summary(result: dict[str, Any], *, json_path: str | None = None, markdown_path: str | None = None) -> str:
    source = result["input"]
    embedder = source["t3a_embedder"]
    lines = [
        f"T3b RRF (k={result['config']['k']}) over frozen T3a rankings — embedder "
        f"{embedder.get('model_id', embedder.get('embedder'))}, {len(result['cases'])} cases",
    ]
    if source["fake_embedder"]:
        lines.append("  FAKE EMBEDDER INPUT: pipeline check only")
    lines += ["", *_metrics_table(result, markdown=False), ""]
    transitions = result["top1_transitions"]
    lines.append(
        "  top-1 vs T3a: "
        + ", ".join(
            f"{short} {verb} {len(transitions[bucket]['rrf_correct'])}/{transitions[bucket]['n']}"
            for bucket, _, short, verb in _TRANSITIONS
        )
    )
    oracle = result["oracle_top1_union"]
    rrf_top1 = sum(1 for case in result["cases"] if case["top1"]["rrf"])
    lines.append(f"  RRF top-1 {rrf_top1}/{oracle['n']}; oracle union (reference only) {oracle['correct']}/{oracle['n']}")
    rank_vs = result["rank_vs_inputs"]
    lines.append(
        f"  expected doc vs both inputs: better {_ids(rank_vs['better_than_both'])}; "
        f"worse {_ids(rank_vs['worse_than_both'])}"
    )
    lines.append(f"  top-1 decided by path tie-break: {_ids(result['ties']['rrf_top1_decided_by_path'])}")
    if json_path:
        lines.append(f"  json:     {json_path}")
    if markdown_path:
        lines.append(f"  markdown: {markdown_path}")
    return "\n".join(lines)


def _metrics_table(result: dict[str, Any], *, markdown: bool = True) -> list[str]:
    metrics = result["metrics"]
    slices = ["overall", *result["metrics"]["rrf"]["by_category"]]

    def values(system: str, name: str) -> dict[str, Any]:
        data = metrics[system]
        return data["overall"] if name == "overall" else data["by_category"][name]

    if markdown:
        lines = ["| slice | system | n | top-1 | top-3 | MRR |", "|---|---|---:|---:|---:|---:|"]
        for name in slices:
            for system, label in SYSTEM_LABELS.items():
                m = values(system, name)
                lines.append(
                    f"| {name} | {label} | {m['n']} | {_pct(m['top1'])} | {_pct(m['top3'])} | {_f(m['mrr'])} |"
                )
        return lines

    header = f"  {'slice':11} {'n':>3}" + "".join(f"   {label + ' top1/top3/MRR':>22}" for label in SYSTEM_LABELS.values())
    lines = [header]
    for name in slices:
        row = f"  {name:11} {values('rrf', name)['n']:>3}"
        for system in SYSTEM_LABELS:
            m = values(system, name)
            row += f"   {_pct(m['top1']):>6} {_pct(m['top3']):>6} {_f(m['mrr']):>6}  "
        lines.append(row.rstrip())
    return lines


def _case_detail(case: dict[str, Any]) -> list[str]:
    ranks = case["ranks"]
    lines = [
        f"### {case['id']} ({case['category']}, T3a {case['t3a_agreement']})",
        "",
        f"- Query: {case['query']}",
        f"- Expected: `{case['expected_path']}` — lexical {_rank(ranks['lexical'])}, semantic "
        f"{_rank(ranks['semantic'])}, RRF {_rank(ranks['rrf'])} (score {case['rrf_expected_score']:.6f})",
    ]
    if case["rrf_top1_decided_by_path"]:
        lines.append("- RRF top-1 is an exact score tie broken by path.")
    lines.append("- RRF top-3:")
    for position, doc in enumerate(case["rrf_ranking"][:3], start=1):
        lines.append(
            f"  {position}. `{doc['path']}` — {doc['rrf']:.6f} "
            f"(lexical {_rank(doc['lexical_rank'])}, semantic {_rank(doc['semantic_rank'])})"
        )
    lines.append("")
    return lines


def _ids(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "none"


def _rank(rank: int | None) -> str:
    return "—" if rank is None else str(rank)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
