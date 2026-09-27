"""Render a T3c candidate-audit result dict as Markdown and as a terminal summary."""

from __future__ import annotations

from typing import Any

_OUTCOME_TITLES = {
    "resolved_correct": "resolved correctly",
    "resolved_wrong_in_union": "resolved to the wrong document although the expected one was a candidate",
    "resolved_retrieval_failure": "resolved, but the expected document was not a candidate (retrieval failure)",
    "ambiguous_in_union": "ambiguous, expected document is a candidate (bounded-judgment scope)",
    "ambiguous_retrieval_failure": "ambiguous, expected document is not a candidate (retrieval failure)",
}
_ROLES = {
    "lexical_top1": "lexical #1",
    "semantic_top1": "semantic #1",
    "other_candidate": "another candidate",
    "not_in_union": "not a candidate",
}
_SHORT = {
    "both_correct": "both-correct",
    "lexical_correct_semantic_wrong": "lexical-only",
    "semantic_correct_lexical_wrong": "semantic-only",
    "both_wrong": "both-wrong",
}
_BUCKETS = {
    "both_correct": "Both correct in T3a",
    "lexical_correct_semantic_wrong": "Lexical-only T3a wins",
    "semantic_correct_lexical_wrong": "Semantic-only T3a wins",
    "both_wrong": "Both wrong in T3a",
}


def render_markdown(result: dict[str, Any]) -> str:
    source = result["input"]
    config = result["config"]
    recall = result["candidate_recall"]
    size = result["candidate_size"]
    resolution = result["resolution"]
    ambiguous = result["ambiguous"]
    refs = result["end_to_end_references"]
    n = refs["n"]
    counts = {kind: len(ids) for kind, ids in resolution["counts"].items()}
    accuracy = resolution["resolved_accuracy"]
    failures = result["retrieval_failures"]

    lines = [
        "# T3c: candidate union and deterministic ambiguity audit",
        "",
        f"{result['experiment']}. Generated {result['created_at']} from recorded T3a rankings only; "
        "no model was loaded, nothing was re-embedded or re-retrieved.",
        "",
    ]
    if source["fake_embedder"]:
        lines += [
            "> **FAKE EMBEDDER INPUT.** The T3a run used the hashing stand-in, so these numbers say "
            "nothing about Qwen3-Embedding.",
            "",
        ]

    lines += [
        "## Decision inputs",
        "",
        f"- **Union top-{config['top_n']} candidate recall:** {_frac(recall['overall']['union'], n)}.",
        f"- **Deterministically resolved:** {accuracy['n']}/{n} cases, {accuracy['correct']} correct "
        f"({_pct(accuracy['rate'])} of resolutions).",
        f"- **Ambiguous:** {ambiguous['n']}/{n} cases; the expected document is a candidate in "
        f"{ambiguous['expected_in_union']} of them.",
        f"- **Retrieval failures** (expected document not a candidate; no judge can repair these): "
        f"{len(failures)} — {_ids(failures)}.",
        "",
        "## Resolution classes",
        "",
        "Classes come from the two rankings alone. Correctness is evaluated afterwards against the labels.",
        "",
        f"- **AGREEMENT** — {counts['agreement']} cases, {resolution['resolved_accuracy']['by_kind']['agreement']['correct']} correct.",
        f"- **DOMINANCE** (not agreement; exactly one candidate dominates every other) — {counts['dominance']} cases, "
        f"{resolution['resolved_accuracy']['by_kind']['dominance']['correct']} correct.",
        f"- **AMBIGUOUS** — {counts['ambiguous']} cases, expected document a candidate in {ambiguous['expected_in_union']}.",
        f"- **RETRIEVAL_FAILURE** (evaluation only, across classes) — {len(failures)} cases.",
        "",
        "Evaluation outcome of every case:",
        "",
    ]
    for outcome, ids in result["outcomes"].items():
        lines.append(f"- {_OUTCOME_TITLES[outcome]}: {len(ids)} — {_ids(ids)}")
    lines += [
        "",
        "Why dominance adds no resolutions beyond agreement here: the union always contains lexical #1 and "
        "semantic #1. A candidate that dominates every other must rank at least as well as lexical #1 on the "
        "lexical ranking and as semantic #1 on the semantic ranking, so it must be #1 on both, which is agreement. "
        "Unique dominance can differ from agreement only when one ranking returns nothing. Dominance still "
        "prunes: the non-dominated candidates (the Pareto front) are reported below as the smallest set a "
        "judge would need to see.",
        "",
    ]

    lines += ["## Candidate recall", "", f"| slice | n | lexical top-{config['top_n']} | semantic top-{config['top_n']} | union |", "|---|---:|---:|---:|---:|"]
    for name, data in [("overall", recall["overall"]), *recall["by_category"].items()]:
        lines.append(
            f"| {name} | {data['n']} | {_frac(data['lexical_top'], data['n'])} | "
            f"{_frac(data['semantic_top'], data['n'])} | {_frac(data['union'], data['n'])} |"
        )
    lines.append("")

    lines += [
        "## Candidate-set size",
        "",
        f"Mean {_num(size['mean'])}, median {_num(size['median'])}, min {size['min']}, max {size['max']}. "
        f"Distribution: {_histogram(size['histogram'])}.",
        "",
        "| category | n | mean size | agreement | dominance | ambiguous |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in size["by_category"].items():
        by = resolution["by_category"][name]
        lines.append(
            f"| {name} | {stats['n']} | {_num(stats['mean'])} | {by['agreement']} | {by['dominance']} | {by['ambiguous']} |"
        )
    lines.append("")

    lines += ["## T3a top-1 wins inside the union", ""]
    for bucket, title in _BUCKETS.items():
        data = result["t3a_wins_in_union"][bucket]
        classes = "; ".join(f"{kind} {len(ids)}" for kind, ids in data["resolution"].items() if ids)
        lines.append(f"- **{title}:** {data['in_union']}/{data['n']} in the union ({classes or 'none'}).")
    lines += [
        "",
        "A T3a win means one retriever ranked the expected document first, and each retriever's #1 is always "
        "in the union, so every lexical-only and semantic-only win is a candidate by construction.",
        "",
    ]

    selectors = ambiguous["selector_diagnostics"]
    front = ambiguous["pareto_front_size"]
    lines += [
        "## Ambiguous cases",
        "",
        f"{ambiguous['n']} cases. Expected document is "
        + ", ".join(f"{_ROLES[role]} in {len(ids)}" for role, ids in ambiguous["expected_is"].items())
        + ".",
        "",
        f"- Candidate-set size: mean {_num(ambiguous['candidate_size']['mean'])}, max {ambiguous['candidate_size']['max']}.",
        f"- Pareto front (non-dominated candidates): mean {_num(front['mean'])}, max {front['max']}, "
        f"distribution {_histogram(front['histogram'])}; expected document on the front in "
        f"{ambiguous['expected_on_pareto_front']}/{ambiguous['n']}.",
        "- Simple selectors on these cases (diagnostics, not resolvers): "
        f"lexical #1 {_frac(selectors['lexical_top1'], selectors['lexical_top1']['n'])}, "
        f"semantic #1 {_frac(selectors['semantic_top1'], selectors['semantic_top1']['n'])}, "
        f"RRF #1 {_frac(selectors['rrf_top1'], selectors['rrf_top1']['n'])}.",
        "",
        f"End-to-end reference counts out of {n} (evaluation only): deterministic resolutions plus a perfect "
        f"judge {refs['deterministic_plus_perfect_judge']}; plus always lexical #1 "
        f"{refs['deterministic_plus_lexical_top1']}; plus always semantic #1 {refs['deterministic_plus_semantic_top1']}; "
        f"T3b RRF {refs['rrf_top1']}.",
        "",
    ]
    for case in result["cases"]:
        if case["resolution"] == "ambiguous":
            lines += _case_detail(case)

    wrong = [c for c in result["cases"] if c["evaluation"]["outcome"] == "resolved_wrong_in_union"]
    failed = [c for c in result["cases"] if not c["evaluation"]["expected_in_union"]]
    if wrong:
        lines += ["## Deterministic resolutions that picked the wrong candidate", ""]
        for case in wrong:
            lines += _case_detail(case)
    lines += ["## Retrieval failures", ""]
    if failed:
        lines += [
            "The expected document is in neither retriever's top "
            f"{config['top_n']}. Improve retrieval; a judge over this candidate set cannot fix these.",
            "",
        ]
        for case in failed:
            lines += _case_detail(case)
    else:
        lines += ["None.", ""]

    lines += [
        "## Per-query results",
        "",
        "| id | category | candidates | resolution | outcome | expected lexical | expected semantic | expected is |",
        "|---|---|---:|---|---|---:|---:|---|",
    ]
    for case in result["cases"]:
        ev = case["evaluation"]
        lines.append(
            f"| {case['id']} | {case['category']} | {case['candidate_count']} | {case['resolution']} | "
            f"{ev['outcome']} | {_rank(ev['expected_ranks']['lexical'])} | {_rank(ev['expected_ranks']['semantic'])} | "
            f"{ev['expected_is'].replace('_', ' ')} |"
        )
    lines.append("")

    lines += [
        "## The possible bounded-judgment job",
        "",
        "If a later experiment tests a judge, this is the one job it would do:",
        "",
        "```text",
        "Input:   one query",
        "         the small deterministic candidate set for an AMBIGUOUS case",
        "         bounded evidence/features per candidate",
        "Output:  one choice, a score per candidate, or no selection",
        "```",
        "",
        "Jev and Laya would be alternative implementations of that single role, compared on the same "
        "cases; they would not both run in sequence by default. Neither is a retriever, an embedding model, "
        "a permission engine, a lifecycle authority, an action gate, or an autonomous planner. AGREEMENT "
        "cases never reach the judge, and retrieval failures are outside its reach. T3c calls neither model.",
        "",
        "## Method",
        "",
        f"- Candidate set: {config['candidate_set']}. {config['candidate_ranks']}. Order: {config['candidate_order']}.",
        f"- AGREEMENT: {config['agreement']}. DOMINANCE: {config['dominance']}. AMBIGUOUS: {config['ambiguous']}.",
        f"- Labels: {config['labels_used_for']}. RRF: {config['rrf']}.",
        f"- Input: `{source['path']}` (sha256 `{(source['sha256'] or 'n/a')[:12]}`), T3a run {source['t3a_started_at']}, "
        f"embedder {source['t3a_embedder'].get('model_id', source['t3a_embedder'].get('embedder'))}, "
        f"{source['corpus_documents']} documents. Labels match `{result['manifest_check']['path']}`.",
        "",
    ]
    return "\n".join(lines)


def render_summary(result: dict[str, Any], *, json_path: str | None = None, markdown_path: str | None = None) -> str:
    source = result["input"]
    config = result["config"]
    recall = result["candidate_recall"]
    size = result["candidate_size"]
    resolution = result["resolution"]
    ambiguous = result["ambiguous"]
    refs = result["end_to_end_references"]
    n = refs["n"]
    by_kind = resolution["resolved_accuracy"]["by_kind"]
    embedder = source["t3a_embedder"]
    lines = [
        f"T3c candidate audit — union(lexical top-{config['top_n']}, semantic top-{config['top_n']}), {n} cases, "
        f"embedder {embedder.get('model_id', embedder.get('embedder'))}",
    ]
    if source["fake_embedder"]:
        lines.append("  FAKE EMBEDDER INPUT: pipeline check only")
    overall = recall["overall"]
    lines += [
        f"  candidate recall: lexical top-{config['top_n']} {_frac(overall['lexical_top'], n)}, "
        f"semantic top-{config['top_n']} {_frac(overall['semantic_top'], n)}, union {_frac(overall['union'], n)}",
        "    union by category: "
        + ", ".join(f"{name} {_frac(data['union'], data['n'])}" for name, data in recall["by_category"].items()),
        f"  candidate-set size: mean {_num(size['mean'])}, median {_num(size['median'])}, max {size['max']} "
        f"({_histogram(size['histogram'])})",
        f"  agreement {len(resolution['counts']['agreement'])} (correct {by_kind['agreement']['correct']}), "
        f"dominance {len(resolution['counts']['dominance'])} (correct {by_kind['dominance']['correct']}), "
        f"ambiguous {ambiguous['n']} (expected is a candidate in {ambiguous['expected_in_union']})",
        f"  retrieval failures: {_ids(result['retrieval_failures'])}",
        "  ambiguous, expected is: "
        + ", ".join(f"{_ROLES[role]} {len(ids)}" for role, ids in ambiguous["expected_is"].items()),
        "  on ambiguous (diagnostics): "
        + ", ".join(
            f"{label} {_frac(ambiguous['selector_diagnostics'][key], ambiguous['selector_diagnostics'][key]['n'])}"
            for key, label in (("lexical_top1", "lexical #1"), ("semantic_top1", "semantic #1"), ("rrf_top1", "RRF #1"))
        ),
        f"  references /{n}: deterministic + perfect judge {refs['deterministic_plus_perfect_judge']}, "
        f"+ lexical #1 {refs['deterministic_plus_lexical_top1']}, + semantic #1 "
        f"{refs['deterministic_plus_semantic_top1']}, RRF {refs['rrf_top1']}",
        "  T3a buckets in union: "
        + ", ".join(
            f"{_SHORT[bucket]} {data['in_union']}/{data['n']}" for bucket, data in result["t3a_wins_in_union"].items()
        ),
    ]
    if json_path:
        lines.append(f"  json:     {json_path}")
    if markdown_path:
        lines.append(f"  markdown: {markdown_path}")
    return "\n".join(lines)


def _case_detail(case: dict[str, Any]) -> list[str]:
    ev = case["evaluation"]
    header = f"### {case['id']} ({case['category']}, {case['resolution']}"
    header += f" → `{case['resolved_path']}`)" if case["resolved_path"] else ")"
    lines = [
        header,
        "",
        f"- Query: {case['query']}",
        f"- Expected: `{case['expected_path']}` — lexical {_rank(ev['expected_ranks']['lexical'])}, "
        f"semantic {_rank(ev['expected_ranks']['semantic'])}; {ev['outcome'].replace('_', ' ')}",
        f"- Candidates ({case['candidate_count']}; Pareto front {case['pareto_front_size']}):",
    ]
    for candidate in case["candidates"]:
        marks = []
        if candidate["path"] == case["expected_path"]:
            marks.append("expected")
        if candidate["on_pareto_front"]:
            marks.append("front")
        lines.append(
            f"  - `{candidate['path']}` — lexical {_rank(candidate['lexical_rank'])}, "
            f"semantic {_rank(candidate['semantic_rank'])}" + (f" ({', '.join(marks)})" if marks else "")
        )
    lines.append("")
    return lines


def _frac(data: dict[str, Any], n: int) -> str:
    count = data["count"] if "count" in data else data["correct"]
    return f"{count}/{n} ({_pct(count / n if n else None)})"


def _histogram(histogram: dict[str, int]) -> str:
    return ", ".join(f"{size}: {count}" for size, count in sorted(histogram.items(), key=lambda kv: int(kv[0]))) or "none"


def _ids(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "none"


def _rank(rank: int | None) -> str:
    return "—" if rank is None else str(rank)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
