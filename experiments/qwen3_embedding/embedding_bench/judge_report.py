"""Render a T3d comparison as Markdown and as a terminal summary."""

from __future__ import annotations

from typing import Any

from embedding_bench.judge_compare import BASELINE_LABELS, BASELINES

ROLE_LABELS = {
    "lexical_top1": "answer is lexical #1",
    "semantic_top1": "answer is semantic #1",
    "other_candidate": "answer is neither #1",
}

DECOMPOSITION = """```text
deterministic (ZOMAH):
  filesystem scope, candidate construction (T3c union), candidate dedupe,
  provenance, excerpt generation (lexical localization), budgets,
  routing conditions (agreement vs ambiguous)

probabilistic:
  lexical / semantic retrieval scores
  bounded Jev or Laya relevance judgment over the supplied candidates
  final LLM interpretation of the evidence
```

The judge chooses among candidates it is given, or declines. It does not
retrieve, search, read other files, set permissions, change lifecycle state,
gate actions, or plan. Jev and Laya are alternative implementations of this
one role, run independently here; neither sees the other's output."""


def render_markdown(result: dict[str, Any]) -> str:
    names = list(result["primary"])
    n = result["counts"]["ambiguous"]
    lines = [
        "# T3d: Jev vs Laya on bounded retrieval selection",
        "",
        f"{result['experiment']}. Generated {result['created_at']}. Cases file sha256 "
        f"`{result['cases_sha256'][:12]}`; every judge received the identical stored request per case.",
        "",
        f"Primary set: the {n} T3c AMBIGUOUS cases. Diagnostic set: the {result['counts']['resolved']} "
        "deterministically resolved (agreement) cases.",
        "",
        "## Primary results (ambiguous cases)",
        "",
        "| system | correct | accuracy | abstained | protocol failures | accuracy when selecting | median latency | mean latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        m = result["primary"][name]
        lines.append(
            f"| **{name}** | {m['correct']}/{m['n']} | {_pct(m['accuracy'])} | {len(m['abstained'])} | "
            f"{len(m['protocol_failures'])} | {_pct(m['accuracy_when_selecting'])} | {_ms(m['latency_ms']['median'])} | "
            f"{_ms(m['latency_ms']['mean'])} |"
        )
    for name in BASELINES:
        b = result["baselines"][name]
        lines.append(f"| {BASELINE_LABELS[name]} | {b['correct']}/{b['n']} | {_pct(b['accuracy'])} | — | — | — | — | — |")
    lines += [
        "",
        "Baselines are recomputed from the frozen cases. \"First listed option\" measures position bias: "
        "options are ordered by path, not by retrieval rank.",
        "",
        "## Where the answer sits",
        "",
        "| system | " + " | ".join(ROLE_LABELS.values()) + " |",
        "|---|" + "---:|" * len(ROLE_LABELS),
    ]
    for name in names:
        roles = result["primary"][name]["by_expected_role"]
        lines.append(
            f"| {name} | " + " | ".join(f"{len(roles[r]['correct'])}/{roles[r]['n']}" for r in ROLE_LABELS) + " |"
        )
    for name in ("lexical_top1", "semantic_top1", "rrf_top1"):
        lines.append(
            f"| {BASELINE_LABELS[name]} | "
            + " | ".join(
                f"{sum(1 for c in result['cases'] if c['set'] == 'ambiguous' and c['expected_is'] == r and c['baselines'][name]['correct'])}"
                f"/{sum(1 for c in result['cases'] if c['set'] == 'ambiguous' and c['expected_is'] == r)}"
                for r in ROLE_LABELS
            )
            + " |"
        )
    lines.append("")

    lines += ["## Against RRF", ""]
    for name in names:
        vs = result["primary"][name]["vs_rrf"]
        lines.append(f"- **{name}** improves on RRF: {_ids(vs['improves'])}; worse than RRF: {_ids(vs['worsens'])}.")
    lines.append("")

    categories = list(result["baselines"]["rrf_top1"]["by_category"])
    lines += ["## By category (ambiguous cases)", "", "| system | " + " | ".join(categories) + " |", "|---|" + "---:|" * len(categories)]
    for name in names:
        by = result["primary"][name]["by_category"]
        lines.append(f"| {name} | " + " | ".join(f"{by[c]['correct']}/{by[c]['n']}" for c in categories) + " |")
    for name in ("lexical_top1", "semantic_top1", "rrf_top1"):
        by = result["baselines"][name]["by_category"]
        lines.append(f"| {BASELINE_LABELS[name]} | " + " | ".join(f"{by[c]['correct']}/{by[c]['n']}" for c in categories) + " |")
    lines.append("")

    if "pairwise" in result:
        pw = result["pairwise"]
        a, b = pw["judges"]
        lines += ["## Pairwise (ambiguous cases)", ""]
        for key, label in (
            ("both_correct", "both correct"),
            (f"{a}_only_correct", f"{a} only correct"),
            (f"{b}_only_correct", f"{b} only correct"),
            ("both_wrong", "both wrong"),
            (f"{a}_abstains_{b}_selects", f"{a} abstains, {b} selects"),
            (f"{b}_abstains_{a}_selects", f"{b} abstains, {a} selects"),
            ("both_abstain", "both abstain"),
            ("same_selection", "same selection"),
        ):
            lines.append(f"- {label}: {len(pw[key])} — {_ids(pw[key])}")
        lines += ["", "Complementarity is reported, not exploited: no ensemble or sequential judging is run.", ""]

    lines += ["## Agreement diagnostics (resolved cases)", ""]
    for name in names:
        d = result["agreement_diagnostics"][name]
        wrong = "; ".join(
            f"{w['id']}: {'fixed' if w['judge_correct'] else 'not fixed'} ({w['judge_status']})" for w in d["wrong_agreements"]
        )
        damaged = "; ".join(f"{x['id']} ({x['status']}, {x['selected_path'] or 'no selection'})" for x in d["damaged"])
        lines.append(
            f"- **{name}** agrees with the retrievers in {d['agrees_with_retrievers']}/{d['n']}; "
            f"wrong agreement(s): {wrong or 'none'}; correct agreements damaged: "
            f"{len(d['damaged'])}/{d['correct_agreements']}{' — ' + damaged if damaged else ''}."
        )
    lines += ["", "This measures whether \"skip judging on agreement\" is a safe routing rule; it is not a proposal to judge agreement cases.", ""]

    ev = result["evidence_pack"]
    lines += [
        "## Evidence-pack limits",
        "",
        f"- Ambiguous cases whose expected candidate's excerpt does not cover the benchmark evidence lines: "
        f"{len(ev['expected_excerpt_misses_anchor'])} — {_ids(ev['expected_excerpt_misses_anchor'])}.",
    ]
    for name in names:
        lines.append(
            f"- {name}: accuracy when the evidence was in the excerpt {_pct(ev['accuracy_when_anchor_shown'][name])}, "
            f"when it was not {_pct(ev['accuracy_when_anchor_missed'][name])}."
        )
    lines += [
        f"- Every judge wrong and the evidence not in the expected excerpt (likely evidence-pack limitation): "
        f"{_ids(ev['all_judges_wrong_anchor_missed'])}.",
        f"- Every judge wrong although the evidence was shown (judgment failure): {_ids(ev['all_judges_wrong_anchor_shown'])}.",
    ]
    for name in names:
        budget = ev.get(f"{name}_token_budget")
        if budget:
            lines.append(
                f"- {name} token budget: options truncated in {budget['cases_with_truncated_options']}/{budget['cases']} "
                f"cases, state truncated in {budget['cases_with_truncated_state']}; expected candidate's option "
                f"truncated in {_ids(budget['expected_option_truncated'])}."
            )
    lines.append("")

    e2e = result["end_to_end_reference"]
    lines += [
        "## End-to-end references (not systems)",
        "",
        f"Out of {e2e['n']}: agreement routing is correct in {e2e['resolved_correct']} resolved cases; RRF top-1 "
        f"{e2e['rrf_top1']}. Agreement routing plus one judge on the ambiguous cases: "
        + ", ".join(f"{n} {v}" for n, v in e2e["routing_plus_judge"].items())
        + ". Judge on every case: "
        + ", ".join(f"{n} {v}" for n, v in e2e["judge_everywhere"].items())
        + ".",
        "",
        "## Per-case results",
        "",
        "| id | set | category | answer is | " + " | ".join(names) + " | RRF |",
        "|---|---|---|---|" + "---|" * len(names) + "---|",
    ]
    for case in result["cases"]:
        cells = [_judge_cell(case, name) for name in names]
        rrf = "✓" if case["baselines"]["rrf_top1"]["correct"] else "✗"
        lines.append(
            f"| {case['id']} | {case['set']} | {case['category']} | {case['expected_is'].replace('_', ' ')} | "
            + " | ".join(cells)
            + f" | {rrf} |"
        )
    lines.append("")

    misses = [c for c in result["cases"] if c["set"] == "ambiguous" and names and not all(c["judges"][n]["correct"] for n in names)]
    if misses:
        lines += ["## Ambiguous cases at least one judge missed", ""]
        for case in misses:
            lines += _case_detail(case, names)

    lines += ["## Responsibility split", "", DECOMPOSITION, "", "## Reproducibility", ""]
    render = result.get("render") or {}
    evidence = result.get("evidence") or {}
    lines += [
        f"- Question `{render.get('question_id')}` ({render.get('question_type')}): “{render.get('instructions')}”",
        f"- State: `{render.get('state_template')}`; option: `{render.get('option_template')}`; "
        f"abstention label `{render.get('abstention_label')}`.",
        f"- Evidence: {evidence.get('excerpt_algorithm')}, ≤{evidence.get('excerpt_max_chars')} characters; "
        f"candidates {evidence.get('candidate_order')}.",
        f"- Never sent to a judge: {', '.join(evidence.get('never_sent', []))}.",
    ]
    for name, info in result["judges"].items():
        lines.append(f"- **{name}** config: {_kv(info['config'] or {})}")
        lines.append(f"  - environment: {_kv(info['environment'] or {})}; load {_s(info.get('load_s'))}, warm-up {_ms(info.get('warmup_ms'))}")
    source = result.get("source") or {}
    lines += [
        f"- T3a artifact sha256 `{(source.get('t3a_sha256') or 'n/a')[:12]}`, corpus at ZOMAH commit "
        f"`{(source.get('zomah_commit') or 'n/a')[:10]}`.",
        "",
    ]
    return "\n".join(lines)


def render_summary(result: dict[str, Any], *, json_path: str | None = None, markdown_path: str | None = None) -> str:
    names = list(result["primary"])
    lines = [f"T3d bounded judgment — {result['counts']['ambiguous']} ambiguous cases (primary), {result['counts']['resolved']} resolved (diagnostic)"]
    for name in names:
        m = result["primary"][name]
        vs = m["vs_rrf"]
        lines.append(
            f"  {name:8} {m['correct']}/{m['n']} correct, {len(m['abstained'])} abstained, "
            f"{len(m['protocol_failures'])} protocol failures, median {_ms(m['latency_ms']['median'])}; "
            f"vs RRF +{len(vs['improves'])} / -{len(vs['worsens'])}"
        )
    lines.append(
        "  baselines: "
        + ", ".join(f"{BASELINE_LABELS[b]} {result['baselines'][b]['correct']}/{result['baselines'][b]['n']}" for b in BASELINES)
    )
    if "pairwise" in result:
        pw = result["pairwise"]
        a, b = pw["judges"]
        lines.append(
            f"  pairwise: both {len(pw['both_correct'])}, {a} only {len(pw[f'{a}_only_correct'])}, "
            f"{b} only {len(pw[f'{b}_only_correct'])}, neither {len(pw['both_wrong'])}"
        )
    for name in names:
        d = result["agreement_diagnostics"][name]
        lines.append(
            f"  agreement ({name}): agrees {d['agrees_with_retrievers']}/{d['n']}, fixes wrong "
            f"{sum(1 for w in d['wrong_agreements'] if w['judge_correct'])}/{len(d['wrong_agreements'])}, "
            f"damages {len(d['damaged'])}/{d['correct_agreements']}"
        )
    if json_path:
        lines.append(f"  json:     {json_path}")
    if markdown_path:
        lines.append(f"  markdown: {markdown_path}")
    return "\n".join(lines)


def _judge_cell(case: dict[str, Any], name: str) -> str:
    judged = case["judges"][name]
    if judged["status"] == "abstained":
        return "abstain"
    if judged["status"] == "protocol_failure":
        return "protocol failure"
    return f"{judged['selected_candidate']} {'✓' if judged['correct'] else '✗'}"


def _case_detail(case: dict[str, Any], names: list[str]) -> list[str]:
    lines = [
        f"### {case['id']} ({case['category']}; {case['expected_is'].replace('_', ' ')})",
        "",
        f"- Query: {case['query']}",
        f"- Expected: `{case['expected_path']}` ({case['expected_candidate_id']}); evidence in its excerpt: "
        f"{case['expected_excerpt_shows_anchor']}",
    ]
    for name in names:
        judged = case["judges"][name]
        parts = [judged["status"]]
        if judged["selected_candidate"]:
            parts.append(judged["selected_candidate"])
        text = " ".join(parts)
        if isinstance(judged["selected_probability"], float):
            text += f", p={judged['selected_probability']:.3f}"
        if judged["protocol_error"]:
            text += f" ({judged['protocol_error']})"
        lines.append(f"- {name}: {text}")
    lines.append("- Candidates:")
    for c in case["candidates"]:
        mark = " (expected)" if c["id"] == case["expected_candidate_id"] else ""
        lines.append(
            f"  - {c['id']} `{c['path']}`{mark} — lexical {c['lexical_rank'] or '—'}, semantic {c['semantic_rank'] or '—'}, "
            f"lines {c['start_line']}-{c['end_line']}: {' '.join(c['excerpt'].split())}"
        )
    lines.append("")
    return lines


def _ids(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "none"


def _kv(mapping: dict[str, Any]) -> str:
    return "; ".join(f"{k}={v}" for k, v in mapping.items())


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def _s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} s"
