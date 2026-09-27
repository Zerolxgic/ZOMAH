"""Shared command-line flow for the T3d judge runners (standard library only)."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from embedding_bench.judge_contract import JudgeContractError, judgments_document, load_cases, run_judge, run_summary


def execute(
    *,
    judge_name: str,
    cases_path: Path,
    output_dir: Path,
    overwrite: bool,
    warmup: bool,
    make_judge: Callable[[], tuple[Any, dict[str, Any]]],
    environment: Callable[[], dict[str, Any]],
) -> int:
    """Load frozen cases, build one judge, run every case independently, write judgments.json."""

    output = output_dir.expanduser().resolve() / "judgments.json"
    if output.exists() and not overwrite:
        print(f"{output} exists; judgments are not replaced without --overwrite", file=sys.stderr)
        return 2
    try:
        cases_doc, cases_sha256 = load_cases(str(cases_path.expanduser()))
    except (OSError, JudgeContractError) as exc:
        print(f"cannot load T3d cases: {exc}", file=sys.stderr)
        return 2

    start = time.perf_counter()
    judge, config = make_judge()
    load_s = time.perf_counter() - start
    run = run_judge(cases_doc, judge, warmup=warmup)
    document = judgments_document(
        judge=judge_name,
        cases_path=str(cases_path),
        cases_sha256=cases_sha256,
        config=config,
        environment=environment(),
        run=run,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        load_s=load_s,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"{judge_name}: {run_summary(run)}; load {load_s:.2f} s\n  judgments: {output}")
    return 0
