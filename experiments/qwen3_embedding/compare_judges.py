"""T3d step 3: compare independent judge runs on the frozen cases (standard library only).

    python experiments/qwen3_embedding/compare_judges.py \\
      --cases ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/cases.json \\
      --jev   ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/jev/judgments.json \\
      --laya  ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/laya/judgments.json \\
      --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/comparison

Either judge may be omitted to score one on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from embedding_bench.judge_compare import compare, load_judgments  # noqa: E402
from embedding_bench.judge_contract import JudgeContractError, load_cases  # noqa: E402
from embedding_bench.judge_report import render_markdown, render_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--jev", type=Path, help="Jev judgments.json")
    parser.add_argument("--laya", type=Path, help="Laya judgments.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.jev and not args.laya:
        parser.error("give --jev, --laya, or both")

    try:
        cases_doc, cases_sha256 = load_cases(str(args.cases.expanduser()))
        judgments = {}
        for name, path in (("jev", args.jev), ("laya", args.laya)):
            if path:
                loaded = load_judgments(str(path.expanduser()), cases_doc, cases_sha256)
                if loaded.get("judge") != name:
                    raise JudgeContractError(f"{path} holds {loaded.get('judge')!r} judgments, not {name!r}")
                judgments[name] = loaded
    except (OSError, JudgeContractError) as exc:
        print(f"cannot compare: {exc}", file=sys.stderr)
        return 2

    result = compare(cases_doc, cases_sha256, judgments)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(render_summary(result, json_path=str(json_path), markdown_path=str(markdown_path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
