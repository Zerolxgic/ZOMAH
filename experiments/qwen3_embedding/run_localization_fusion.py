"""T3f: deterministic evidence projection + bounded localization fusion over a frozen T3e run.

Needs no model and no corpus: it reads one T3e results.json. Run it with the
ZOMAH environment (query phrases use ZOMAH's own lexical matcher):

    .venv/bin/python experiments/qwen3_embedding/run_localization_fusion.py \\
      --t3e-results ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3e/results.json \\
      --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3f

Writes results.json and report.md and prints a summary. Stops without writing
anything if the artifact is malformed or its recomputed baselines differ from
the accepted T3e live run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from embedding_bench.localization_fusion import T3E_ACCEPTED, T3eInputError, T3eReferenceDrift, run_t3f  # noqa: E402
from embedding_bench.localization_fusion_report import render_markdown, render_summary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--t3e-results", type=Path, required=True, help="the T3e run's results.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-t3e-reference",
        action="store_true",
        help="do not require the accepted T3e live baselines (fixtures only; flagged in every output)",
    )
    args = parser.parse_args(argv)

    t3e_results = args.t3e_results.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == t3e_results.parent:
        print("--output-dir must not be the T3e run's directory", file=sys.stderr)
        return 2
    try:
        result = run_t3f(t3e_results, reference=None if args.skip_t3e_reference else T3E_ACCEPTED)
    except T3eReferenceDrift as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 3
    except T3eInputError as exc:
        print(f"cannot run T3f: {exc}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": output_dir / "results.json", "markdown": output_dir / "report.md"}
    paths["json"].write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["markdown"].write_text(render_markdown(result), encoding="utf-8")
    print(render_summary(result, paths={k: str(v) for k, v in paths.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
