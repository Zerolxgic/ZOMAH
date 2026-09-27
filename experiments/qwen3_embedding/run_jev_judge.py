"""T3d step 2a: run Jev (TypeSafe System One) on the frozen cases, alone.

Needs `typesafe-sdk` (installed with `jev`) and TYPESAFE_API_KEY in the environment.

    python experiments/qwen3_embedding/run_jev_judge.py \\
      --cases ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/cases.json \\
      --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/jev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from embedding_bench.judge_adapters import create_jev_judge, environment  # noqa: E402
from embedding_bench.judge_runner import execute  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", help="Jev model; default: the SDK default (jev-latest)")
    parser.add_argument("--timeout", type=float, help="per-request timeout in seconds; default: the SDK default")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return execute(
        judge_name="jev",
        cases_path=args.cases,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        warmup=not args.no_warmup,
        make_judge=lambda: create_jev_judge(model=args.model, timeout=args.timeout),
        environment=lambda: environment(("typesafe-sdk", "jev", "httpx2", "pydantic")),
    )


if __name__ == "__main__":
    sys.exit(main())
