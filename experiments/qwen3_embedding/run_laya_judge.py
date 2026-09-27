"""T3d step 2b: run a stock Laya checkpoint on the frozen cases, alone.

Needs the `laya` package (and its torch/transformers stack) in the environment.

    python experiments/qwen3_embedding/run_laya_judge.py \\
      --cases ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/cases.json \\
      --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/laya
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from embedding_bench.judge_adapters import LAYA_DEFAULT_MODEL, create_laya_judge, environment  # noqa: E402
from embedding_bench.judge_runner import execute  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=LAYA_DEFAULT_MODEL)
    parser.add_argument("--subfolder", help="checkpoint subfolder; default: the repo root (English)")
    parser.add_argument("--device", help="cpu, cuda, mps...; default: Laya's own choice")
    parser.add_argument("--max-len", type=int, help="token budget; default: the checkpoint's")
    parser.add_argument("--head-max-len", type=int, help="option/instruction budget; default: the checkpoint's")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return execute(
        judge_name="laya",
        cases_path=args.cases,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        warmup=not args.no_warmup,
        make_judge=lambda: create_laya_judge(
            model=args.model,
            subfolder=args.subfolder,
            device=args.device,
            max_len=args.max_len,
            head_max_len=args.head_max_len,
        ),
        environment=lambda: environment(("laya", "torch", "transformers", "huggingface-hub", "numpy")),
    )


if __name__ == "__main__":
    sys.exit(main())
