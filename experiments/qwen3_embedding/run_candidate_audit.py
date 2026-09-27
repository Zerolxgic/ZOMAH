"""T3c: candidate-union recall and deterministic ambiguity audit over frozen T3a rankings.

Plain Python only (no torch, no model, no ZOMAH install):

    python experiments/qwen3_embedding/run_candidate_audit.py \\
      --input ~/.local/share/zomah/experiments/qwen3-embedding/run-1/results.json \\
      --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1-candidate-audit

Writes results.json and report.md to --output-dir and prints a short summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "benchmark.json"
sys.path.insert(0, str(HERE))

from embedding_bench.candidates import run_audit  # noqa: E402
from embedding_bench.candidates_report import render_markdown, render_summary  # noqa: E402
from embedding_bench.fusion import FusionInputError, check_labels, load_t3a  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="a T3a results.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="where results.json and report.md are written")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="benchmark manifest whose labels the T3a cases must carry unchanged",
    )
    args = parser.parse_args(argv)

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == input_path.parent:
        print(
            "--output-dir must differ from the T3a run's directory; writing there would overwrite its results",
            file=sys.stderr,
        )
        return 2

    try:
        artifact = load_t3a(input_path)
        label_check = check_labels(artifact, args.manifest)
    except FusionInputError as exc:
        print(f"cannot audit T3a results: {exc}", file=sys.stderr)
        return 2
    result = run_audit(artifact, label_check=label_check)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(render_summary(result, json_path=str(json_path), markdown_path=str(markdown_path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
