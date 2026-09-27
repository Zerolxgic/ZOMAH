"""T3d step 1: freeze the bounded-judgment cases every judge will receive.

Runs in ZOMAH's environment (it uses the unchanged KnowledgeIndex corpus and
lexical localization). Judges only ever read the cases file this writes.

    python experiments/qwen3_embedding/prepare_cases.py \\
      --input ~/.local/share/zomah/experiments/qwen3-embedding/run-1/results.json \\
      --t3c-results ~/.local/share/zomah/experiments/qwen3-embedding/run-1-candidate-audit/results.json \\
      --output ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3d/cases.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from embedding_bench.fusion import FusionInputError, check_labels, load_t3a  # noqa: E402
from embedding_bench.judge_prepare import PrepareError, prepare_document  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="the T3a results.json")
    parser.add_argument("--output", type=Path, required=True, help="the cases.json file to write")
    parser.add_argument("--t3c-results", type=Path, help="optional T3c results.json to cross-check against")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="corpus root T3a ranked (default: this checkout)")
    parser.add_argument("--manifest", type=Path, default=HERE / "benchmark.json")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing cases file")
    args = parser.parse_args(argv)

    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.parent == input_path.parent:
        print("--output must not be inside the T3a run's directory", file=sys.stderr)
        return 2
    if output.exists() and not args.overwrite:
        print(f"{output} exists; frozen cases are not replaced without --overwrite", file=sys.stderr)
        return 2
    try:
        artifact = load_t3a(input_path)
        label_check = check_labels(artifact, args.manifest)
        t3c = None
        if args.t3c_results:
            t3c = json.loads(args.t3c_results.expanduser().read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="zomah-t3d-") as workdir:
            document = prepare_document(
                artifact,
                root=args.root,
                manifest_path=args.manifest,
                label_check=label_check,
                workdir=Path(workdir),
                t3c_results=t3c,
                t3c_path=str(args.t3c_results) if args.t3c_results else None,
            )
    except (FusionInputError, PrepareError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"cannot prepare T3d cases: {exc}", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = document["summary"]
    baselines = summary["ambiguous_baselines"]
    print(
        f"T3d cases: {summary['ambiguous']} ambiguous (primary), {summary['resolved']} resolved (diagnostic)\n"
        f"  ambiguous baselines: lexical #1 {baselines['lexical_top1']}, semantic #1 {baselines['semantic_top1']}, "
        f"RRF #1 {baselines['rrf_top1']}, T3c order first {baselines['t3c_order_first']}, "
        f"first option {baselines['first_option']} (of {summary['ambiguous']})"
        + (" — verified against T3c" if summary["baselines_verified_against_t3c"] else "")
        + f"\n  expected excerpt shows the evidence in {summary['ambiguous_expected_excerpt_shows_anchor']}/"
        f"{summary['ambiguous']} ambiguous cases\n  cases: {output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
