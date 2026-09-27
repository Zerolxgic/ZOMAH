"""T3e: semantic within-document evidence localization vs lexical localize().

Run in the T3a experiment environment (Python 3.13, cached Qwen3-Embedding-0.6B):

    experiments/qwen3_embedding/.venv/bin/python experiments/qwen3_embedding/run_localization.py \\
      --t3a-results ~/.local/share/zomah/experiments/qwen3-embedding/run-1/results.json \\
      --output-dir ~/.local/share/zomah/experiments/qwen3-embedding/run-1-t3e

Writes results.json and report.md (and candidate_packs.json with --t3a-results)
and prints a summary. Stops before loading the model if the recomputed lexical
baseline differs from the recorded one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from embedding_bench.fusion import FusionInputError  # noqa: E402
from embedding_bench.judge_prepare import PrepareError  # noqa: E402
from embedding_bench.localization_report import render_markdown, render_summary  # noqa: E402
from embedding_bench.localization_runner import LEXICAL_OVERLAP_REFERENCE, LexicalReferenceDrift, run_t3e  # noqa: E402
from embedding_bench.manifest import ManifestError  # noqa: E402
from embedding_bench.retrieval import HashingEmbedder  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--t3a-results", type=Path, help="T3a results.json: corpus drift check + candidate-pack simulation")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="corpus root (default: this checkout)")
    parser.add_argument("--manifest", type=Path, default=HERE / "benchmark.json")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--revision")
    parser.add_argument("--fake-embedder", action="store_true", help="pipeline check with the hashing stand-in (NOT semantic)")
    parser.add_argument("--skip-lexical-reference", action="store_true", help="do not require the recorded lexical baseline")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    if args.t3a_results and output_dir == args.t3a_results.expanduser().resolve().parent:
        print("--output-dir must not be the T3a run's directory", file=sys.stderr)
        return 2

    import_s = None
    if args.fake_embedder:
        factory = HashingEmbedder
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the timed load never touches the network
        start = time.perf_counter()
        from embedding_bench import qwen

        import_s = time.perf_counter() - start
        if not qwen.is_available_offline(args.model, args.revision):
            print(f"{args.model} is not in the local Hugging Face cache; run run_benchmark.py --download-only first", file=sys.stderr)
            return 2

        def factory():
            return qwen.QwenEmbedder(args.model, revision=args.revision)

    try:
        with tempfile.TemporaryDirectory(prefix="zomah-t3e-") as workdir:
            result, packs = run_t3e(
                root=args.root,
                manifest_path=args.manifest.resolve(),
                embedder_factory=factory,
                workdir=Path(workdir),
                t3a_path=args.t3a_results.expanduser().resolve() if args.t3a_results else None,
                lexical_reference=None if args.skip_lexical_reference else LEXICAL_OVERLAP_REFERENCE,
                progress=lambda line: print(line, file=sys.stderr, flush=True),
            )
    except LexicalReferenceDrift as exc:
        print(f"stopped before loading the model: {exc}", file=sys.stderr)
        return 3
    except (ManifestError, FusionInputError, PrepareError) as exc:
        print(f"cannot run T3e: {exc}", file=sys.stderr)
        return 2
    result["timings"]["library_import_s"] = import_s

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": output_dir / "results.json", "markdown": output_dir / "report.md"}
    paths["json"].write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["markdown"].write_text(render_markdown(result), encoding="utf-8")
    if packs is not None:
        paths["packs"] = output_dir / "candidate_packs.json"
        paths["packs"].write_text(json.dumps(packs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(render_summary(result, paths={k: str(v) for k, v in paths.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
