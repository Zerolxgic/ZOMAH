"""T3a: rank ZOMAH documents with Qwen3-Embedding-0.6B and compare to the lexical baseline.

Run inside the isolated experiment environment (see README.md):

    python experiments/qwen3_embedding/run_benchmark.py --download-only   # once
    python experiments/qwen3_embedding/run_benchmark.py                   # CPU benchmark

Writes results.json and report.md to --output-dir (default: outside the
repository, under $XDG_DATA_HOME/zomah/experiments/qwen3-embedding/<UTC time>/)
and prints a short summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_MANIFEST = HERE / "benchmark.json"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

from embedding_bench.manifest import ManifestError  # noqa: E402
from embedding_bench.report import render_markdown, render_summary  # noqa: E402
from embedding_bench.retrieval import HashingEmbedder  # noqa: E402
from embedding_bench.runner import run_benchmark  # noqa: E402


def default_output_dir() -> Path:
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return data_home / "zomah" / "experiments" / "qwen3-embedding" / stamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="corpus root (default: this ZOMAH checkout)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, help="where results.json and report.md are written")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hub id or local sentence-transformers directory")
    parser.add_argument("--revision", help="optional Hub revision to pin")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download-only", action="store_true", help="fetch the model into the HF cache and exit")
    mode.add_argument(
        "--fake-embedder",
        action="store_true",
        help="pipeline check with a hashing bag-of-words stand-in (no torch; NOT semantic)",
    )
    args = parser.parse_args(argv)

    import_s = None
    if args.fake_embedder:
        factory = HashingEmbedder
    else:
        if not args.download_only:
            # Read by huggingface_hub at import: the timed load can never touch the network.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        start = time.perf_counter()
        from embedding_bench import qwen

        import_s = time.perf_counter() - start
        if args.download_only:
            snapshot = qwen.download(args.model, revision=args.revision)
            print(f"cached {args.model} (snapshot {snapshot or 'n/a'})")
            return 0
        if not qwen.is_available_offline(args.model, args.revision):
            print(
                f"{args.model} is not in the local Hugging Face cache; run with --download-only first",
                file=sys.stderr,
            )
            return 2

        def factory() -> qwen.QwenEmbedder:
            return qwen.QwenEmbedder(args.model, revision=args.revision)

    output_dir = (args.output_dir or default_output_dir()).expanduser()
    try:
        with tempfile.TemporaryDirectory(prefix="zomah-t3a-") as workdir:
            result = run_benchmark(
                root=args.root,
                manifest_path=args.manifest.resolve(),
                embedder_factory=factory,
                workdir=Path(workdir),
            )
    except ManifestError as exc:
        print(f"invalid benchmark manifest: {exc}", file=sys.stderr)
        return 2
    result["timings"]["library_import_s"] = import_s

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(render_summary(result, json_path=str(json_path), markdown_path=str(markdown_path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
