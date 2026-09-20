from __future__ import annotations

import argparse
import json
from pathlib import Path

from zomah.access import ReadScope
from zomah.capabilities import SearchKnowledgeRequest, search_knowledge
from zomah.knowledge import KnowledgeIndex, default_knowledge_db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Index and search one approved knowledge root.")
    parser.add_argument("query", help="lexical query to search")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="approved root to index (defaults to the ZOMAH repo)",
    )
    parser.add_argument("--max-results", type=int, default=10)
    args = parser.parse_args()

    root = args.root.expanduser().resolve(strict=True)
    scope = ReadScope.from_paths([root])
    index = KnowledgeIndex(default_knowledge_db_path(), scope)
    report = index.refresh()
    response = search_knowledge(
        SearchKnowledgeRequest(query=args.query, max_results=args.max_results),
        index,
    )

    print(
        json.dumps(
            {
                "refresh": {
                    "scanned_files": report.scanned_files,
                    "indexed_files": report.indexed_files,
                    "unchanged_files": report.unchanged_files,
                    "removed_files": report.removed_files,
                    "skipped_files": report.skipped_files,
                },
                "search": response.model_dump(mode="json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
