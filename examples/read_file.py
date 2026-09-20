from __future__ import annotations

from pathlib import Path

from zomah.access import ReadScope
from zomah.capabilities import ReadFileRequest, read_file


repo_root = Path(__file__).resolve().parents[1]
scope = ReadScope.from_paths([repo_root])

response = read_file(
    ReadFileRequest(
        path=str(repo_root / "README.md"),
        start_line=1,
        max_lines=40,
    ),
    scope,
)

print(response.model_dump_json(indent=2))
