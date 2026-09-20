from __future__ import annotations

import json
from pathlib import Path

from zomah.access import ReadScope
from zomah.capabilities import ListDirectoryRequest, list_directory


repo_root = Path(__file__).resolve().parents[1]
scope = ReadScope.from_paths([repo_root])
response = list_directory(
    ListDirectoryRequest(path=str(repo_root), max_entries=50),
    scope,
)

print(json.dumps(response.model_dump(mode="json"), indent=2))
