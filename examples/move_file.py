from __future__ import annotations

import json
import tempfile
from pathlib import Path

from zomah.access import WriteScope
from zomah.capabilities import MoveFileRequest, move_file


with tempfile.TemporaryDirectory(prefix="zomah-move-demo-") as temp_dir:
    root = Path(temp_dir)
    inbox = root / "inbox"
    archive = root / "archive"
    inbox.mkdir()
    archive.mkdir()

    source = inbox / "note.md"
    destination = archive / "note.md"
    source.write_text("move me safely\n", encoding="utf-8")

    response = move_file(
        MoveFileRequest(source=str(source), destination=str(destination)),
        WriteScope.from_paths([root]),
    )

    print(
        json.dumps(
            {
                "write_root": str(root),
                "response": response.model_dump(mode="json"),
                "source_exists": source.exists(),
                "destination_content": destination.read_text(encoding="utf-8"),
            },
            indent=2,
        )
    )
