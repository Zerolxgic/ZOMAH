from __future__ import annotations

import json
import tempfile
from pathlib import Path

from zomah.access import WriteScope
from zomah.capabilities import WriteFileRequest, write_file


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="zomah-write-demo-") as temp_dir:
        root = Path(temp_dir)
        target = root / "demo.txt"
        scope = WriteScope.from_paths([root])

        responses = [
            write_file(
                WriteFileRequest(
                    path=str(target),
                    content="created by ZOMAH\n",
                    mode="create",
                ),
                scope,
            ),
            write_file(
                WriteFileRequest(
                    path=str(target),
                    content="appended safely\n",
                    mode="append",
                ),
                scope,
            ),
            write_file(
                WriteFileRequest(
                    path=str(target),
                    content="final replacement\n",
                    mode="replace",
                ),
                scope,
            ),
        ]

        print(
            json.dumps(
                {
                    "write_root": str(root),
                    "responses": [item.model_dump(mode="json") for item in responses],
                    "final_content": target.read_text(encoding="utf-8"),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
