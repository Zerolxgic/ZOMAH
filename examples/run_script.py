from __future__ import annotations

import json
import tempfile
from pathlib import Path

from zomah.capabilities.run_script import RunScriptRequest, run_script
from zomah.execution import RegisteredScript, ScriptArgumentSpec, ScriptRegistry


with tempfile.TemporaryDirectory(prefix="zomah-script-demo-") as temp_dir:
    root = Path(temp_dir)
    script = root / "system-summary.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mode=short\n"
        "verbose=0\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  case \"$1\" in\n"
        "    --mode) mode=\"$2\"; shift 2 ;;\n"
        "    --verbose) verbose=1; shift ;;\n"
        "  esac\n"
        "done\n"
        "printf 'mode=%s\\n' \"$mode\"\n"
        "printf 'cwd=%s\\n' \"$PWD\"\n"
        "printf 'verbose=%s\\n' \"$verbose\"\n"
    )
    script.chmod(0o700)

    registry = ScriptRegistry(
        (
            RegisteredScript(
                name="system-summary",
                path=script,
                working_directory=root,
                arguments=(
                    ScriptArgumentSpec(
                        name="mode",
                        flag="--mode",
                        kind="string",
                        choices=("short", "full"),
                    ),
                    ScriptArgumentSpec(
                        name="verbose",
                        flag="--verbose",
                        kind="boolean",
                    ),
                ),
                timeout_seconds=5,
            ),
        )
    )

    response = run_script(
        RunScriptRequest(
            script="system-summary",
            arguments={"mode": "full", "verbose": True},
        ),
        registry,
    )

    print(json.dumps(response.model_dump(mode="json"), indent=2))
