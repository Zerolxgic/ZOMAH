from __future__ import annotations

import argparse
import json

from zomah.capabilities import InspectSystemRequest, inspect_system


parser = argparse.ArgumentParser(description="Inspect one supported ZOMAH system domain.")
parser.add_argument(
    "domain",
    nargs="?",
    default="memory",
    choices=["processes", "services", "storage", "memory", "gpu", "mounts"],
)
parser.add_argument("--query", help="Optional filter for processes, services, or mounts.")
parser.add_argument("--max-entries", type=int, default=100)
args = parser.parse_args()

response = inspect_system(
    InspectSystemRequest(
        domain=args.domain,
        query=args.query,
        max_entries=args.max_entries,
    )
)
print(json.dumps(response.model_dump(mode="json"), indent=2))
