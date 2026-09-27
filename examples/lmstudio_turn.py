"""Send one turn to a running LM Studio server through WorkerSession.

Usage:
    python examples/lmstudio_turn.py --model <exact model id> "Your message"
"""

import argparse
import asyncio

from zomah.lmstudio import DEFAULT_BASE_URL, LMStudioRuntime
from zomah.model_runtime import ModelRuntimeError
from zomah.worker_session import WorkerSession

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="exact id from GET /v1/models")
parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
parser.add_argument("--timeout", type=float, default=120.0)
parser.add_argument("message")
args = parser.parse_args()

session = WorkerSession(
    LMStudioRuntime(base_url=args.base_url, timeout=args.timeout),
    worker="elyria",
    model=args.model,
)

try:
    result = asyncio.run(session.send(args.message))
except ModelRuntimeError as error:
    raise SystemExit(f"error: {error} (retryable: {error.retryable})")

print(f"assistant: {result.assistant.text}")
if result.usage is None:
    print("usage: not reported by the runtime")
else:
    print(
        f"usage: input_tokens={result.usage.input_tokens} "
        f"output_tokens={result.usage.output_tokens} "
        f"context_used={session.context_used}"
    )
