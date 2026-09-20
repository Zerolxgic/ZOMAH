from __future__ import annotations

import json

from zomah.capabilities import GetProjectStateRequest, get_project_state
from zomah.model_boundary import invoke_model_capability
from zomah.state import ProjectStateRepository, default_database_path
from zomah.tracing import TraceStore, default_trace_db_path


repository = ProjectStateRepository(default_database_path())
trace_store = TraceStore(default_trace_db_path())

envelope = invoke_model_capability(
    GetProjectStateRequest,
    get_project_state,
    {"project_id": "zomah"},
    worker="elyria-demo",
    trace_store=trace_store,
    repository=repository,
)

print(json.dumps(envelope.model_dump(mode="json"), indent=2))
print("\nLatest trace:")
print(json.dumps(trace_store.recent(limit=1)[0].model_dump(mode="json"), indent=2))
