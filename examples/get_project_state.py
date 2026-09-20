from zomah.capabilities import GetProjectStateRequest, get_project_state
from zomah.state import ProjectStateRepository, default_database_path


repo = ProjectStateRepository(default_database_path())
response = get_project_state(
    GetProjectStateRequest(project_id="zomah"),
    repository=repo,
)

print(response.model_dump_json(indent=2))
