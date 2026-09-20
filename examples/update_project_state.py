from zomah.capabilities import (
    GetProjectStateRequest,
    UpdateProjectStateRequest,
    get_project_state,
    update_project_state,
)
from zomah.state import ProjectStatePatch, ProjectStateRepository, default_database_path


repo = ProjectStateRepository(default_database_path())
current = get_project_state(
    GetProjectStateRequest(project_id="zomah"),
    repository=repo,
).project

response = update_project_state(
    UpdateProjectStateRequest(
        project_id="zomah",
        patch=ProjectStatePatch(
            expected_revision=current.revision,
            updated_by="zerrius",
            current_focus="Validate the ProjectState read/write capability loop.",
            last_action="Implemented and verified get_project_state against canonical SQLite state.",
            next_action="Verify update_project_state, then choose the next minimal read capability.",
        ),
    ),
    repository=repo,
)

print(response.model_dump_json(indent=2))
