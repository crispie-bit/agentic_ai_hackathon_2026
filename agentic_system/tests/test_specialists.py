from pathlib import Path

from langchain_core.messages import AIMessage

from agentic_system.services.preparation import WorkspacePreparation
from agentic_system.services.specialists import run_course_specialist
from agentic_system.services.workspace_store import WorkspaceStore


class FakeSpecialistModel:
    def invoke(self, messages):
        return AIMessage(content="The project checkpoint is due Monday at 17:00.")


def test_course_specialist_uses_course_records_only(tmp_path: Path):
    preparation = WorkspacePreparation(WorkspaceStore(str(tmp_path / "demo.sqlite")))
    preparation.load_demo_data()

    result = run_course_specialist("What is due on Monday?", preparation, FakeSpecialistModel())

    assert result.startswith("course specialist:")
    assert "Monday" in result
    assert "DEMO group meeting" not in result
