from pathlib import Path

from langchain_core.messages import AIMessage

from agentic_system.services.lead_loop import run_lead_loop
from agentic_system.services.preparation import WorkspacePreparation
from agentic_system.services.workspace_store import WorkspaceStore
from agentic_system.workday_graph import run_workday_graph


class FakeModel:
    def __init__(self):
        self.tools = []
        self.calls = 0

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_workday_tasks",
                        "args": {},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="Start the project checkpoint because it has the highest priority.")


class ConstraintFakeModel(FakeModel):
    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_available_time",
                        "args": {},
                        "id": "call-time",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="Choose the 30-minute group evaluation plan.")


def test_lead_loop_calls_tool_then_answers(tmp_path: Path):
    preparation = WorkspacePreparation(WorkspaceStore(str(tmp_path / "demo.sqlite")))
    preparation.load_demo_data()
    model = FakeModel()

    answer = run_lead_loop("What should I do next?", preparation, model)

    assert model.calls == 2
    assert "highest priority" in answer


def test_lead_loop_replans_with_available_time(tmp_path: Path):
    preparation = WorkspacePreparation(WorkspaceStore(str(tmp_path / "demo.sqlite")))
    preparation.load_demo_data()

    answer = run_lead_loop("I only have 30 minutes. What should I do?", preparation, ConstraintFakeModel())

    assert "30-minute" in answer


def test_workday_graph_routes_course_question(tmp_path: Path):
    preparation = WorkspacePreparation(WorkspaceStore(str(tmp_path / "demo.sqlite")))
    preparation.load_demo_data()

    state = run_workday_graph(
        "What course deadline do I have?",
        preparation,
        lambda question, workspace: "course response",
    )

    assert state["route"] == "course"
    assert state["records"]
    assert state["response"] == "course response"
