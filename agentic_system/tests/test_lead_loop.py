from pathlib import Path

from langchain_core.messages import AIMessage

from agentic_system.services.lead_loop import run_lead_loop
from agentic_system.services.preparation import WorkspacePreparation
from agentic_system.services.workspace_store import WorkspaceStore


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


def test_lead_loop_calls_tool_then_answers(tmp_path: Path):
    preparation = WorkspacePreparation(WorkspaceStore(str(tmp_path / "demo.sqlite")))
    preparation.load_demo_data()
    model = FakeModel()

    answer = run_lead_loop("What should I do next?", preparation, model)

    assert model.calls == 2
    assert "highest priority" in answer
