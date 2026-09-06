from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agentic_system.services.preparation import WorkspacePreparation
from agentic_system.services.specialists import run_course_specialist, run_inbox_specialist


MAX_STEPS = 4


def run_lead_loop(question: str, preparation: WorkspacePreparation, model: Any) -> str:
    """Run a bounded model/tool/observation loop over the student workspace."""

    @tool
    def get_workday_tasks() -> str:
        """Read the currently available tasks, deadlines, effort, and priorities."""
        return json.dumps([task.__dict__ for task in preparation.get_tasks()])

    @tool
    def search_workspace(query: str) -> str:
        """Search grounded course and calendar records for a specific question."""
        return json.dumps(preparation.store.search(query, limit=5))

    @tool
    def get_available_time() -> str:
        """Read the student's currently available focus time for replanning."""
        return json.dumps({"available_minutes": 30, "constraint": "student requested a short focus block"})

    @tool
    def ask_course_specialist(question: str) -> str:
        """Delegate course, assignment, lecture, or deadline questions to the course specialist."""
        return run_course_specialist(question, preparation, model)

    @tool
    def ask_inbox_specialist(question: str) -> str:
        """Delegate email, meeting, or reminder questions to the inbox specialist."""
        return run_inbox_specialist(question, preparation, model)

    tools = [
        get_workday_tasks,
        search_workspace,
        get_available_time,
        ask_course_specialist,
        ask_inbox_specialist,
    ]
    by_name = {tool_item.name: tool_item for tool_item in tools}
    bound_model = model.bind_tools(tools)
    messages = [
        SystemMessage(
            "You are the lead student workday agent. Work in a bounded loop. "
            "Use get_workday_tasks for planning questions, get_available_time when "
            "the student mentions a time constraint, and search_workspace for "
            "specific facts. Delegate course questions to ask_course_specialist and "
            "email or meeting questions to ask_inbox_specialist. After observing "
            "tool results, answer concisely "
            "with a recommendation and reason. Never invent tasks, dates, or sources."
        ),
        HumanMessage(question),
    ]

    for _ in range(MAX_STEPS):
        reply = bound_model.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return str(reply.content).strip()

        for call in reply.tool_calls:
            tool_item = by_name.get(call["name"])
            if tool_item is None:
                result = f"Unknown tool: {call['name']}"
            else:
                result = tool_item.invoke(call.get("args", {}))
            messages.append(ToolMessage(str(result), tool_call_id=call["id"]))

    return "I could not complete the workday plan within the allowed steps."
