from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agentic_system.services.preparation import WorkspacePreparation


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

    tools = [get_workday_tasks, search_workspace]
    by_name = {tool_item.name: tool_item for tool_item in tools}
    bound_model = model.bind_tools(tools)
    messages = [
        SystemMessage(
            "You are the lead student workday agent. Work in a bounded loop. "
            "Use get_workday_tasks for planning questions and search_workspace for "
            "specific course facts. After observing tool results, answer concisely "
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
