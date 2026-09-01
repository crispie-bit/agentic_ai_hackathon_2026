from __future__ import annotations

from typing import Any

from agentic_system.base_agent import BaseAgent


class LeadAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="lead_agent",
            description="Main coordinator that routes tasks to specialist agents and keeps the workflow concise.",
            capabilities=["delegation", "prioritization", "planning", "response_summarization"],
        )

    def run(self, request: str, context: dict[str, Any] | None = None) -> str:
        context = context or {}
        delegated = context.get("delegated_to", [])

        if not delegated:
            return (
                "Lead agent is online. It will select the most relevant specialist, keep the "
                "workflow efficient, and return only the action items that matter."
            )

        summary = "; ".join(delegated)
        return f"Lead agent delegated this request to: {summary}."
