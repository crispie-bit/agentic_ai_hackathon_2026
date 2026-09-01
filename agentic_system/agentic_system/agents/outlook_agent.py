from __future__ import annotations

from typing import Any

from agentic_system.base_agent import BaseAgent
from agentic_system.config import ENABLE_OUTLOOK


class OutlookAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="outlook_agent",
            description="Focuses only on urgent emails, filtering noise and identifying actionable senders.",
            capabilities=["email_triage", "urgency_detection", "action_summary"],
        )

    def run(self, request: str, context: dict[str, Any] | None = None) -> str:
        if not ENABLE_OUTLOOK:
            return (
                "Outlook agent is enabled in the architecture but disabled in this setup. "
                "It will later query Microsoft Graph for only urgent and actionable emails."
            )

        return (
            f"Outlook agent reviewed: '{request}'. "
            "It would rank messages by urgency, filter spam and newsletters, and return only the top action items."
        )
