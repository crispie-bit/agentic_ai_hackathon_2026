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

    @staticmethod
    def _is_urgent(message: dict[str, Any]) -> bool:
        subject = str(message.get("subject", "")).lower()
        sender = str(message.get("sender", "")).lower()
        body = str(message.get("body", "")).lower()
        text = f"{subject} {sender} {body}"

        if any(keyword in text for keyword in ["weekly digest", "newsletter", "promo", "marketing", "spam", "sale"]):
            return False

        urgency_terms = [
            "urgent",
            "action required",
            "respond by",
            "reply by",
            "due today",
            "deadline",
            "today",
            "asap",
            "immediately",
            "review",
            "approve",
            "meeting",
            "call me",
            "follow up",
            "please review",
        ]
        return any(term in text for term in urgency_terms)

    def run(self, request: str, context: dict[str, Any] | None = None) -> str:
        context = context or {}
        messages = context.get("messages") or context.get("emails") or []

        if not messages:
            if not ENABLE_OUTLOOK:
                return (
                    "Outlook agent is enabled in the architecture but disabled in this setup. "
                    "It will later query Microsoft Graph for only urgent and actionable emails."
                )
            return (
                f"Outlook agent reviewed: '{request}'. "
                "No messages were provided, so there were no urgent items to triage."
            )

        urgent_messages = [message for message in messages if self._is_urgent(message)]

        if not urgent_messages:
            return (
                "Outlook agent filtered the inbox and found no urgent action items. "
                "Routine newsletters, bulk updates, and non-actionable messages were ignored."
            )

        summary_lines = []
        for message in urgent_messages[:3]:
            subject = str(message.get("subject", "No subject"))
            sender = str(message.get("sender", "Unknown sender"))
            summary_lines.append(f"{subject} from {sender}")

        summary = "; ".join(summary_lines)
        if not ENABLE_OUTLOOK:
            return (
                f"Outlook agent reviewed the available inbox items and flagged urgent messages: {summary}. "
                "Live Microsoft Graph sync remains disabled in this setup."
            )

        return f"Outlook agent found urgent emails: {summary}."
