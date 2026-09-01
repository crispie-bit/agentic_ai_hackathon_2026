from __future__ import annotations

from typing import Any

from agentic_system.base_agent import BaseAgent
from agentic_system.config import ENABLE_NTU_LEARN


class NTULearnAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ntulearn_agent",
            description="Tracks only course announcements, deadlines, and assignment changes relevant to the student.",
            capabilities=["announcement_scan", "deadline_tracking", "assignment_monitoring"],
        )

    def run(self, request: str, context: dict[str, Any] | None = None) -> str:
        if not ENABLE_NTU_LEARN:
            return (
                "NTULearn agent is built for a narrow educational workflow but stays disabled in this setup. "
                "It will later check only announcements, deadlines, and assignment updates."
            )

        return (
            f"NTULearn agent reviewed: '{request}'. "
            "It would filter course updates to critical posts, due dates, and assignment changes only."
        )
