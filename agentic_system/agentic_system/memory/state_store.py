from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMemory:
    user_profile: dict[str, Any] = field(default_factory=dict)
    tasks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)


class MemoryStore:
    def __init__(self):
        self.state = AgentMemory()

    def add_task(self, task: str) -> None:
        self.state.tasks.append(task)

    def add_note(self, note: str) -> None:
        self.state.notes.append(note)

    def add_alert(self, alert: str) -> None:
        self.state.alerts.append(alert)

    def snapshot(self) -> dict[str, Any]:
        return {
            "user_profile": self.state.user_profile,
            "tasks": list(self.state.tasks),
            "notes": list(self.state.notes),
            "alerts": list(self.state.alerts),
        }
