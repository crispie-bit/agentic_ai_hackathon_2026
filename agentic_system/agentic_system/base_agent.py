from __future__ import annotations

from typing import Any


class BaseAgent:
    def __init__(self, name: str, description: str, capabilities: list[str] | None = None):
        self.name = name
        self.description = description
        self.capabilities = capabilities or []

    def run(self, request: str, context: dict[str, Any] | None = None) -> str:
        raise NotImplementedError("Subclasses must define run().")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
