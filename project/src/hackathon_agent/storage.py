"""Persistence for agent decisions."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .models import AgentDecision, Ticket


class DecisionStore(Protocol):
    def append(self, ticket: Ticket, decision: AgentDecision) -> None:
        """Persist one decision."""


class JSONLDecisionStore:
    """Append-only JSONL audit log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, ticket: Ticket, decision: AgentDecision) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self._record(ticket, decision), sort_keys=True) + "\n")

    def load_all(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        records = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return tuple(records)

    def extend(self, records: Iterable[tuple[Ticket, AgentDecision]]) -> None:
        for ticket, decision in records:
            self.append(ticket, decision)

    @staticmethod
    def _record(ticket: Ticket, decision: AgentDecision) -> dict[str, Any]:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ticket": {
                "text": ticket.text,
                "channel": ticket.channel,
                "customer": ticket.customer.to_dict(),
            },
            "decision": decision.to_dict(),
        }

