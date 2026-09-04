"""Evaluation helpers for the project agent."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import ProjectAgent
from .models import CustomerSignal, Route, Ticket

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_PATH = PROJECT_ROOT / "data" / "evaluation_cases.jsonl"


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    text: str
    route: Route
    tier: str = "standard"
    region: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationCase":
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            route=Route(str(data["route"]).lower()),
            tier=str(data.get("tier", "standard")),
            region=data.get("region"),
        )

    def to_ticket(self) -> Ticket:
        return Ticket(self.text, customer=CustomerSignal(tier=self.tier, region=self.region))


@dataclass(frozen=True)
class EvaluationReport:
    total: int
    correct: int
    mistakes: tuple[dict[str, Any], ...]
    confusion: dict[str, dict[str, int]]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "mistakes": list(self.mistakes),
            "confusion": self.confusion,
        }


def load_cases(path: str | Path = DEFAULT_EVALUATION_PATH) -> tuple[EvaluationCase, ...]:
    cases = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(EvaluationCase.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid evaluation case at {path}:{line_number}: {exc}") from exc
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return tuple(cases)


def evaluate(agent: ProjectAgent, cases: Iterable[EvaluationCase]) -> EvaluationReport:
    total = 0
    correct = 0
    mistakes = []
    confusion_counts: Counter[tuple[str, str]] = Counter()

    for case in cases:
        total += 1
        decision = agent.handle(case.to_ticket())
        got = decision.route.value
        want = case.route.value
        confusion_counts[(want, got)] += 1
        if decision.route == case.route:
            correct += 1
            continue
        mistakes.append({
            "id": case.id,
            "expected": want,
            "actual": got,
            "text": case.text,
            "confidence": round(decision.confidence, 3),
        })

    confusion: dict[str, dict[str, int]] = {}
    for (want, got), count in confusion_counts.items():
        confusion.setdefault(want, {})[got] = count
    return EvaluationReport(total=total, correct=correct, mistakes=tuple(mistakes), confusion=confusion)

