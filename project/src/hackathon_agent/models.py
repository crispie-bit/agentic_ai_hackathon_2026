"""Typed data contracts for the project starter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Route(str, Enum):
    ACCOUNT = "account"
    BILLING = "billing"
    GENERAL = "general"
    SECURITY = "security"
    TECHNICAL = "technical"


@dataclass(frozen=True)
class CustomerSignal:
    name: str = "customer"
    tier: str = "standard"
    region: str | None = None
    account_age_days: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "region": self.region,
            "account_age_days": self.account_age_days,
        }


@dataclass(frozen=True)
class Ticket:
    text: str
    customer: CustomerSignal = field(default_factory=CustomerSignal)
    channel: str = "chat"

    def __post_init__(self) -> None:
        cleaned = " ".join(self.text.split())
        if not cleaned:
            raise ValueError("Ticket text cannot be empty")
        object.__setattr__(self, "text", cleaned)


@dataclass(frozen=True)
class KnowledgeArticle:
    id: str
    title: str
    route: Route
    content: str
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "route": self.route.value,
            "content": self.content,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class RetrievedContext:
    article: KnowledgeArticle
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.article.id,
            "title": self.article.title,
            "route": self.article.route.value,
            "score": round(self.score, 4),
            "snippet": self.article.content[:220],
        }


@dataclass(frozen=True)
class AgentDecision:
    route: Route
    confidence: float
    summary: str
    recommended_action: str
    next_steps: tuple[str, ...]
    retrieved_context: tuple[RetrievedContext, ...]
    rationale: str = ""
    classifier: str = "rules"
    flags: tuple[str, ...] = ()
    handoff_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route.value,
            "confidence": round(self.confidence, 3),
            "summary": self.summary,
            "recommended_action": self.recommended_action,
            "next_steps": list(self.next_steps),
            "flags": list(self.flags),
            "handoff_required": self.handoff_required,
            "rationale": self.rationale,
            "classifier": self.classifier,
            "retrieved_context": [context.to_dict() for context in self.retrieved_context],
        }
