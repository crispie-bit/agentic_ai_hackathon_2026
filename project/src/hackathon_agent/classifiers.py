"""Classifier interfaces for routing tickets.

The project starts with deterministic rules, but the agent depends on this
small interface rather than the implementation. Later branches can plug in an
LLM, a fine-tuned model, or a remote service without changing `ProjectAgent`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .models import RetrievedContext, Route, Ticket
from .tools import classify_route


@dataclass(frozen=True)
class ClassificationResult:
    route: Route
    confidence: float
    rationale: str
    classifier: str = "rules"


class RouteClassifier(Protocol):
    def classify(self, ticket: Ticket, contexts: Sequence[RetrievedContext]) -> ClassificationResult:
        """Classify a ticket using retrieved context."""


class RuleBasedClassifier:
    """Default classifier with no network or API-key dependency."""

    name = "rules"

    def classify(self, ticket: Ticket, contexts: Sequence[RetrievedContext]) -> ClassificationResult:
        route, confidence = classify_route(ticket.text, contexts)
        source = contexts[0].article.title if contexts else "keyword rules"
        return ClassificationResult(
            route=route,
            confidence=confidence,
            rationale=f"Selected from keyword signals and context: {source}.",
            classifier=self.name,
        )


class CallableLLMClassifier:
    """LLM-backed classifier using an injected completion function.

    `complete` receives chat-style messages and returns text. The returned text
    should be JSON with `route`, optional `confidence`, and optional `rationale`.
    This keeps tests and provider-specific integrations simple.
    """

    def __init__(
        self,
        complete: Callable[[list[dict[str, str]]], str],
        *,
        fallback: RouteClassifier | None = None,
        name: str = "llm",
    ) -> None:
        self.complete = complete
        self.fallback = fallback or RuleBasedClassifier()
        self.name = name

    def classify(self, ticket: Ticket, contexts: Sequence[RetrievedContext]) -> ClassificationResult:
        messages = build_classifier_messages(ticket, contexts)
        try:
            raw = self.complete(messages)
            body = _parse_json_object(raw)
            route = Route(str(body["route"]).lower())
            confidence = _clamp(float(body.get("confidence", 0.75)), 0.0, 1.0)
            rationale = str(body.get("rationale") or "LLM classifier selected the route.")
            return ClassificationResult(route, confidence, rationale, self.name)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            result = self.fallback.classify(ticket, contexts)
            return ClassificationResult(
                route=result.route,
                confidence=min(result.confidence, 0.55),
                rationale=f"LLM classifier returned unusable output; fallback used. {exc}",
                classifier=f"{self.name}->fallback",
            )


class OpenAICompatibleClassifier(CallableLLMClassifier):
    """Minimal HTTP adapter for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        fallback: RouteClassifier | None = None,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        super().__init__(self._complete, fallback=fallback, name="openai-compatible")

    def _complete(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        request = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ValueError(f"LLM request failed: {exc}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise ValueError("LLM response had no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("LLM response content was not text")
        return content


def classifier_from_environment() -> OpenAICompatibleClassifier:
    """Build an optional LLM classifier from environment variables."""
    missing = [
        name
        for name in ("HACKATHON_LLM_URL", "HACKATHON_LLM_API_KEY", "HACKATHON_LLM_MODEL")
        if not os.environ.get(name)
    ]
    if missing:
        raise ValueError("Missing environment variables: " + ", ".join(missing))
    return OpenAICompatibleClassifier(
        api_url=os.environ["HACKATHON_LLM_URL"],
        api_key=os.environ["HACKATHON_LLM_API_KEY"],
        model=os.environ["HACKATHON_LLM_MODEL"],
    )


def build_classifier_messages(ticket: Ticket, contexts: Sequence[RetrievedContext]) -> list[dict[str, str]]:
    routes = ", ".join(route.value for route in Route)
    context_text = "\n".join(
        f"- {item.article.title} ({item.article.route.value}, score={item.score:.3f}): {item.article.content}"
        for item in contexts[:4]
    ) or "(no retrieved context)"
    return [
        {
            "role": "system",
            "content": (
                "Classify support tickets. Reply with JSON only: "
                '{"route":"<route>","confidence":0.0,"rationale":"<short reason>"}. '
                f"Allowed routes: {routes}."
            ),
        },
        {
            "role": "user",
            "content": f"TICKET:\n{ticket.text}\n\nCUSTOMER TIER: {ticket.customer.tier}\n\nCONTEXT:\n{context_text}",
        },
    ]


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        stripped = stripped.removesuffix("```").strip()
    body = json.loads(stripped)
    if not isinstance(body, dict):
        raise TypeError("expected a JSON object")
    return body


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
