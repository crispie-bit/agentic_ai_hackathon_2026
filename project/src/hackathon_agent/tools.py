"""Deterministic tools used by the starter agent."""

from __future__ import annotations

from .models import RetrievedContext, Route, Ticket

ROUTE_PATTERNS: dict[Route, tuple[tuple[str, float], ...]] = {
    Route.SECURITY: (
        ("never made", 6.0),
        ("did not make", 6.0),
        ("don't recognise", 5.0),
        ("don't recognize", 5.0),
        ("do not recognise", 5.0),
        ("do not recognize", 5.0),
        ("device i don't own", 5.0),
        ("unknown device", 4.0),
        ("active sessions", 4.0),
        ("unauthorized", 5.0),
        ("unauthorised", 5.0),
        ("suspicious", 3.0),
    ),
    Route.TECHNICAL: (
        ("500 error", 6.0),
        ("won't download", 6.0),
        ("will not download", 6.0),
        ("throws", 3.0),
        ("error", 3.0),
        ("blank", 4.0),
        ("does nothing", 4.0),
        ("fail", 3.0),
        ("fails", 3.0),
        ("upload", 2.0),
        ("download", 2.0),
        ("crash", 4.0),
        ("broken", 3.0),
    ),
    Route.BILLING: (
        ("charged", 4.0),
        ("charge", 2.0),
        ("invoice", 2.0),
        ("refund", 4.0),
        ("payment", 2.0),
        ("currency", 3.0),
        ("tax", 3.0),
        ("vat", 3.0),
        ("price", 2.0),
        ("receipt", 2.0),
    ),
    Route.ACCOUNT: (
        ("close my subscription", 6.0),
        ("cancel", 4.0),
        ("subscription", 3.0),
        ("add my colleague", 5.0),
        ("second user", 4.0),
        ("change the email", 5.0),
        ("email on my login", 4.0),
        ("login details", 3.0),
        ("seat", 3.0),
        ("user", 2.0),
        ("plan", 1.5),
    ),
}


def classify_route(text: str, contexts: tuple[RetrievedContext, ...] | list[RetrievedContext] = ()) -> tuple[Route, float]:
    """Return the most likely route and a confidence estimate."""
    lowered = text.lower()
    scores = {route: 0.0 for route in Route}

    for route, patterns in ROUTE_PATTERNS.items():
        for phrase, weight in patterns:
            if phrase in lowered:
                scores[route] += weight

    for context in contexts[:2]:
        scores[context.article.route] += max(0.25, context.score * 3)

    best = max(scores, key=scores.get)
    best_score = scores[best]
    if best_score == 0:
        return Route.GENERAL, 0.35

    runner_up = max(score for route, score in scores.items() if route != best)
    margin = best_score - runner_up
    confidence = min(0.95, 0.55 + (margin / max(best_score, 1.0)) * 0.35)
    return best, max(0.45, confidence)


def build_summary(text: str, *, max_words: int = 24) -> str:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def detect_flags(ticket: Ticket, route: Route) -> list[str]:
    lowered = ticket.text.lower()
    flags: list[str] = []
    if route is Route.SECURITY:
        flags.append("possible_account_compromise")
    if route is Route.BILLING and any(word in lowered for word in ("refund", "charged", "charge")):
        flags.append("needs_billing_review")
    if any(word in lowered for word in ("urgent", "asap", "immediately", "cannot access", "can't access")):
        flags.append("time_sensitive")
    if ticket.customer.tier.lower() in {"enterprise", "vip"}:
        flags.append("priority_customer")
    return flags


def recommend_action(route: Route, flags: list[str], contexts: tuple[RetrievedContext, ...] | list[RetrievedContext]) -> str:
    if "possible_account_compromise" in flags:
        return "Escalate to security, verify identity, and pause account-changing actions until reviewed."
    if route is Route.BILLING:
        return "Route to billing with the disputed amount, invoice or receipt details, and requested resolution."
    if route is Route.TECHNICAL:
        return "Route to engineering support with reproduction steps, browser or device details, and recent timestamps."
    if route is Route.ACCOUNT:
        return "Route to account support to validate ownership and complete the requested subscription or user change."
    if contexts:
        return "Use the retrieved context to draft a response, then ask for missing details before taking action."
    return "Ask one clarifying question and keep the ticket in general triage."


def build_next_steps(route: Route, flags: list[str], contexts: tuple[RetrievedContext, ...] | list[RetrievedContext]) -> tuple[str, ...]:
    steps = []
    if "priority_customer" in flags:
        steps.append("Apply priority handling and note the customer tier.")
    if "possible_account_compromise" in flags:
        steps.extend(("Verify identity through the approved flow.", "Collect device, session, and payment evidence."))
    elif route is Route.TECHNICAL:
        steps.extend(("Ask for reproduction steps and environment details.", "Attach logs or screenshots if available."))
    elif route is Route.BILLING:
        steps.extend(("Confirm the disputed invoice, charge, or receipt.", "Check refund and pricing policy before responding."))
    elif route is Route.ACCOUNT:
        steps.extend(("Confirm account ownership.", "Identify the subscription, user, or login detail to change."))
    else:
        steps.append("Ask for the missing detail that would determine the route.")

    if contexts:
        steps.append(f"Reference policy context: {contexts[0].article.title}.")
    return tuple(steps)

