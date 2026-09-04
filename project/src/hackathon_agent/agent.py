"""Project-level agent facade."""

from __future__ import annotations

from .classifiers import RouteClassifier, RuleBasedClassifier
from .knowledge import KnowledgeBase
from .models import AgentDecision, CustomerSignal, Ticket
from .storage import DecisionStore
from .tools import build_next_steps, build_summary, detect_flags, recommend_action


class ProjectAgent:
    """Handle one support-style request and return a structured decision."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        *,
        classifier: RouteClassifier | None = None,
        decision_store: DecisionStore | None = None,
        retrieval_top_k: int = 3,
        retrieval_min_score: float = 0.04,
    ) -> None:
        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be at least 1")
        if retrieval_min_score < 0:
            raise ValueError("retrieval_min_score cannot be negative")

        self.knowledge_base = knowledge_base or KnowledgeBase.default()
        self.classifier = classifier or RuleBasedClassifier()
        self.decision_store = decision_store
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_min_score = retrieval_min_score

    def handle(
        self,
        ticket: str | Ticket,
        *,
        customer: CustomerSignal | None = None,
    ) -> AgentDecision:
        """Process a ticket string or `Ticket` object.

        The return value is intentionally serialisable and stable, so UI/API
        branches can build against it while the internals continue to evolve.
        """
        request = ticket if isinstance(ticket, Ticket) else Ticket(ticket, customer=customer or CustomerSignal())
        contexts = self.knowledge_base.search(
            request.text,
            top_k=self.retrieval_top_k,
            min_score=self.retrieval_min_score,
        )
        classification = self.classifier.classify(request, contexts)
        route = classification.route
        confidence = classification.confidence
        flags = detect_flags(request, route)
        decision = AgentDecision(
            route=route,
            confidence=confidence,
            summary=build_summary(request.text),
            recommended_action=recommend_action(route, flags, contexts),
            next_steps=build_next_steps(route, flags, contexts),
            retrieved_context=tuple(contexts),
            rationale=classification.rationale,
            classifier=classification.classifier,
            flags=tuple(flags),
            handoff_required="possible_account_compromise" in flags or request.customer.tier.lower() in {"enterprise", "vip"},
        )
        if self.decision_store is not None:
            self.decision_store.append(request, decision)
        return decision
