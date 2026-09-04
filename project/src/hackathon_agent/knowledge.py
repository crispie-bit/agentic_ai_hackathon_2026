"""Small local retrieval layer for project context."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from .models import KnowledgeArticle, RetrievedContext, Route

TOKEN_RE = re.compile(r"[a-z0-9']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "with",
    "you",
    "your",
}


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS)


def vectorize(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    numerator = sum(weight * b.get(term, 0) for term, weight in a.items())
    denominator = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return numerator / denominator if denominator else 0.0


class KnowledgeBase:
    """In-memory article search.

    This is deliberately simple. It gives later branches a contract to replace
    with embeddings or a vector database without touching `ProjectAgent`.
    """

    def __init__(self, articles: Iterable[KnowledgeArticle]) -> None:
        self.articles = tuple(articles)
        if not self.articles:
            raise ValueError("KnowledgeBase needs at least one article")
        self._vectors = {
            article.id: vectorize(" ".join((article.title, article.route.value, " ".join(article.tags), article.content)))
            for article in self.articles
        }

    @classmethod
    def default(cls) -> "KnowledgeBase":
        return cls(default_articles())

    def search(self, query: str, *, top_k: int = 3, min_score: float = 0.04) -> list[RetrievedContext]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_vector = vectorize(query)
        ranked = sorted(
            (
                RetrievedContext(article=article, score=cosine(query_vector, self._vectors[article.id]))
                for article in self.articles
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return [item for item in ranked[:top_k] if item.score >= min_score]


def default_articles() -> tuple[KnowledgeArticle, ...]:
    return (
        KnowledgeArticle(
            id="billing-dispute",
            title="Billing disputes and refunds",
            route=Route.BILLING,
            tags=("charge", "invoice", "refund", "currency", "tax"),
            content=(
                "Use billing when the customer disputes an amount, currency, tax, invoice details, "
                "refund timing, or pricing. If the customer reports an unauthorized payment, route "
                "to security first and let billing follow after the account is safe."
            ),
        ),
        KnowledgeArticle(
            id="security-compromise",
            title="Possible account compromise",
            route=Route.SECURITY,
            tags=("unauthorized", "device", "session", "statement", "charge"),
            content=(
                "Use security when a customer sees a payment they did not make, an unknown device, "
                "an unknown active session, credential changes they did not request, or any sign of "
                "account takeover. Verify identity before exposing details."
            ),
        ),
        KnowledgeArticle(
            id="technical-failure",
            title="Broken product behavior",
            route=Route.TECHNICAL,
            tags=("error", "download", "upload", "blank", "crash", "button"),
            content=(
                "Use technical when a product feature fails, throws an error, will not load, will "
                "not upload or download, renders incorrectly, or produces unexpected output."
            ),
        ),
        KnowledgeArticle(
            id="account-change",
            title="Account and subscription changes",
            route=Route.ACCOUNT,
            tags=("subscription", "seat", "user", "email", "login", "cancel"),
            content=(
                "Use account when the user wants to cancel or change a subscription, add or remove "
                "users, change login details, update an email address, or manage account access."
            ),
        ),
        KnowledgeArticle(
            id="human-handoff",
            title="Human handoff criteria",
            route=Route.GENERAL,
            tags=("handoff", "urgent", "enterprise", "vip", "risk"),
            content=(
                "Escalate when the request indicates account compromise, legal or safety risk, "
                "an enterprise or VIP customer, repeated failure, or a side effect that needs approval."
            ),
        ),
    )

