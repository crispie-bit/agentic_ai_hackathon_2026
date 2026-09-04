import io
import json
import tempfile
import unittest
from pathlib import Path

from hackathon_agent import CallableLLMClassifier, CustomerSignal, JSONLDecisionStore, ProjectAgent, Route, Ticket
from hackathon_agent.evaluation import evaluate, load_cases
from hackathon_agent.knowledge import KnowledgeBase
from hackathon_agent.server import make_handler


class ProjectAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = ProjectAgent()

    def test_routes_unauthorized_payment_to_security(self) -> None:
        decision = self.agent.handle("I see a payment I never made on my statement.")

        self.assertEqual(decision.route, Route.SECURITY)
        self.assertTrue(decision.handoff_required)
        self.assertIn("possible_account_compromise", decision.flags)

    def test_routes_invoice_download_failure_to_technical(self) -> None:
        decision = self.agent.handle("The invoice PDF won't download.")

        self.assertEqual(decision.route, Route.TECHNICAL)
        self.assertIn("reproduction steps", decision.recommended_action)

    def test_routes_subscription_change_to_account(self) -> None:
        decision = self.agent.handle("I can't afford this anymore, close my subscription.")

        self.assertEqual(decision.route, Route.ACCOUNT)
        self.assertFalse(decision.handoff_required)

    def test_priority_customer_sets_handoff(self) -> None:
        ticket = Ticket(
            "The dashboard is blank on Safari.",
            customer=CustomerSignal(tier="enterprise", region="apac"),
        )
        decision = self.agent.handle(ticket)

        self.assertEqual(decision.route, Route.TECHNICAL)
        self.assertTrue(decision.handoff_required)
        self.assertIn("priority_customer", decision.flags)

    def test_knowledge_search_finds_relevant_policy(self) -> None:
        results = KnowledgeBase.default().search("unknown device in active sessions")

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].article.route, Route.SECURITY)

    def test_decision_is_json_serialisable(self) -> None:
        decision = self.agent.handle("Refund the annual charge, I cancelled in June.")

        encoded = json.dumps(decision.to_dict(), sort_keys=True)
        self.assertIn('"route": "billing"', encoded)

    def test_can_load_knowledge_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "knowledge.jsonl"
            path.write_text(
                json.dumps({
                    "id": "custom-security",
                    "title": "Custom security article",
                    "route": "security",
                    "tags": ["token", "leak"],
                    "content": "Leaked tokens and unknown sessions route to security.",
                }) + "\n",
                encoding="utf-8",
            )

            results = KnowledgeBase.from_path(path).search("unknown token leak")

        self.assertEqual(results[0].article.id, "custom-security")

    def test_llm_classifier_can_replace_rule_classifier(self) -> None:
        classifier = CallableLLMClassifier(
            lambda _messages: json.dumps({
                "route": "account",
                "confidence": 0.91,
                "rationale": "The request is an account administration task.",
            })
        )
        decision = ProjectAgent(classifier=classifier).handle("Please change the workspace owner.")

        self.assertEqual(decision.route, Route.ACCOUNT)
        self.assertEqual(decision.classifier, "llm")
        self.assertIn("account administration", decision.rationale)

    def test_llm_classifier_falls_back_on_bad_json(self) -> None:
        decision = ProjectAgent(classifier=CallableLLMClassifier(lambda _messages: "not json")).handle(
            "The invoice PDF won't download."
        )

        self.assertEqual(decision.route, Route.TECHNICAL)
        self.assertEqual(decision.classifier, "llm->fallback")

    def test_llm_classifier_falls_back_on_completion_error(self) -> None:
        def fail(_messages):
            raise ValueError("network unavailable")

        decision = ProjectAgent(classifier=CallableLLMClassifier(fail)).handle("The invoice PDF won't download.")

        self.assertEqual(decision.route, Route.TECHNICAL)
        self.assertEqual(decision.classifier, "llm->fallback")

    def test_decisions_can_be_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JSONLDecisionStore(Path(tmp) / "decisions.jsonl")
            ProjectAgent(decision_store=store).handle("Refund the annual charge, I cancelled in June.")
            records = store.load_all()

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["decision"]["route"], "billing")
        self.assertIn("created_at", records[0])

    def test_default_evaluation_cases_score_cleanly(self) -> None:
        report = evaluate(self.agent, load_cases())

        self.assertEqual(report.total, 8)
        self.assertGreaterEqual(report.accuracy, 0.875)

    def test_http_handle_endpoint_returns_decision(self) -> None:
        body = json.dumps({"ticket": "Invoice PDF will not download.", "channel": "api"}).encode("utf-8")
        request = (
            b"POST /handle HTTP/1.1\r\n"
            b"Host: test\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            + body
        )
        socket = InMemorySocket(request)

        make_handler(self.agent)(socket, ("127.0.0.1", 0), object())
        _headers, _separator, response_body = socket.output.getvalue().partition(b"\r\n\r\n")
        payload = json.loads(response_body.decode("utf-8"))

        self.assertIn(b"200 OK", socket.output.getvalue())
        self.assertEqual(payload["route"], "technical")


class InMemorySocket:
    def __init__(self, request: bytes):
        self.input = io.BytesIO(request)
        self.output = io.BytesIO()

    def makefile(self, mode: str, *_args, **_kwargs):
        if "r" in mode:
            return self.input
        return self.output

    def sendall(self, data: bytes) -> None:
        self.output.write(data)


if __name__ == "__main__":
    unittest.main()
