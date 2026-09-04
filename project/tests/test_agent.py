import json
import unittest

from hackathon_agent import CustomerSignal, ProjectAgent, Route, Ticket
from hackathon_agent.knowledge import KnowledgeBase


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


if __name__ == "__main__":
    unittest.main()

