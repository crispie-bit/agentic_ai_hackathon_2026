from __future__ import annotations

from typing import Any

from agentic_system.agents.lead_agent import LeadAgent
from agentic_system.agents.ntulearn_agent import NTULearnAgent
from agentic_system.agents.outlook_agent import OutlookAgent


class AgentOrchestrator:
    def __init__(self):
        self.lead_agent = LeadAgent()
        self.agents: dict[str, Any] = {}

    def register_default_agents(self) -> None:
        self.agents["lead_agent"] = self.lead_agent
        self.agents["outlook_agent"] = OutlookAgent()
        self.agents["ntulearn_agent"] = NTULearnAgent()

    def route(self, request: str) -> list[str]:
        lower = request.lower()
        delegated: list[str] = []

        if "email" in lower or "outlook" in lower or "inbox" in lower:
            delegated.append("outlook_agent")

        if "ntulearn" in lower or "course" in lower or "assignment" in lower or "announcement" in lower:
            delegated.append("ntulearn_agent")

        if not delegated:
            delegated.append("lead_agent")

        return delegated

    def run(self, request: str) -> str:
        delegated = self.route(request)
        outputs = []

        for agent_name in delegated:
            agent = self.agents.get(agent_name, self.lead_agent)
            response = agent.run(request, {"delegated_to": delegated})
            outputs.append(f"[{agent.name}] {response}")

        if delegated == ["lead_agent"]:
            return self.lead_agent.run(request, {"delegated_to": delegated})

        return "\n".join(outputs)
