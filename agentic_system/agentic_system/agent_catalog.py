from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    model_id: str
    tool_names: list[str]
    goal: str
    token_budget: str


AGENT_CATALOG = {
    "lead_agent": AgentSpec(
        name="lead_agent",
        role="central orchestrator",
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        tool_names=["task_router", "priority_planner", "response_summarizer"],
        goal="Delegate tasks, decide urgency, and keep the workflow compact.",
        token_budget="medium",
    ),
    "outlook_agent": AgentSpec(
        name="outlook_agent",
        role="email triage specialist",
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        tool_names=["microsoft_graph_email_query", "message_ranker", "summary_shortener"],
        goal="Filter inbox noise and summarize only urgent or actionable messages.",
        token_budget="low",
    ),
    "ntulearn_agent": AgentSpec(
        name="ntulearn_agent",
        role="course updates specialist",
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        tool_names=["ntulearn_fetcher", "deadline_parser", "announcement_filter"],
        goal="Track deadlines, announcements, and assignment changes with minimal output.",
        token_budget="low",
    ),
    "voice_agent": AgentSpec(
        name="voice_agent",
        role="voice interface specialist",
        model_id="amazon.nova-micro-v1:0",
        tool_names=["speech_to_text", "voice_response_generator"],
        goal="Handle spoken input and produce short spoken responses.",
        token_budget="very_low",
    ),
}

# Recommended usage strategy:
# - keep the lead agent as the only full-context model
# - keep specialist agents narrow and low-cost
# - do not let every agent call the full model unless necessary
