from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agentic_system.services.preparation import WorkspacePreparation


class WorkdayState(TypedDict, total=False):
    question: str
    route: str
    records: list[dict[str, Any]]
    response: str


def build_workday_graph(preparation: WorkspacePreparation, responder):
    """Build the explicit state graph for one workday request."""

    def route_request(state: WorkdayState) -> WorkdayState:
        question = state["question"].lower()
        if any(word in question for word in ("course", "assignment", "lecture", "deadline")):
            route = "course"
        elif any(word in question for word in ("email", "inbox", "meeting", "reminder")):
            route = "inbox"
        else:
            route = "workday"
        return {"route": route}

    def retrieve_context(state: WorkdayState) -> WorkdayState:
        return {"records": preparation.store.search(state["question"], limit=8)}

    def reason_and_respond(state: WorkdayState) -> WorkdayState:
        return {"response": responder(state["question"], preparation)}

    graph = StateGraph(WorkdayState)
    graph.add_node("route_request", route_request)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("reason_and_respond", reason_and_respond)
    graph.set_entry_point("route_request")
    graph.add_edge("route_request", "retrieve_context")
    graph.add_edge("retrieve_context", "reason_and_respond")
    graph.add_edge("reason_and_respond", END)
    return graph.compile()


def run_workday_graph(question: str, preparation: WorkspacePreparation, responder) -> WorkdayState:
    """Run one request through the explicit LangGraph state machine."""
    return build_workday_graph(preparation, responder).invoke({"question": question})
