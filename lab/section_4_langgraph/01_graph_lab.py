"""
§4 · 01 — A generate/evaluate loop as a graph.

    uv run section_4_langgraph/01_graph_lab.py

Summarise a paragraph in one sentence, have it checked, rewrite it.
Bounded by MAX_ROUNDS.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import _bootstrap  # noqa: F401

from _common import banner, chat_model, model_label

MAX_ROUNDS = 2
WORD_LIMIT = 20

TEXT = ("The 09:40 flight from Singapore to Bangkok was delayed by four hours "
        "after a hydraulic fault was found during pre-flight checks. Around "
        "180 passengers were rebooked onto two later services, and the airline "
        "said meal vouchers were issued at the gate.")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    text: str
    draft: str
    critique: str
    rounds: int


def generate(state: AgentState) -> dict:
    """Write the summary. If there is a critique, fix it."""
    ask = (f"Summarise this in ONE sentence of at most {WORD_LIMIT} words. "
           f"Reply with the sentence only.\n\n{state['text']}")
    if state.get("critique"):
        ask += f"\n\nPrevious: {state['draft']}\nFix this: {state['critique']}"

    reply = chat_model(temperature=0.3).invoke([HumanMessage(content=ask)])

    return {
        "draft": reply.content,
        "rounds": state.get("rounds", 0) + 1,
        "messages": [AIMessage(content=reply.content)],
    }


def evaluate(state: AgentState) -> dict:
    """Check the summary against the source. ACCEPT means stop."""
    words = len(state["draft"].split())
    if words > WORD_LIMIT:                       # a check Python can make
        return {"critique": f"Too long: {words} words, limit is {WORD_LIMIT}."}

    reply = chat_model(temperature=0.0).invoke([HumanMessage(
        content="Does this summary keep the key facts and add nothing that is "
                "not in the source? If yes, reply with exactly ACCEPT. If no, "
                f"give ONE specific fix.\n\nSource: {state['text']}"
                f"\n\nSummary: {state['draft']}")])

    verdict = reply.content.strip()
    return {"critique": "" if verdict.upper().startswith("ACCEPT") else verdict}


def should_continue(state: AgentState) -> str:
    if not state.get("critique"):
        return "end"

    if state["rounds"] >= MAX_ROUNDS:
        return "end"

    return "continue"


def build():
    builder = StateGraph(AgentState)

    builder.add_node("generate", generate)
    builder.add_node("evaluate", evaluate)

    builder.add_edge(START, "generate")
    builder.add_edge("generate", "evaluate")
    builder.add_conditional_edges(
        "evaluate", should_continue,
        {"continue": "generate", "end": END},
    )

    return builder.compile()


def main() -> None:
    print(f"model: {model_label()}")
    app = build()

    banner("THE GRAPH")
    try:
        print(app.get_graph().draw_ascii())
    except Exception:
        print("  (uv add grandalf for the ASCII diagram)")
    print("  evaluate -> generate is the back-edge (not drawn above)")

    banner("INPUT")
    print(TEXT)

    final = app.invoke({
        "text": TEXT,
        "messages": [HumanMessage(content=TEXT)],
        "rounds": 0,
    })

    banner("OUTPUT")
    print(final["draft"])
    print(f"\n  words: {len(final['draft'].split())}   rounds: {final['rounds']}"
          f"   messages: {len(final['messages'])}")


if __name__ == "__main__":
    main()
