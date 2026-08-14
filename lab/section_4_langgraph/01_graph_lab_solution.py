"""
§4 · 01 — SOLUTION. The finished graph, runnable.

    uv run section_4_langgraph/01_graph_lab_solution.py

All three TODOs completed. Read 01_graph_lab.py first and try them; this is
here to run when you are stuck, or to diff against your own version:

    diff section_4_langgraph/01_graph_lab.py section_4_langgraph/01_graph_lab_solution.py

Each answer is marked with  # <-- TODO n  below.

WHAT THE THREE TODOs WERE ABOUT

  1. add_node AND add_edge are separate calls. Miss the edge and the node
     never runs — no error, no warning, just a step that silently does not
     happen. That is the most common LangGraph bug. (Miss the NODE, as the
     lab file does, and it fails loudly at compile() instead. Loud is better.)

  2. `evaluate -> generate` is a back-edge, and a back-edge runs forever on
     its own. Two things end this loop, and they differ in kind:
         the reviewer says ACCEPT   a JUDGEMENT, made by a model
         rounds >= MAX_ROUNDS       a GUARANTEE, made by you
     Never rely on the first. A stricter critic, a worse model, an ambiguous
     task — and ACCEPT never comes.

  3. A reducer is a function (old, new) -> merged attached to one state key.
     add_messages appends; without it, every node's return REPLACES messages.
     You need your own reducer the moment two nodes can write one key in the
     same step, because otherwise the last writer silently wins.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import _bootstrap  # noqa: F401

from _common import banner, chat_model, model_label

MAX_ROUNDS = 2

TICKET = ("Ticket #4412: the office printer on level 3 has been offline since "
          "Monday. I've tried turning it off and on. I need to print a client "
          "deck before Thursday.")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # <-- TODO 3: the reducer appends
    ticket: str        # seeded by the caller
    draft: str         # written by generate
    critique: str      # written by evaluate
    rounds: int        # incremented by generate; this is what bounds the loop


def generate(state: AgentState) -> dict:
    """Draft a reply. If there is a critique, address it."""
    model = chat_model(temperature=0.3)

    ask = f"Reply to this support ticket in 2-3 sentences.\n\n{state['ticket']}"
    if state.get("critique"):
        ask += (f"\n\nYour previous reply:\n{state['draft']}"
                f"\n\nFix this:\n{state['critique']}")

    reply = model.invoke([HumanMessage(content=ask)])
    n = state.get("rounds", 0) + 1
    print(f"  [generate {n}] {reply.content.strip()[:90]}...")

    return {                                   # only what changed
        "draft": reply.content,
        "rounds": n,
        "messages": [AIMessage(content=reply.content)],
    }


def evaluate(state: AgentState) -> dict:
    """Critique the draft. Reply ACCEPT if it is good enough."""
    model = chat_model(temperature=0.0)
    reply = model.invoke([HumanMessage(
        content="You review support replies. If this reply is clear, polite and "
                "actionable, reply with exactly ACCEPT. Otherwise give ONE "
                f"specific improvement.\n\nTicket:\n{state['ticket']}"
                f"\n\nReply:\n{state['draft']}")])

    verdict = reply.content.strip()
    print(f"  [evaluate]   {verdict[:90]}")
    return {"critique": "" if verdict.upper().startswith("ACCEPT") else verdict}


def should_continue(state: AgentState) -> str:
    """A PURE function of state. No model call, no side effects — so the whole
    control flow is testable without an API key."""
    if not state.get("critique"):
        return "end"                                    # reviewer said ACCEPT

    if state["rounds"] >= MAX_ROUNDS:                   # <-- TODO 2: the bound
        print(f"  [router]     hit MAX_ROUNDS={MAX_ROUNDS}, stopping")
        return "end"

    return "continue"


def build():
    builder = StateGraph(AgentState)

    builder.add_node("generate", generate)
    builder.add_node("evaluate", evaluate)              # <-- TODO 1a

    builder.add_edge(START, "generate")                 # entry point
    builder.add_edge("generate", "evaluate")            # <-- TODO 1b, always review

    # The one conditional edge. It hangs off EVALUATE, not generate: the
    # router reads `critique`, and evaluate is what writes it. Put the
    # condition on generate instead and it fires before anything has been
    # reviewed — the router sends the run straight to END and evaluate never
    # executes at all.
    builder.add_conditional_edges(
        "evaluate", should_continue,
        {"continue": "generate", "end": END},           # "continue" = redraft
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

    banner("RUNNING")
    final = app.invoke({
        "ticket": TICKET,
        "messages": [HumanMessage(content=TICKET)],
        "rounds": 0,
    })

    banner("STATE AFTER THE RUN")
    print(f"  rounds:   {final['rounds']}")
    kept = len(final["messages"])
    expected = 1 + final["rounds"]
    print(f"  messages: {kept}   (1 ticket + {final['rounds']} draft(s) "
          f"= {expected} expected)")
    if kept == expected:
        print("            ^ the reducer APPENDED. Delete add_messages in the")
        print("              state above and this drops to 1 — that was TODO 3.")

    banner("FINAL REPLY")
    print(final["draft"])


if __name__ == "__main__":
    main()
