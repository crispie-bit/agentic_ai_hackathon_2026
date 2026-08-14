"""
§4 · 01 — LAB: build the graph.  (slides 44-46)   ** YOU WRITE THIS ONE **

    uv run section_4_langgraph/01_graph_lab.py

Section 3 ended with an agent that was a `for` loop. Nothing about the model
call changes here. LangGraph just gives that loop a runtime: named steps,
explicit routing, and one state object passed between them.

    START -> generate -> should_continue -> evaluate -> (back to generate)
                              |
                              +-> END

Four things, and that is the whole library:

    State     the shared object every node reads and writes
    Nodes     one unit of work: a model call, a tool, a check
    Edges     which node runs next
    Runtime   checkpointing, streaming, retries, replay (compile() gives you this)

# ======================================================================
# TODO(student) 1 — the graph is missing its second node. In build(),
#   register `evaluate` with add_node, then add the edge that sends it
#   back to `generate`. Two lines. Forgetting the EDGE is the usual
#   mistake: the node exists, and simply never runs.
#
# TODO(student) 2 — should_continue() currently never stops. Add the
#   bound: if state["rounds"] >= MAX_ROUNDS, return "end". Run it before
#   you fix it if you like — the recursion limit will stop it for you,
#   which is not a plan.
#
# TODO(student) 3 — delete `add_messages` from the messages annotation so
#   it reads `messages: list`. Run again and watch the history get
#   clobbered: each node's return REPLACES messages instead of appending.
#   That is what a reducer is for. Put it back afterwards.
# ======================================================================

Worked answers: 01_graph_lab_solution.py (runnable — diff it against yours)
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


# --------------------------------------------------------------------------
# STATE — a TypedDict every node reads and writes.  (slide 45)
#
# The comment on each field says WHICH NODE fills it in. That convention costs
# nothing and is worth more than a diagram: state is the contract between
# nodes, and this is where the contract is written down.
# --------------------------------------------------------------------------

class AgentState(TypedDict):
    # Annotated with a reducer, so nodes APPEND to history instead of
    # replacing it. Remove the annotation and see TODO 3.
    messages: Annotated[list, add_messages]
    ticket: str        # seeded by the caller
    draft: str         # written by generate
    critique: str      # written by evaluate
    rounds: int        # incremented by generate; this is what bounds the loop


# --------------------------------------------------------------------------
# NODES — state in, PARTIAL update out.
#
# A node returns only the keys it changed. LangGraph merges that dict into the
# shared state before the next node runs. Returning the whole state works too,
# and is how people accidentally overwrite each other's fields.
# --------------------------------------------------------------------------

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

    return {                                   # <- only what changed
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
    # An empty critique is the signal that we are done.
    return {"critique": "" if verdict.upper().startswith("ACCEPT") else verdict}


# --------------------------------------------------------------------------
# ROUTER — a PURE function of state that returns a label.  (slide 46)
#
# No model call, no side effects. That is what makes your control flow
# deterministic and testable without an API key.
# --------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    if not state.get("critique"):
        return "end"                     # the reviewer said ACCEPT

    # TODO 2: the bound. Without it the back-edge runs forever.
    #     if state["rounds"] >= MAX_ROUNDS:
    #         print(f"  [router]     hit MAX_ROUNDS={MAX_ROUNDS}, stopping")
    #         return "end"

    return "continue"


# --------------------------------------------------------------------------
# ASSEMBLY — four calls and a compile.  (slide 46)
# --------------------------------------------------------------------------

def build():
    builder = StateGraph(AgentState)

    builder.add_node("generate", generate)
    # TODO 1a: builder.add_node("evaluate", evaluate)

    builder.add_edge(START, "generate")              # entry point
    # TODO 1b: builder.add_edge("generate", "evaluate")   # always review

    # The one conditional edge. Note it hangs off EVALUATE, not generate: the
    # router reads `critique`, and evaluate is what writes it. Hang it off
    # generate and it fires before anything has been reviewed — the run goes
    # straight to END and evaluate never executes at all.
    builder.add_conditional_edges(
        "evaluate", should_continue,
        {"continue": "generate", "end": END},        # "continue" = redraft
    )

    # compile() is what turns this into a runnable — and what gives you
    # streaming, checkpointing and replay for free.
    return builder.compile()


def main() -> None:
    print(f"model: {model_label()}")

    try:
        app = build()
    except ValueError as exc:
        # LangGraph validates the wiring at compile() time, before a single
        # model call. Read the message — it names the node it could not find.
        raise SystemExit(
            f"\nThe graph did not compile:\n  {exc}\n\n"
            "  That is TODO 1. The graph refers to a node called 'evaluate',\n"
            "  but nothing registered it. Add both lines in build():\n"
            "      builder.add_node(\"evaluate\", evaluate)\n"
            "      builder.add_edge(\"generate\", \"evaluate\")\n\n"
            "  Worth noticing: this failed at compile time, not at run time.\n"
            "  A graph checks its own wiring before it costs you a token."
        ) from exc

    banner("THE GRAPH")
    try:
        print(app.get_graph().draw_ascii())
    except Exception:
        print("  (uv add grandalf for the ASCII diagram)")

    banner("RUNNING")
    final = app.invoke({
        "ticket": TICKET,
        # Seed the history with the user's turn too, or a one-round run leaves
        # exactly one message and TODO 3 shows no difference at all.
        "messages": [HumanMessage(content=TICKET)],
        "rounds": 0,
    })

    banner("STATE AFTER THE RUN")
    print(f"  rounds:   {final['rounds']}")
    kept = len(final["messages"])
    expected = 1 + final["rounds"]
    print(f"  messages: {kept}   (1 ticket + {final['rounds']} draft(s) "
          f"= {expected} expected)")
    if kept < expected:
        print("            ^ history was CLOBBERED — that is TODO 3, and it is "
              "what a reducer prevents")

    banner("FINAL REPLY")
    print(final["draft"])


if __name__ == "__main__":
    main()
