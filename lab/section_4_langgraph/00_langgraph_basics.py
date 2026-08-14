"""
§4 · 00 — LangGraph in 60 lines. No model, no API key, no cost.

    uv run section_4_langgraph/00_langgraph_basics.py

Four things to learn, and nothing else in the way:

    STATE    the shared object passed between steps
    NODES    plain functions: state in, partial update out
    EDGES    which node runs next
    RUNTIME  what compile() gives you

There is deliberately no LLM here. Every node is three lines of ordinary
Python, so the only thing you are watching is the graph.
"""

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


# ---------------------------------------------------------------- 1. STATE
# One TypedDict, shared by every node. The reducer on `log` appends;
# everything else is replaced key by key.

def append(old: list, new: list) -> list:
    """A reducer: (old, new) -> merged. This is all a reducer ever is."""
    return (old or []) + (new or [])


class State(TypedDict):
    order: dict                        # seeded by the caller
    total: float                       # written by price
    status: str                        # written by validate and confirm
    log: Annotated[list, append]       # appended by every node


# ---------------------------------------------------------------- 2. NODES
# state in, PARTIAL dict out. Return only what you changed.

def validate(state: State) -> dict:
    ok = state["order"].get("qty", 0) > 0
    print(f"  [validate] qty={state['order'].get('qty')} -> {'ok' if ok else 'REJECTED'}")
    return {"status": "valid" if ok else "rejected", "log": ["validate"]}


def price(state: State) -> dict:
    total = state["order"]["qty"] * state["order"]["unit_price"]
    print(f"  [price]    {state['order']['qty']} x {state['order']['unit_price']} = {total}")
    return {"total": total, "log": ["price"]}


def confirm(state: State) -> dict:
    print(f"  [confirm]  order confirmed at {state['total']}")
    return {"status": "confirmed", "log": ["confirm"]}


# ---------------------------------------------------------------- 3. EDGES
# A router is a PURE function of state returning a label. No model call.

def is_valid(state: State) -> str:
    return "ok" if state["status"] == "valid" else "reject"


builder = StateGraph(State)

builder.add_node("validate", validate)          # register each node by name
builder.add_node("price", price)
builder.add_node("confirm", confirm)

builder.add_edge(START, "validate")             # where to begin
builder.add_conditional_edges(                  # a fork: label -> next node
    "validate", is_valid,
    {"ok": "price", "reject": END},
)
builder.add_edge("price", "confirm")            # plain edge: always next
builder.add_edge("confirm", END)                # where to stop

# ---------------------------------------------------------------- 4. RUNTIME
graph = builder.compile()      # <- validates the wiring, returns a runnable


if __name__ == "__main__":
    print(graph.get_graph().draw_ascii())

    print("\n--- a good order ---")
    final = graph.invoke({"order": {"qty": 3, "unit_price": 12.50}, "log": []})
    print(f"  final state: status={final['status']} total={final['total']}")
    print(f"  log:         {final['log']}   <- the reducer appended")

    print("\n--- a bad order (qty 0) ---")
    final = graph.invoke({"order": {"qty": 0, "unit_price": 12.50}, "log": []})
    print(f"  final state: status={final['status']}")
    print(f"  log:         {final['log']}   <- price and confirm never ran")
