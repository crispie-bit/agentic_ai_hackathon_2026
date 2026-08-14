"""
§4 · 02 — Tools can write to state.  (slide 47)

    uv run section_4_langgraph/02_tool_writes_state.py

The sales table lives in STATE. It is never put in the prompt.

The model is asked "which product made the most money?" — it cannot answer
that, because it has never seen the numbers. It calls a tool. The tool reaches
into state, does the arithmetic in Python, and writes the answers back into
state as structured fields.

    prompt   ->  "which product made the most money?"      (23 tokens)
    state    ->  the sales table                            (never sent)
    tool     ->  reads state, computes, writes state
    result   ->  state.best_seller, state.revenue, ...

Three annotations do the work:

    state:        Annotated[dict, InjectedState]        the live graph state
    tool_call_id: Annotated[str, InjectedToolCallId]    the id to answer with
    -> Command(update={...})                            a state update

Injected parameters are hidden from the model, so it cannot pass the wrong
table — it cannot pass one at all.

The rule: a tool returning Command(update=...) must write its own keys AND
emit a ToolMessage carrying the tool_call_id. Returning a plain string does
the second for you; returning a Command means you own it.
"""

from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

import _bootstrap  # noqa: F401

from _common import chat_model, model_label

# Pretend this is 100,000 rows out of a database. The point is that it does
# not matter how big it is, because it never reaches the model.
SALES = [
    {"product": "keyboard", "units": 120, "unit_price": 45.00},
    {"product": "monitor", "units": 30, "unit_price": 320.00},
    {"product": "mouse", "units": 260, "unit_price": 18.50},
    {"product": "laptop stand", "units": 75, "unit_price": 60.00},
    {"product": "webcam", "units": 40, "unit_price": 89.00},
]

QUESTION = "Which product made the most money last month, and how much?"


class SalesState(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: int
    sales_rows: list      # seeded at invoke; never sent to the model
    best_seller: str      # written by analyse_sales
    revenue: float        # written by analyse_sales
    total_revenue: float  # written by analyse_sales


@tool
def analyse_sales(
    metric: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Analyse the loaded sales table and record the results.

    Args:
        metric: either "revenue" (units x price) or "units".

    The sales table is already loaded — you do not pass it.
    """
    rows = state.get("sales_rows", [])
    if not rows:
        return Command(update={"messages": [ToolMessage(
            "ERROR: no sales data is loaded in state.", tool_call_id=tool_call_id)]})

    def value(row):
        return row["units"] * row["unit_price"] if metric == "revenue" else row["units"]

    best = max(rows, key=value)
    total = sum(r["units"] * r["unit_price"] for r in rows)

    print(f"  [tool] read {len(rows)} rows from STATE, ranked by {metric!r}")

    return Command(update={
        "best_seller": best["product"],
        "revenue": round(value(best), 2),
        "total_revenue": round(total, 2),
        "messages": [ToolMessage(
            f"Top product by {metric}: {best['product']} at "
            f"{value(best):,.2f}. Total across all products: {total:,.2f}.",
            tool_call_id=tool_call_id,
        )],
    })


def main() -> None:
    print(f"model: {model_label()}")

    agent = create_agent(
        model=chat_model(),
        tools=[analyse_sales],
        state_schema=SalesState,      # makes the custom keys reachable
        system_prompt=("You answer questions about sales using your tools. The "
                       "data is already loaded — call analyse_sales, then answer "
                       "in one sentence."),
    )

    print("\nIN")
    print(f"  prompt:            {QUESTION}")
    print(f"  state.sales_rows:  {len(SALES)} rows (never sent to the model)")

    final = agent.invoke({
        "messages": [HumanMessage(content=QUESTION)],
        "sales_rows": SALES,
    })

    print("\nOUT")
    print(f"  state.best_seller:   {final.get('best_seller')}")
    print(f"  state.revenue:       {final.get('revenue')}")
    print(f"  state.total_revenue: {final.get('total_revenue')}")
    print(f"  state.messages:      {len(final['messages'])}")
    print(f"\n  {final['messages'][-1].content.strip()}")


if __name__ == "__main__":
    main()
