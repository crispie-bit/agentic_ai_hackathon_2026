"""
§2 · 00 — Anatomy of an agent.  (slides 15, 16)

    uv run section_2_agentic_ai_basic/00_anatomy_of_an_agent.py

    PLANNING    the model                <- the provider
    MEMORY      the messages list        <- your code
    TOOLS       plain functions          <- your code
    ACTIONS     calling them             <- your code
    THE LOOP    the for below            <- your code
"""

import _bootstrap  # noqa: F401

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from _common import banner, chat_model

MAX_STEPS = 5

GOAL = ("SKU-77 may be running low. Check it, and restock if it is below "
        "the reorder level of 50.")


# TOOLS
@tool
def check_stock(sku: str) -> str:
    """Units currently in stock for one SKU."""
    return f"{sku}: 12 units"


@tool
def raise_purchase_order(sku: str, qty: int) -> str:
    """Place a restock order."""
    return f"PO-4471 raised for {qty} x {sku}"

# STOCK = 12

# # TOOLS
# @tool
# def check_stock(sku: str) -> str:
#     """Units currently in stock for one SKU."""
#     return f"{sku}: {STOCK} units"


# @tool
# def raise_purchase_order(sku: str, qty: int) -> str:
#     """Place a restock order."""
#     global STOCK
#     STOCK = STOCK + qty
#     return f"PO raised for {qty} x {sku}"



TOOLS = [check_stock, raise_purchase_order]
BY_NAME = {t.name: t for t in TOOLS}


banner("ONE GOAL, RUN UNTIL DONE")
print(f"  goal: {GOAL}")

# PLANNING
model = chat_model(temperature=0).bind_tools(TOOLS)

# MEMORY
messages = [
    SystemMessage("You manage inventory. Check before you act. Be terse."),
    HumanMessage(GOAL),
]

# THE LOOP
for step in range(1, MAX_STEPS + 1):
    print(f"\n  ---------- step {step} ----------")
    print(f"  perception    {len(messages)} messages")

    reply = model.invoke(messages)        # reasoning and planning happen in here
    messages.append(reply)

    if not reply.tool_calls:
        print(f"  answer        {' '.join(reply.content.split())}")
        break

    # ACTIONS
    for call in reply.tool_calls:
        args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
        print(f"  action        {call['name']}({args})")

        result = BY_NAME[call["name"]].invoke(call["args"])

        messages.append(ToolMessage(result, tool_call_id=call["id"]))
        print(f"  observation   {result}")

banner("DONE")