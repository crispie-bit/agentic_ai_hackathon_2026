"""
§2 · 04b — Reactive planning, the smallest version.  (slide 19)

    uv run section_2_agentic_ai_basic/04b_reactive.py

Compare with 04a_decomposition.py. Same goal, same tools. The difference is
that nothing is planned. There is no PLAN section in the output, because no
plan is ever produced.

Each step is chosen from the transcript as it stands. The result of that step
goes back into the transcript, and the next step is chosen from the longer
version. The number of steps is not known until the run ends.
"""

import json

import _bootstrap  # noqa: F401

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from _common import banner, chat_model

GOAL = ("Decide whether to restock SKU-77. Check the stock level and the "
        "supplier lead time, then say restock or hold, with the reason.")

MAX_STEPS = 6          # the only bound on the loop


@tool
def check_stock(sku: str) -> str:
    """Units currently in stock for one SKU, and the reorder level."""
    print(f"      [tool] check_stock({sku!r})")
    return json.dumps({"sku": sku, "units": 12, "reorder_level": 50})


@tool
def supplier_lead_time(sku: str) -> str:
    """Days between placing an order for a SKU and receiving it."""
    print(f"      [tool] supplier_lead_time({sku!r})")
    return json.dumps({"sku": sku, "lead_time_days": 14})


TOOLS = [check_stock, supplier_lead_time]
BY_NAME = {t.name: t for t in TOOLS}


# --------------------------------------------------------------------------
# No planning call. The loop is the whole strategy.
#
# The model sees the full goal here, unlike decomposition. That is the point:
# it has to work out what to do next by itself, every time.
# --------------------------------------------------------------------------

banner("NO PLAN")

model = chat_model(temperature=0).bind_tools(TOOLS)
messages = [
    SystemMessage("You answer stock questions using the tools. Be terse."),
    HumanMessage(GOAL),
]

steps_taken = 0

for step in range(1, MAX_STEPS + 1):
    print(f"\n  step {step}")
    steps_taken = step

    reply = model.invoke(messages)
    messages.append(reply)

    if not reply.tool_calls:
        print(f"      -> {' '.join(reply.content.split())}")
        break                                  # no tool asked for: it is done

    for call in reply.tool_calls:
        result = BY_NAME[call["name"]].invoke(call["args"])
        messages.append(ToolMessage(result, tool_call_id=call["id"]))
    print("      (result appended, deciding again)")

banner("DONE")
print(f"  {steps_taken} steps. That number was not known before the run.")
print(f"  The loop ends when a reply asks for no tool, or at MAX_STEPS = {MAX_STEPS}.\n")