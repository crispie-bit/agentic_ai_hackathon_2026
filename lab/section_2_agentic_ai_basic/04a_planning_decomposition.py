"""
§2 · 04a — Planning by decomposition, the smallest version.  (slide 18)

    uv run section_2_agentic_ai_basic/04a_decomposition.py

One model call produces the whole list of steps. Nothing has run yet at that
point. Everything after it is an ordinary for-loop over that list.

Watch the output in two halves:

    PLAN      printed once, before any tool runs
    EXECUTE   one step at a time, in the order the plan gave
"""

import json

import _bootstrap  # noqa: F401

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from _common import banner, chat_model

GOAL = ("Decide whether to restock SKU-77. Check the stock level and the "
        "supplier lead time, then say restock or hold, with the reason.")


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


class Plan(BaseModel):
    steps: list[str]


# --------------------------------------------------------------------------
# 1. PLAN.  One call. No tool runs here.
# --------------------------------------------------------------------------

banner("PLAN")

planner = chat_model(temperature=0).with_structured_output(Plan) # no tools
steps = planner.invoke([
    SystemMessage("Break the goal into 2-4 steps, each doable with one tool call."),
    HumanMessage(GOAL),
]).steps

for i, step in enumerate(steps, 1):
    print(f"  {i}. {step}")


# --------------------------------------------------------------------------
# 2. EXECUTE.  A for-loop over that list. The model never sees the full goal,
#    so each step is the only instruction it has.
# --------------------------------------------------------------------------

banner("EXECUTE")

actor = chat_model(temperature=0).bind_tools(TOOLS) ## TOOLS
messages = [SystemMessage(
    "Do only the step you are given. Call at most one tool. "
    "Do not anticipate later steps. "
    "Report the result in one short sentence. Do not show your working."
)]

for i, step in enumerate(steps, 1):
    print(f"\n  step {i}: {step}")
    messages.append(HumanMessage(step))

    reply = actor.invoke(messages) # one plan string becomes one user turn
    messages.append(reply) # actor sees tools, picks one

    # Keep going while the model is still asking for tools. A single
    # re-invoke is not enough: the reply after a tool result can itself be
    # another tool call, and printing it would show an empty line.
    while reply.tool_calls:
        for call in reply.tool_calls:
            result = BY_NAME[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(result, tool_call_id=call["id"]))
        reply = actor.invoke(messages)
        messages.append(reply)

    print(f"      -> {' '.join(reply.content.split())}")

banner("DONE")
print(f"  The plan was fixed after {len(steps)} steps were decided, and never changed.\n")