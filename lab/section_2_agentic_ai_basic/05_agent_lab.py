"""
§2 · 05 — LAB SOLUTION.  (slide 25)

    uv run section_2_agentic_ai_basic/05_agent_lab_solution.py

The three TODOs filled in. Read this AFTER you have made each one fail.

The loop is nine lines. Slide 25 is these nine lines written against the raw
Bedrock Converse API instead of LangChain — which is what §3 does. The shape
does not change; only the dict keys do.
"""

import json

import _bootstrap  # noqa: F401

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from _common import banner, chat_model

MAX_STEPS = 4

GOAL = ("Order A-1042 never arrived. Find out where it is, then say refund "
        "or wait, with the reason.")


@tool
def lookup_order(order_id: str) -> str:
    """Order details: which carrier has it, and its tracking number."""
    print(f"      [tool] lookup_order({order_id!r})")
    return json.dumps({"order_id": order_id, "carrier": "Fleetline",
                       "tracking": "FL77213", "shipped_days_ago": 9})


@tool
def track_shipment(carrier: str, tracking: str) -> str:
    """Where a parcel was last seen. Needs a carrier and a tracking number."""
    print(f"      [tool] track_shipment({carrier!r}, {tracking!r})")
    return json.dumps({"status": "in transit", "last_scan": "Ipoh depot",
                       "days_since_last_scan": 6})


TOOLS = [lookup_order, track_shipment]
BY_NAME = {t.name: t for t in TOOLS}


banner("RUN")

model = chat_model(temperature=0).bind_tools(TOOLS)
messages = [
    SystemMessage("You handle delivery complaints. Use the tools before you "
                  "judge. A parcel with no scan for over 5 days is lost. "
                  "Be terse."),
    HumanMessage(GOAL),
]

for step in range(1, MAX_STEPS + 1):
    print(f"\n  ---------- step {step} ----------")

    reply = model.invoke(messages)

    # TODO 1 — the tool request lives inside this message, so a ToolMessage
    # appended below has nothing to answer without it.
    messages.append(reply)

    # TODO 2 — prose and no tool call is the model saying it is finished.
    if not reply.tool_calls:
        print(f"      -> {' '.join(reply.content.split())}")
        break

    # TODO 3 — run each requested tool; tool_call_id is the join back to the
    # question it answers. This append is also how FL77213 reaches step 2.
    for call in reply.tool_calls:
        result = BY_NAME[call["name"]].invoke(call["args"])
        messages.append(ToolMessage(result, tool_call_id=call["id"]))
    print("      (result appended, deciding again)")

banner("DONE")
print(f"  {len(messages)} messages in the transcript.\n")