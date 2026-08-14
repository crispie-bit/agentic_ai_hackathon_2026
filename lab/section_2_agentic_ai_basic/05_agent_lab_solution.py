"""
§2 · 05 — LAB SOLUTION.  (slide 25)

    uv run section_2_agentic_ai_basic/05_agent_lab_solution.py

The four TODOs from 05_agent_lab.py, filled in. Read this AFTER you have made
each one fail.

The whole loop is 20 lines, and slide 25 is the same 20 lines written against
the raw Bedrock Converse API instead of LangChain — which is what §3 does. The
shape does not change; only the dict keys do.
"""

import json

import _bootstrap  # noqa: F401

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from _common import banner, chat_model, model_label, report_usage

MAX_STEPS = 6

INVENTORY = {"SKU-77": 12, "SKU-14": 240, "SKU-93": 0}
REORDER_LEVEL = 50


@tool
def check_stock(sku: str) -> str:
    """Units currently in stock for one SKU, e.g. SKU-77."""
    print(f"      [tool] check_stock({sku!r})")
    if sku.upper() not in INVENTORY:
        return f"No SKU {sku}. Known: {', '.join(INVENTORY)}."
    return json.dumps({"sku": sku.upper(), "units": INVENTORY[sku.upper()],
                       "reorder_level": REORDER_LEVEL})


@tool
def supplier_lead_time(sku: str) -> str:
    """Days between placing a restock order for a SKU and receiving it."""
    print(f"      [tool] supplier_lead_time({sku!r})")
    return json.dumps({"sku": sku.upper(), "lead_time_days": 14})


@tool
def raise_purchase_order(sku: str, qty: int) -> str:
    """Place a restock order. Use only after confirming stock is below the
    reorder level. This one changes the outside world."""
    print(f"      [tool] raise_purchase_order({sku!r}, {qty})")
    return f"PO-4471 raised: {qty} x {sku.upper()}"


TOOLS = [check_stock, supplier_lead_time, raise_purchase_order]
BY_NAME = {t.name: t for t in TOOLS}

SYSTEM = ("You manage inventory. Use the tools to check facts before you act. "
          "Restock to 3 months of cover when stock is below the reorder level. "
          "Finish with one sentence stating what you did and why.")


def run_agent(goal: str) -> dict:
    model = chat_model(temperature=0).bind_tools(TOOLS)

    messages = [SystemMessage(SYSTEM), HumanMessage(goal)]
    stopped_because = "step cap"

    for step in range(1, MAX_STEPS + 1):
        print(f"\n  ---------- step {step} ----------")

        reply = model.invoke(messages)
        report_usage(f"step {step}", reply.usage_metadata)

        messages.append(reply)                                    # TODO 1

        if not reply.tool_calls:                                  # TODO 2
            stopped_because = "model answered"
            break

        for call in reply.tool_calls:                             # TODO 3
            fn = BY_NAME.get(call["name"])
            if fn is None:
                # The model can name a tool that does not exist. Answer in a
                # form it can read and recover from, rather than raising.
                result = f"No tool named {call['name']}. Available: {', '.join(BY_NAME)}."
            else:
                try:
                    result = fn.invoke(call["args"])
                except Exception as exc:                # bad args reach here
                    result = f"{call['name']} failed: {exc}"
            messages.append(ToolMessage(result, tool_call_id=call["id"]))

    return {                                                      # TODO 4
        "answer": messages[-1].content if hasattr(messages[-1], "content") else "",
        "messages": messages,
        "stopped_because": stopped_because,
    }


if __name__ == "__main__":
    print(f"model: {model_label()}")

    goal = ("SKU-77 may be running low. Check it, and if it is below the "
            "reorder level, work out a sensible restock quantity and raise "
            "the purchase order.")

    banner("RUN")
    print(f"  goal: {goal}")
    out = run_agent(goal)

    banner("RESULT")
    print(f"  answer          : {out['answer'].strip()}")
    print(f"  stopped because : {out['stopped_because']}")
    print(f"  transcript      : {len(out['messages'])} messages")

    tool_calls = sum(len(getattr(m, "tool_calls", []) or []) for m in out["messages"])
    print(f"  tool calls      : {tool_calls}")
