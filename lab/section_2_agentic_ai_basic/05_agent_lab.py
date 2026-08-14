"""
§2 · 05 — LAB: build the agent.  (slide 25)

    uv run section_2_agentic_ai_basic/05_agent_lab.py

Everything up to here you read. This one you write.

Four TODOs, one line or so each, in `run_agent` below. The file runs before
you touch it — it just gets the answer wrong in an instructive way. Re-run
between each TODO and read what changed.

    TODO 1   append the reply, so the model can see its own request
    TODO 2   stop when no tool was requested
    TODO 3   execute the tool and append the result with its id
    TODO 4   report WHY the loop ended

Stuck? `05_agent_lab_solution.py` next to this file is the finished version.
Read it after you have made each TODO fail at least once.

Provider comes from lab/.env — this is provider-agnostic, and §3 rewrites the
same loop against the raw Bedrock Converse API.
"""

import json

import _bootstrap  # noqa: F401

# ToolMessage looks unused until you fill in TODO 3. It is imported for you.
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage  # noqa: F401
from langchain_core.tools import tool

from _common import banner, chat_model, model_label, report_usage

MAX_STEPS = 6           # the cap. Slide 25: it is the safety net, not a detail.


# --------------------------------------------------------------------------
# The tool layer. Already written — do not change it yet.
# --------------------------------------------------------------------------

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


# ==========================================================================
# THE LOOP — this is the part you write.
# ==========================================================================

def run_agent(goal: str) -> dict:
    model = chat_model(temperature=0).bind_tools(TOOLS)

    # MEMORY: the transcript. Everything the agent knows lives in this list.
    messages = [SystemMessage(SYSTEM), HumanMessage(goal)]
    stopped_because = "step cap"

    for step in range(1, MAX_STEPS + 1):
        print(f"\n  ---------- step {step} ----------")

        reply = model.invoke(messages)               # PERCEPTION + REASONING + PLANNING
        report_usage(f"step {step}", reply.usage_metadata)

        # ------------------------------------------------------------------
        # TODO 1 — append `reply` to `messages`.
        #
        # Without this, the model never sees its own tool request, so on the
        # next turn it asks for the same thing again. Run the file as-is
        # first and watch it call check_stock over and over until the cap.
        #
        #   messages.append(reply)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # TODO 2 — if the model requested no tools, it has answered. Stop.
        #
        # `reply.tool_calls` is a list, empty when the model replied with
        # text. This is the NORMAL exit; the step cap is the abnormal one.
        #
        #   if not reply.tool_calls:
        #       stopped_because = "model answered"
        #       break
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # TODO 3 — ACTION and OBSERVATION. For each requested call:
        #   a. look the function up in BY_NAME by call["name"]
        #   b. run it with call["args"]
        #   c. append a ToolMessage with the result AND call["id"]
        #
        # The id is not decoration. It pairs the result with the request, and
        # the provider rejects the next call without it. Try dropping it once.
        #
        #   for call in reply.tool_calls:
        #       result = BY_NAME[call["name"]].invoke(call["args"])
        #       messages.append(ToolMessage(result, tool_call_id=call["id"]))
        # ------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # TODO 4 — return `stopped_because` alongside the answer.
    #
    # An agent that hit the cap and one that finished look identical from
    # the outside: both return text. Only one of them is trustworthy.
    # ----------------------------------------------------------------------
    return {
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
    print(f"  answer          : {out['answer'].strip() or '(none — the loop never got one)'}")
    print(f"  stopped because : {out['stopped_because']}")
    print(f"  transcript      : {len(out['messages'])} messages")

    tool_calls = sum(len(getattr(m, "tool_calls", []) or []) for m in out["messages"])
    print(f"  tool calls      : {tool_calls}")
