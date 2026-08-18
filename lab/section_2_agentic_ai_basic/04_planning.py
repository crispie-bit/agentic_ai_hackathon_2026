"""
§2 · 04 — Planning: three strategies.  (slides 17, 18, 19, 20)

    uv run section_2_agentic_ai_basic/04_planning.py

The same goal, the same tools, three strategies. What separates them is one
thing only: WHEN the order of work is fixed.

    01 DECOMPOSITION   before execution        a plan, then iterate over it
    02 REACTIVE        one step at a time      no plan; the last result decides
    03 HIERARCHICAL    at two levels           coarse plan, reactive beneath

None of the three is a component you install. A plan is text the model
produced, sitting in the messages like any other content — the loop that reads
it is what gives it effect. Delete the loop and the plan is a paragraph.

Provider comes from lab/.env.
"""

import json
import time

import _bootstrap  # noqa: F401

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from _common import banner, chat_model, model_label

GOAL = ("Decide whether to restock SKU-77. Check the stock level and the "
        "supplier lead time, then say restock or hold, with the reason.")

MAX_STEPS = 6


@tool
def check_stock(sku: str) -> str:
    """Units currently in stock for one SKU, and the reorder level."""
    print(f"      [tool] check_stock({sku!r})")
    return json.dumps({"sku": sku.upper(), "units": 12, "reorder_level": 50})


@tool
def supplier_lead_time(sku: str) -> str:
    """Days between placing an order for a SKU and receiving it."""
    print(f"      [tool] supplier_lead_time({sku!r})")
    return json.dumps({"sku": sku.upper(), "lead_time_days": 14, "supplier": "Ipoh Metals"})


@tool
def sales_velocity(sku: str) -> str:
    """Average units sold per day for a SKU over the last 30 days."""
    print(f"      [tool] sales_velocity({sku!r})")
    return json.dumps({"sku": sku.upper(), "units_per_day": 3.5})


TOOLS = [check_stock, supplier_lead_time, sales_velocity]
BY_NAME = {t.name: t for t in TOOLS}
SYSTEM = "You answer stock questions using the tools. Be terse."

# Used when a plan is driving. Without this the model reads ahead, finishes the
# whole goal on step 1, and every later step becomes a no-op — the plan gets
# printed and then ignored.
SYSTEM_STEPWISE = (
    "You answer stock questions using the tools. Be terse. "
    "Do only the step you are given. Do not anticipate later steps, and do "
    "not answer the overall question until you are asked to."
)


class Plan(BaseModel):
    """The planner's output shape. A plan is just text until something parses it."""

    steps: list[str] = Field(description="Ordered steps, each one tool call's worth of work.")


def plan_steps(instruction: str, goal: str) -> list[str]:
    """One model call that returns a list of steps.

    `with_structured_output` binds the Plan schema as a tool and parses the
    reply for you. Hand-parsing JSON out of prose works until the day the
    model wraps it in a code fence or adds a sentence in front of it — and
    then your planner is down, not your agent.
    """
    planner = chat_model(temperature=0).with_structured_output(Plan)
    return planner.invoke([SystemMessage(instruction), HumanMessage(goal)]).steps


def short(text: str, width: int = 240) -> str:
    """One line, trimmed. Long answers are not the lesson here."""
    text = " ".join((text or "").split())
    return text[:width] + ("..." if len(text) > width else "")


def tokens(messages) -> int:
    """Input tokens across every model reply in a run — the number that bills."""
    return sum((getattr(m, "usage_metadata", None) or {}).get("input_tokens", 0)
               for m in messages)


def react(messages: list, model, max_steps: int = MAX_STEPS) -> list:
    """The plain tool loop from file 00. All three strategies below use it;
    what differs is what they put in `messages` before calling it."""
    for _ in range(max_steps):
        reply = model.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return messages
        for call in reply.tool_calls:
            result = BY_NAME[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(result, tool_call_id=call["id"]))
    return messages


# ==========================================================================
# 01 · DECOMPOSITION — the whole sequence exists before the first action.
#      (slide 18)
# ==========================================================================

PLANNER = ("Expand the goal into steps, each one doable with a single tool "
           "call. Three or four steps at most.")


def decomposition() -> tuple[int, int, float]:
    banner("01 · DECOMPOSITION — order fixed BEFORE execution")
    started = time.time()

    acting_model = chat_model(temperature=0).bind_tools(TOOLS)

    # -- one model call produces the entire plan --------------------------
    steps = plan_steps(PLANNER, GOAL)

    print("  plan (one model call, before anything runs):")
    for i, s in enumerate(steps, 1):
        print(f"    {i}. {s}")

    # -- everything after this is iteration over a list -------------------
    # The actor never sees GOAL. Each step is its only instruction, which is
    # what makes the plan binding rather than decorative.
    messages = [SystemMessage(SYSTEM_STEPWISE)]
    for i, step in enumerate(steps, 1):
        print(f"\n  executing step {i}")
        messages.append(HumanMessage(f"Step {i}: {step}"))
        messages = react(messages, acting_model, max_steps=3)

    print(f"\n  answer: {short(messages[-1].content)}")

    return len(messages), tokens(messages), time.time() - started


# ==========================================================================
# 02 · REACTIVE — no plan. Each action is chosen from the state as it stands.
#      (slide 19)
# ==========================================================================

def reactive() -> tuple[int, int, float]:
    banner("02 · REACTIVE — order emerges ONE STEP AT A TIME")
    started = time.time()

    model = chat_model(temperature=0).bind_tools(TOOLS)
    messages = [SystemMessage(SYSTEM), HumanMessage(GOAL)]

    # No planning call at all. `react` IS the strategy.
    messages = react(messages, model)

    print(f"\n  no plan produced. steps taken: {(len(messages) - 2) // 2}")
    print(f"  answer: {short(messages[-1].content)}")

    return len(messages), tokens(messages), time.time() - started


# ==========================================================================
# 03 · HIERARCHICAL — a coarse plan above, a reactive loop beneath.
#      (slide 20)
# ==========================================================================

def contradicts(reply: AIMessage, remaining: list[str]) -> bool:
    """Cheap revision trigger: did the observation invalidate what is left?

    Deliberately a plain function, not a model call. Every check you can make
    in Python is a check that costs nothing and behaves the same every time."""
    return bool(remaining) and "cannot" in (reply.content or "").lower()


def hierarchical() -> tuple[int, int, float]:
    banner("03 · HIERARCHICAL — coarse plan ABOVE, reactive loop BENEATH")
    started = time.time()

    actor = chat_model(temperature=0).bind_tools(TOOLS)

    # OUTER: coarse, once, and it survives the whole run.
    plan = plan_steps("Give 2-3 coarse phases for this goal. Phases, not "
                      "individual tool calls.", GOAL)

    print("  plan (held for the whole run):")
    for i, s in enumerate(plan, 1):
        print(f"    {i}. {s}")

    messages = [SystemMessage(SYSTEM_STEPWISE)]

    i = 0
    while i < len(plan):
        step = plan[i]
        print(f"\n  phase {i + 1}: {step}")
        messages.append(HumanMessage(f"Work on this phase only: {step}"))

        # INNER: reactive, with its own step budget. max_steps resets at
        # every phase; the transcript does NOT — it carries through the whole
        # run. Two loops, two bounds. That is the entire idea.
        messages = react(messages, actor, max_steps=3)

        if contradicts(messages[-1], plan[i + 1:]):
            print("    observation contradicts the remaining plan -> revising")
            plan = plan[:i + 1] + plan_steps(
                "Revise the REMAINING phases in light of what just happened.",
                f"Goal: {GOAL}\nDone: {plan[:i + 1]}\nLatest: {messages[-1].content}")
        i += 1

    print(f"\n  answer: {short(messages[-1].content)}")

    return len(messages), tokens(messages), time.time() - started


if __name__ == "__main__":
    print(f"model: {model_label()}")
    print(f"goal : {GOAL}")

    rows = [("01 decomposition", *decomposition()),
            ("02 reactive", *reactive()),
            ("03 hierarchical", *hierarchical())]

    banner("THREE STRATEGIES, ONE GOAL")
    print(f"  {'strategy':20}{'messages':>10}{'input tok':>12}{'secs':>8}")
    print("  " + "-" * 50)
    for name, msgs, toks, secs in rows:
        print(f"  {name:20}{msgs:>10}{toks:>12}{secs:>8.1f}")