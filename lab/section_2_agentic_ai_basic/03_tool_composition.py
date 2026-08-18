"""
§2 · 03 — Tool composition.  (slide 22)

    uv run section_2_agentic_ai_basic/03_tool_composition.py

One tool call is rarely enough. Slide 22: three shapes account for nearly all
agent behaviour, and none of them is something you declare. They emerge at run
time from what the model asks for.

    01 SEQUENTIAL   each output is the next input   latency = SUM of the steps
    02 PARALLEL     independent calls, one reply    latency = the SLOWEST call
    03 CONDITIONAL  the result picks the branch     one branch runs, others never

This file runs the same tool layer three ways and times each one, so the
latency claims above are numbers on your screen rather than claims on a slide.
"""

import time

import _bootstrap  # noqa: F401

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from _common import banner, chat_model, model_label

LATENCY = 0.8          # every tool below sleeps this long, to make timing visible


@tool
def get_order(order_id: str) -> str:
    """Look up a sales order by id. Returns the customer and the total in MYR."""
    time.sleep(LATENCY)
    rows = {"SO-1001": "Lim Trading, total 4820.00 MYR",
            "SO-1002": "Batu Supplies, total 15100.00 MYR"}
    print(f"      [tool] get_order({order_id!r})")
    return rows.get(order_id.upper(), f"No order {order_id}.")


@tool
def convert_currency(amount: float, to: str) -> str:
    """Convert an amount in MYR to USD, SGD or EUR."""
    time.sleep(LATENCY)
    rates = {"USD": 0.21, "SGD": 0.29, "EUR": 0.20}
    print(f"      [tool] convert_currency({amount}, {to!r})")
    if to.upper() not in rates:
        return f"No rate for {to}."
    return f"{amount} MYR = {amount * rates[to.upper()]:.2f} {to.upper()}"


@tool
def credit_check(customer: str) -> str:
    """Return the credit rating for a customer: good, watch, or hold."""
    time.sleep(LATENCY)
    print(f"      [tool] credit_check({customer!r})")
    return f"{customer}: watch (2 late payments in 90 days)"


TOOLS = [get_order, convert_currency, credit_check]
BY_NAME = {t.name: t for t in TOOLS}
SYSTEM = SystemMessage("Use the tools to answer. Be terse. One sentence at the end.")


# --------------------------------------------------------------------------
# One agent loop, reused by all three shapes. This is the same loop as file 00
# with a real model in place of `fake_model`.
# --------------------------------------------------------------------------

def run(question: str, parallel_tools: bool = False, max_steps: int = 6) -> dict:
    model = chat_model(temperature=0).bind_tools(TOOLS)
    messages = [SYSTEM, HumanMessage(question)]

    started, round_trips, tool_calls = time.time(), 0, 0

    for _ in range(max_steps):
        reply = model.invoke(messages)
        messages.append(reply)
        round_trips += 1

        if not reply.tool_calls:
            break

        calls = reply.tool_calls
        tool_calls += len(calls)
        print(f"    round {round_trips}: model asked for {len(calls)} call(s)")

        if parallel_tools and len(calls) > 1:
            # The calls in ONE reply have no dependency on each other — the
            # model issued them together precisely because none needs another's
            # output. So run them together. A thread pool is enough; these are
            # I/O waits, not CPU work.
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                results = list(pool.map(
                    lambda c: BY_NAME[c["name"]].invoke(c["args"]), calls))
        else:
            results = [BY_NAME[c["name"]].invoke(c["args"]) for c in calls]

        for call, result in zip(calls, results):
            messages.append(ToolMessage(result, tool_call_id=call["id"]))

    return {"answer": messages[-1].content.strip(),
            "seconds": time.time() - started,
            "round_trips": round_trips,
            "tool_calls": tool_calls}


# ==========================================================================
# 01 · SEQUENTIAL — the second call needs the first one's output.
# ==========================================================================

def sequential() -> None:
    banner("01 · SEQUENTIAL  ->  each output is the next input")
    print("  Q: What is order SO-1002 worth in USD?\n")

    out = run("What is order SO-1002 worth in USD?")

    print(f"\n  answer      : {out['answer']}")
    print(f"  round trips : {out['round_trips']}   tool calls: {out['tool_calls']}")
    print(f"  wall clock  : {out['seconds']:.1f}s   (latency = the SUM of the steps)")


# ==========================================================================
# 02 · PARALLEL — independent calls, issued in one reply.
# ==========================================================================

def parallel() -> None:
    banner("02 · PARALLEL  ||  independent calls issued together")
    q = ("For order SO-1001: give me the order details and the credit rating "
         "for Lim Trading. These are independent — request both at once.")

    print("  first, executing the calls one after another:")
    serial = run(q, parallel_tools=False)
    print(f"    -> {serial['seconds']:.1f}s for {serial['tool_calls']} tool call(s)")

    print("\n  now the same calls, executed together:")
    together = run(q, parallel_tools=True)
    print(f"    -> {together['seconds']:.1f}s for {together['tool_calls']} tool call(s)")

    print(f"\n  answer      : {together['answer']}")
    print(f"  serial      : {serial['seconds']:.1f}s")
    print(f"  parallel    : {together['seconds']:.1f}s   (latency = the SLOWEST call)")
    print(f"  round trips : {serial['round_trips']} -> {together['round_trips']}   (unchanged)")


# ==========================================================================
# 03 · CONDITIONAL — the result selects the branch. Your code, not the model.
# ==========================================================================

def conditional() -> None:
    banner("03 · CONDITIONAL  ?:  the result selects the branch")

    def escalate(customer: str) -> str:
        return f"Escalated {customer} to the credit team. Order held."

    def approve(customer: str) -> str:
        return f"{customer} approved automatically. Order released."

    customer = "Batu Supplies"
    rating = credit_check.invoke({"customer": customer})
    print(f"  observation : {rating}")

    # An ordinary `if`. No model call, no tokens, and you can unit-test it.
    if "hold" in rating or "watch" in rating:
        action = escalate(customer)
    else:
        action = approve(customer)

    print(f"  branch taken: {action}")
    print(f"  not run     : approve()   (one branch executes, the other never does)")


if __name__ == "__main__":
    print(f"model: {model_label()}")
    print(f"every tool sleeps {LATENCY}s, so the timings mean something\n")

    sequential()
    parallel()
    conditional()
