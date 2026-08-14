"""
§2 · 00 — Anatomy of an agent.  (slides 15, 16, 17)

    uv run section_2_agentic_ai_basic/00_anatomy_of_an_agent.py

No API key. No network. No cost. There is deliberately no real model in this
file — the "model" below is twelve lines of scripted Python — because the only
thing worth watching first is the SHAPE of the loop.

Slide 15 named four components; slide 17 said where each one lives:

    PLANNING    model reasoning, prompted in the system message   <- the model
    MEMORY      the messages list                                 <- your code
    TOOLS       ordinary Python functions                         <- your code
    ACTIONS     calling those functions                           <- your code
    THE LOOP    the while/for that runs until done                <- your code

Four of the five are yours. That is the sentence the whole section rests on:
a chatbot runs the cycle once, an agent runs it until the goal is met.

Slide 16 split one turn into five stages. Watch which ones print from inside
`fake_model()` and which ones print from the loop.
"""

import json

# --------------------------------------------------------------------------
# TOOLS — component 3. A tool is a plain function. Nothing more, yet.
# --------------------------------------------------------------------------

STOCK = {"SKU-77": 12, "SKU-14": 240}
REORDER_LEVEL = 50


def check_stock(sku: str) -> str:
    """Units currently in stock for one SKU."""
    return f"{sku}: {STOCK.get(sku.upper(), 0)} units"


def raise_purchase_order(sku: str, qty: int) -> str:
    """Place a restock order. This one changes the outside world."""
    return f"PO-4471 raised for {qty} x {sku}"


# The registry is the whole "tool layer": a name -> function dict.
TOOLS = {"check_stock": check_stock, "raise_purchase_order": raise_purchase_order}


# --------------------------------------------------------------------------
# THE MODEL — component 1, and the ONLY part a provider runs.
#
# A real model reads `messages` and returns either text or a tool request.
# This stand-in does exactly that, from a script, so the loop below is
# byte-for-byte the loop you will write against a real model in file 05.
# --------------------------------------------------------------------------

def fake_model(messages: list[dict]) -> dict:
    """Return an assistant message: either {"text": ...} or {"tool": ..., "args": ...}."""
    observations = [m for m in messages if m["role"] == "tool"]

    if not observations:                              # nothing seen yet
        return {"tool": "check_stock", "args": {"sku": "SKU-77"},
                "why": "I need the stock level before I can judge it."}

    if len(observations) == 1:                        # saw the stock level
        return {"tool": "raise_purchase_order", "args": {"sku": "SKU-77", "qty": 100},
                "why": "12 is below the reorder level of 50, so I restock."}

    return {"text": "SKU-77 was at 12 units, below the reorder level of 50. "
                    "I raised PO-4471 for 100 units.",
            "why": "Both facts are in the transcript. Nothing left to do."}


# --------------------------------------------------------------------------
# THE LOOP — component 5. Perception, action and observation are all here.
# --------------------------------------------------------------------------

MAX_STEPS = 5          # the ONLY thing standing between you and an infinite bill


def run_agent(goal: str) -> str:
    # MEMORY — component 2. The transcript IS the memory. There is no other.
    messages: list[dict] = [{"role": "user", "content": goal}]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n  ---------- step {step} ----------")

        # 01 PERCEPTION — what the model gets to see is this list, and only this.
        print(f"  01 PERCEPTION   {len(messages)} messages in the transcript")

        # 02 REASONING + 03 PLANNING happen INSIDE the call. You never see them
        # separately; one request goes out, one message comes back.
        reply = fake_model(messages)
        print(f"  02 REASONING    (inside the call)")
        print(f"  03 PLANNING     {reply['why']}")

        if "text" in reply:                       # no tool requested -> done
            messages.append({"role": "assistant", "content": reply["text"]})
            print(f"  -- no tool requested, the loop ends --")
            return reply["text"]

        # 04 ACTION — your code. The model asked; it did not execute.
        name, args = reply["tool"], reply["args"]
        messages.append({"role": "assistant", "content": json.dumps(reply)})
        print(f"  04 ACTION       {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")

        result = TOOLS[name](**args)

        # 05 OBSERVATION — also your code: append the result, go round again.
        messages.append({"role": "tool", "content": result})
        print(f"  05 OBSERVATION  {result}")

    return "step cap reached without an answer"


if __name__ == "__main__":
    goal = ("SKU-77 may be running low. Check it, and restock if it is below "
            "the reorder level of 50.")

    print("=" * 72)
    print("ONE GOAL, RUN UNTIL DONE")
    print("=" * 72)
    print(f"  goal: {goal}")

    answer = run_agent(goal)

    print("\n" + "=" * 72)
    print("ANSWER")
    print("=" * 72)
    print(f"  {answer}")
