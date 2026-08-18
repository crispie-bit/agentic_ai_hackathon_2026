"""
§2 · 05 — THE LAB.  (slides 24, 25)

    uv run section_2_agentic_ai_basic/05_agent_lab.py


WHAT YOU ARE BUILDING

    A customer's parcel never arrived. You are writing the loop that lets a
    model sort it out for itself: look the order up, track the shipment, then
    say refund or wait. Everything except the loop is already written for you
    — two tools, a system prompt, a goal.


WHAT IS WRONG WITH IT RIGHT NOW

    Run it before you change anything. It does not crash. It is just useless.

    Every step prints `(sending 2 messages)`, and every step asks for the same
    tool. The model requests `lookup_order`, you throw the answer away, and the
    next step asks the identical question — six times, until MAX_STEPS stops
    it. Nothing accumulates, so nothing is learned.

    That is the whole lesson in one sentence: an agent is a loop over a list of
    messages, and if nothing goes into the list, there is no agent.


WHAT TO DO

    Four TODOs, one line of code each. Every line is written out for you in the
    comment above its TODO — this is not a guessing game. Type it, re-run, and
    watch the output change.

        TODO 1   put the model's reply into the list
        TODO 2   stop when the model says it is done
        TODO 3   run the tool and put the result into the list
        TODO 4   say WHY the loop ended

    Do them in order, and re-run after each one. Skipping ahead works, but you
    lose the part that teaches you something: seeing what each line changes.


AFTER TODO 1, IT STILL WILL NOT WORK. THAT IS EXPECTED.

    You will see `(sending N messages)` finally start climbing — 2, 3, 4, 5,
    6, 7 — and the run will still be useless:

        model asked for: lookup_order      ... on every single step, again

    Stop and work out why before you carry on. The model can now see that it
    asked for a tool. It still has never seen an ANSWER, because nothing runs
    the tool yet. So it asks again. Growing the transcript is necessary and it
    is not sufficient — TODO 3 is the other half.

    (If you are running on Bedrock rather than Groq you get a hard error here
    instead — `tool_use ids were found without tool_result blocks`. Same cause,
    louder. Bedrock refuses a transcript where a request has no answer; Groq
    shrugs and lets you keep going.)


HOW YOU KNOW YOU ARE FINISHED

    Two tool calls, a verdict, and it stops on its own at step 3 — not at 6:

        ---------- step 1 ----------
            [tool] lookup_order('A-1042')
        ---------- step 2 ----------
            [tool] track_shipment('Fleetline', 'FL77213')
        ---------- step 3 ----------
            -> REFUND. No scan for 6 days, past the 5-day threshold.

        Ended: the model stopped asking for tools. That is an answer.


STUCK, OR WANT TO COMPARE

    uv run section_2_agentic_ai_basic/05_agent_lab_solution.py
"""

import json

import _bootstrap  # noqa: F401

# ToolMessage looks unused because it is — until you write TODO 3. It is
# imported for you so this lab is about the loop, not about an ImportError.
from langchain_core.messages import (  # noqa: F401
    HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool

from _common import banner, chat_model

MAX_STEPS = 6

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
BY_NAME = {t.name: t for t in TOOLS}          # {"lookup_order": <tool>, ...}


banner("RUN")

model = chat_model(temperature=0).bind_tools(TOOLS)
messages = [
    SystemMessage("You handle delivery complaints. Use the tools before you "
                  "judge. A parcel with no scan for over 5 days is lost. "
                  "Be terse."),
    HumanMessage(GOAL),
]

# TODO 4, part 1 of 2 — uncomment this line. It starts out False, and TODO 4
# part 2 sets it to True. See the bottom of the file.
#
#     finished = False


for step in range(1, MAX_STEPS + 1):
    print(f"\n  ---------- step {step} ----------")
    print(f"      (sending {len(messages)} messages)")

    reply = model.invoke(messages)

    # Diagnostic, not a TODO. Leave it — it is how you see the loop working.
    asked = [c["name"] for c in reply.tool_calls] or ["nothing (just text)"]
    print(f"      model asked for: {', '.join(asked)}")

    # ======================================================================
    # TODO 1 — add the model's reply to the transcript. Type this line:
    #
    #     messages.append(reply)
    #
    # WHY: `reply` is where the model's tool REQUEST lives. If it is not in
    # the list, the result you append in TODO 3 answers a question nobody
    # can see.
    #
    # AFTER: the `(sending N messages)` number starts growing instead of
    # sitting at 2.
    # ======================================================================

    # ======================================================================
    # TODO 2 — leave the loop when the model is done. Type these three lines,
    # indented to here:
    #
    #     if not reply.tool_calls:
    #         print(f"      -> {' '.join(reply.content.split())}")
    #         break
    #
    # WHY: text and no tool call is the model saying it has finished. This is
    # the NORMAL way out. Without it the only exit is MAX_STEPS, and that is
    # not an exit, it is a timeout.
    #
    # AFTER: once TODO 3 is done too, the run stops on its own and prints the
    # answer, instead of always using all 6 steps.
    # ======================================================================

    # ======================================================================
    # TODO 3 — run the tool the model asked for, and add the result. Type:
    #
    #     for call in reply.tool_calls:
    #         result = BY_NAME[call["name"]].invoke(call["args"])
    #         messages.append(ToolMessage(result, tool_call_id=call["id"]))
    #
    # WHY: `call["id"]` is what pairs the result with the question it
    # answers. Get it wrong and the model sees an orphan. This append is also
    # the ONLY way tracking number FL77213 ever reaches step 2.
    #
    # AFTER: the [tool] lines appear, and step 2 asks for a DIFFERENT tool
    # than step 1 — because it finally learned something.
    # ======================================================================

    # Delete this line once TODO 3 works.
    print("      (nothing appended — step 6 will send what step 1 sent)")


banner("DONE")

# ==========================================================================
# TODO 4, part 2 of 2 — say why the loop ended. Add this line just above your
# `break` in TODO 2:
#
#     finished = True
#
# then uncomment this block:
#
#     if finished:
#         print("  Ended: the model stopped asking for tools. "
#               "That is an answer.")
#     else:
#         print(f"  Ended: MAX_STEPS ({MAX_STEPS}) stopped it, mid-thought.")
#
# WHY: both endings leave text in `messages`, and only one of them is an
# answer. Code that cannot tell them apart will hand a half-finished thought
# to a user and call it a result.
# ==========================================================================

print(f"  {len(messages)} messages in the transcript.\n")
