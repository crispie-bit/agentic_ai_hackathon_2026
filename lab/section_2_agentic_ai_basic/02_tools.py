"""
§2 · 02 — Tools: a schema plus a function.  (slide 23)

    uv run section_2_agentic_ai_basic/02_tools.py

Slide 23: a tool is two halves that never meet.

    DECLARATION   name, description, parameters, returns — JSON the model reads
    INVOCATION    an ordinary Python function that runs in YOUR process

The model never runs anything. What comes back from a call is a REQUEST to
invoke a function: a name and a JSON object. Whether that function exists,
whether those arguments are valid, and what happens when it raises are all
decided outside the model.

Provider comes from lab/.env, same as every other file here.
"""

import json

import _bootstrap  # noqa: F401

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from _common import banner, chat_model, model_label, report_usage
# ==========================================================================
# 1. THE DECLARATION — everything the model will ever know about this tool.
# ==========================================================================

# The @tool decorator builds the schema from three things and nothing else:
#   the function NAME          -> "name"
#   the DOCSTRING              -> "description"
#   the TYPE HINTS             -> "parameters"
# So the docstring is not a comment. It is the prompt that decides whether
# this tool gets picked, and vague docstrings are a top cause of an agent
# calling the wrong thing.

ORDERS = {
    "SO-1001": {"customer": "Lim Trading", "status": "shipped", "total": 4820.00},
    "SO-1002": {"customer": "Batu Supplies", "status": "awaiting payment", "total": 15100.00},
}


@tool
def get_order(order_id: str) -> str:
    """Look up one sales order by its id, e.g. SO-1001. Returns status and total."""
    print(f"      [tool ran] get_order({order_id!r})")
    order = ORDERS.get(order_id.upper())
    if order is None:
        return f"No order {order_id}. Known ids: {', '.join(ORDERS)}."
    return json.dumps(order)


@tool
def convert_currency(amount: float, to: str) -> str:
    """Convert an amount in Malaysian ringgit (MYR) to another currency code."""
    print(f"      [tool ran] convert_currency({amount!r}, {to!r})")
    rates = {"USD": 0.21, "SGD": 0.29, "EUR": 0.20}
    if to.upper() not in rates:
        return f"No rate for {to}. Supported: {', '.join(rates)}."
    return f"{amount} MYR = {amount * rates[to.upper()]:.2f} {to.upper()}"


TOOLS = [get_order, convert_currency]
BY_NAME = {t.name: t for t in TOOLS}


def show_the_schema() -> None:
    banner("1 · WHAT THE MODEL ACTUALLY READS")
    print(json.dumps(convert_to_openai_tool(get_order), indent=2))


# ==========================================================================
# 2. THE INVOCATION — the reply is a request, and you are the one who runs it.
# ==========================================================================

def one_round_by_hand() -> None:
    banner("2 · ONE ROUND, EVERY STEP VISIBLE")

    model = chat_model(temperature=0).bind_tools(TOOLS)   # <- schemas sent with the request

    messages = [
        SystemMessage("Use the tools to answer. Be terse."),
        HumanMessage("What is the status of order SO-1002?"),
    ]

    # -- call 1 -----------------------------------------------------------
    reply = model.invoke(messages)
    messages.append(reply)

    # `content` holds no answer text: on Groq it is '', on Bedrock it is the
    # raw content blocks. Either way the answer is not in there yet.
    print(f"  content     : {reply.text()!r}")
    print(f"  tool_calls  : {reply.tool_calls}")
    report_usage("call 1", reply.usage_metadata)

    # -- you execute ------------------------------------------------------
    for call in reply.tool_calls:
        result = BY_NAME[call["name"]].invoke(call["args"])   # YOUR process
        # The id matters: it pairs this result with that request. Lose it and
        # the provider rejects the next call.
        messages.append(ToolMessage(result, tool_call_id=call["id"]))
        print(f"\n  executed {call['name']} -> {result}")

    # -- call 2 -----------------------------------------------------------
    # Not necessarily the last one: the model may look at the result and ask
    # for something else. So this is a loop, and the exit condition is
    # "no tool was requested" — never "we have called twice".
    rounds = 1
    while rounds < 5:
        final = model.invoke(messages)
        messages.append(final)
        rounds += 1
        report_usage(f"call {rounds}", final.usage_metadata)
        if not final.tool_calls:
            break
        for call in final.tool_calls:
            result = BY_NAME[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(result, tool_call_id=call["id"]))
            print(f"  executed {call['name']} -> {result}")

    print(f"\n  final       : {final.content.strip()}")
    print(f"  round trips : {rounds}   transcript: {len(messages)} messages")


# ==========================================================================
# 3. FAILURE — the half nobody demos.
# ==========================================================================

def when_it_goes_wrong() -> None:
    banner("3 · BAD ARGUMENTS ARE NORMAL, NOT EXCEPTIONAL")

    model = chat_model(temperature=0).bind_tools(TOOLS)
    messages = [
        SystemMessage("Use the tools to answer. Be terse."),
        HumanMessage("What is the status of order SO-9999?"),
    ]

    for _ in range(4):
        reply = model.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            print(f"  model asked : {call['name']}({call['args']})")
            # get_order RETURNS a sentence instead of raising. That is
            # deliberate: a tool error the model can READ is a tool error the
            # model can recover from. An exception that escapes just kills
            # the run.
            result = BY_NAME[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(result, tool_call_id=call["id"]))

    print(f"  recovered   : {reply.content.strip()}")


if __name__ == "__main__":
    print(f"model: {model_label()}")

    show_the_schema()
    one_round_by_hand()
    when_it_goes_wrong()
