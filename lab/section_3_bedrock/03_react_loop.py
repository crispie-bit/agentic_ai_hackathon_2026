"""
§3 · 03 — The ReAct loop against raw Bedrock.

    uv run section_3_bedrock/03_react_loop.py

The same loop as §2, with no LangChain in the way. Everything is a dict you
build yourself, which makes one thing visible that LangChain hides:

    A TOOL RESULT IS SENT BACK AS A "user" MESSAGE.

Not an assistant turn, not a special role. In the Anthropic protocol the tool
output arrives from the user side, and `ToolMessage` in §2 is a wrapper over
exactly this shape.

Two other differences from §2:

    reply["content"]      a LIST of blocks, not a string. One reply can hold
                          a text block and several tool_use blocks together.
    reply["stop_reason"]  "tool_use" means it wants a tool; "end_turn" means
                          it is finished. This drives the loop, not the text.

Requires AWS credentials and model access. Run
`uv run section_1_foundation/00_check_bedrock.py` first.
"""

import json

import _bootstrap  # noqa: F401

from _common import MODEL_ID, banner, bedrock_runtime, report_usage

MAX_STEPS = 6
QUESTION = "Should we restock SKU-77? Check the stock and the lead time first."


# --------------------------------------------------------------------------
# The tools. Two halves: the function, and the schema the model reads.
# --------------------------------------------------------------------------

def check_stock(sku: str) -> dict:
    print(f"      [tool] check_stock({sku!r})")
    return {"sku": sku, "units": 12, "reorder_level": 50}


def supplier_lead_time(sku: str) -> dict:
    print(f"      [tool] supplier_lead_time({sku!r})")
    return {"sku": sku, "lead_time_days": 14}


REGISTRY = {"check_stock": check_stock, "supplier_lead_time": supplier_lead_time}

# Written by hand here. In §2 the @tool decorator generated this from the
# function name, the docstring and the type hints.
TOOL_SPECS = [
    {
        "name": "check_stock",
        "description": "Units currently in stock for one SKU, and its reorder level.",
        "input_schema": {
            "type": "object",
            "properties": {"sku": {"type": "string", "description": "e.g. SKU-77"}},
            "required": ["sku"],
        },
    },
    {
        "name": "supplier_lead_time",
        "description": "Days between placing an order for a SKU and receiving it.",
        "input_schema": {
            "type": "object",
            "properties": {"sku": {"type": "string", "description": "e.g. SKU-77"}},
            "required": ["sku"],
        },
    },
]


def call_model(client, messages: list) -> dict:
    """One InvokeModel call. Returns the parsed envelope."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0,
        "tools": TOOL_SPECS,
        "messages": messages,
    })
    response = client.invoke_model(modelId=MODEL_ID, body=body)
    return json.loads(response["body"].read())      # the stream reads once only


def run_tool(block: dict) -> dict:
    """Execute one tool_use block and build the tool_result that answers it.

    The id must match, and a result is required for every tool_use in the
    reply. A failure is reported here as content, not raised: the model reads
    it and can correct the call on the next step.
    """
    fn = REGISTRY.get(block["name"])
    if fn is None:
        output = {"error": f"no tool named {block['name']}. "
                           f"Available: {', '.join(REGISTRY)}."}
    else:
        try:
            output = fn(**block["input"])
        except Exception as exc:                     # noqa: BLE001
            output = {"error": f"{block['name']} failed: {exc}"}

    return {
        "type": "tool_result",
        "tool_use_id": block["id"],
        "content": [{"type": "text", "text": json.dumps(output)}],
    }


def main() -> None:
    client = bedrock_runtime()
    banner(f"ReAct against {MODEL_ID}")
    print(f"  question: {QUESTION}\n")

    messages = [{"role": "user", "content": QUESTION}]

    for step in range(1, MAX_STEPS + 1):
        print(f"  step {step}")
        reply = call_model(client, messages)
        report_usage(f"step {step}", reply.get("usage", {}))

        # 1. The reply goes back in, whole. content is a list of blocks.
        messages.append({"role": "assistant", "content": reply["content"]})

        # 2. stop_reason decides the loop, not the text.
        if reply["stop_reason"] != "tool_use":
            answer = "".join(b.get("text", "") for b in reply["content"]
                             if b["type"] == "text")
            print(f"\n  answer: {' '.join(answer.split())}")
            print(f"  finished after {step} step(s), stop_reason={reply['stop_reason']}")
            return

        # 3. Every tool_use block needs a tool_result, and they travel
        #    together in ONE user message.
        results = [run_tool(b) for b in reply["content"] if b["type"] == "tool_use"]
        messages.append({"role": "user", "content": results})

    print(f"\n  stopped at MAX_STEPS = {MAX_STEPS} without an answer.")
    print("  The cap is the only thing that guarantees this loop terminates.")


if __name__ == "__main__":
    main()