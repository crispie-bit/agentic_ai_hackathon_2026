"""
§3 · 01 — InvokeModel: the raw call.

    uv run section_3_bedrock/01_invoke_model.py

The lowest level there is: boto3 against bedrock-runtime, no framework. Every
abstraction in the rest of this course reduces to what is in this file.

Nothing runs on your laptop. You sign an HTTPS request with your AWS
credentials, it goes to a managed endpoint, and you pay per token.
"""

import json

import _bootstrap  # noqa: F401  — must come first; puts lab/ on sys.path

from _common import MODEL_ID, REGION, banner, bedrock_runtime, report_usage

MAX_TOKENS = 300
QUESTION = "In one sentence: what is Amazon Bedrock?"


def main() -> None:
    banner("InvokeModel — the four keys you must send")
    print(f"  model:  {MODEL_ID}")
    print(f"  region: {REGION}")

    client = bedrock_runtime()

    body = json.dumps({
        # 1. A fixed Bedrock literal. NOT your model's version, and not
        #    optional. Every Anthropic-on-Bedrock request carries it.
        "anthropic_version": "bedrock-2023-05-31",

        # 2. The conversation. Roles must alternate user/assistant.
        "messages": [{"role": "user", "content": QUESTION}],

        # 3. Required on Bedrock, unlike Anthropic's first-party API.
        #    Caps OUTPUT only — it is not a budget for the whole call.
        "max_tokens": MAX_TOKENS,

        # 4. Not required, but always set it. The default is not 0.
        "temperature": 0,
    })

    print("\n  what goes on the wire:")
    print("   ", body[:200], "...")

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=body,                       # a JSON *string*, not a dict
        contentType="application/json",
        accept="application/json",
    )

    # The response body is a STREAMING object. .read() it before json.loads,
    # or you get "Object of type StreamingBody is not JSON serializable".
    envelope = json.loads(response["body"].read())

    banner("the response")
    print(f"  keys:        {list(envelope)}")
    # `content` is a LIST of typed blocks — never a plain string. That is also
    # how images and tool calls arrive, which is why it looks over-built for a
    # one-line answer.
    print(f"  content:     {envelope['content']}")
    print(f"  text:        {envelope['content'][0]['text'].strip()}")
    print(f"  stop_reason: {envelope['stop_reason']}")
    report_usage("invoke_model", envelope.get("usage", {}))


if __name__ == "__main__":
    main()
