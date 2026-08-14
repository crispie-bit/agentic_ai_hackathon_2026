"""
Day 2 pre-work. Run this BEFORE Session 2 starts, not during it.

    uv sync --extra aws
    export LLM_PROVIDER=bedrock
    uv run 00_check_bedrock.py

Session 1 needs none of this — it runs on Groq (00_check_groq.py). This script
is for the Bedrock half.

Verifies, in order, the four things that actually go wrong:
  1. boto3 can find credentials
  2. a region is set
  3. this account can invoke the lab's model on Bedrock
  4. langchain-aws is installed and can reach the same model

If it prints OK, every other lab will run. If it fails, raise your hand — the
message tells the instructor exactly which of the four broke.
"""

import json
import sys

import _bootstrap  # noqa: F401

from _common import MODEL_ID, PROVIDER, REGION, banner, bedrock_runtime, report_usage


def main() -> int:
    banner("SESSION 2 ENVIRONMENT CHECK")
    print(f"  model  {MODEL_ID}")
    print(f"  region {REGION}")

    if PROVIDER != "bedrock":
        print(f"\n  NOTE: PROVIDER is {PROVIDER!r}, so the LangChain check below\n"
              f"  would test Groq, not Bedrock. Fix with:\n"
              f"    export LLM_PROVIDER=bedrock")
        return 1

    # 1 + 2: credentials and region. bedrock_runtime() exits with a readable
    # message rather than letting botocore raise from somewhere deep.
    client = bedrock_runtime()
    ident = None
    try:
        import boto3
        ident = boto3.client("sts", region_name=REGION).get_caller_identity()
        print(f"  caller {ident.get('Arn', '?')}")
    except Exception as exc:
        print(f"  (could not call STS: {exc} — not fatal)")

    # 3: can we actually invoke the model? This is the check that matters; having
    # credentials does not mean the account has this model enabled in this region.
    print("\n  invoking the model...")
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
        "max_tokens": 16,
        "temperature": 0,
    })
    try:
        resp = client.invoke_model(modelId=MODEL_ID, body=body)
    except client.exceptions.AccessDeniedException:
        print(f"\nFAILED: credentials work, but this account cannot invoke\n"
              f"  {MODEL_ID}\n  in {REGION}.\n\n"
              f"  Usually means the model is not enabled for the account, or the\n"
              f"  role lacks bedrock:InvokeModel. Ask the instructor.")
        return 1
    except client.exceptions.ValidationException as exc:
        print(f"\nFAILED: Bedrock rejected the request: {exc}\n"
              f"  If this mentions the model id, the model may not exist in\n"
              f"  {REGION}. Try: export BEDROCK_MODEL=<id from the console>")
        return 1
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        return 1

    envelope = json.loads(resp["body"].read())
    text = envelope["content"][0]["text"].strip()
    print(f"  model said: {text!r}")
    report_usage("invoke_model", envelope.get("usage", {}))

    # 4: the LangChain path, used by labs 02-04.
    print("\n  checking langchain-aws...")
    try:
        from _common import chat_model
        msg = chat_model().invoke("Reply with exactly: ready")
        print(f"  langchain said: {msg.content!r}")
        report_usage("ChatBedrockConverse", msg.usage_metadata or {})
    except ImportError as exc:
        print(f"\nFAILED: {exc}\n  Run: uv sync")
        return 1
    except Exception as exc:
        print(f"\nFAILED on the LangChain path: {type(exc).__name__}: {exc}")
        return 1

    banner("OK — you are ready for every lab in this session")
    return 0


if __name__ == "__main__":
    sys.exit(main())
