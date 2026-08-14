"""
Run this FIRST, at the start of Session 1.  (2 minutes)

    uv run 00_check_groq.py

Checks the three things that actually go wrong on Day 1:
  1. GROQ_API_KEY is set
  2. the raw Groq SDK can reach the model
  3. langchain-groq is installed and can reach the same model

If it prints OK, all three Session 1 labs will run. If it fails, raise your
hand — the message says which of the three broke.

No AWS. Nothing here costs money.
"""

import sys

import _bootstrap  # noqa: F401

from _common import GROQ_MODEL, banner, chat_model, groq_client, report_usage


def main() -> int:
    banner("SESSION 1 ENVIRONMENT CHECK")
    print(f"  model {GROQ_MODEL}")

    # 1 + 2. groq_client() exits with instructions if the key is missing.
    client = groq_client()
    print("\n  calling the raw Groq SDK...")
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: ready"}],
            max_completion_tokens=16,
            temperature=0,
        )
    except Exception as exc:
        name = type(exc).__name__
        print(f"\nFAILED: {name}: {exc}\n")
        if "authentication" in str(exc).lower() or "401" in str(exc):
            print("  The key was rejected. Copy it again from console.groq.com —\n"
                  "  a trailing space or a truncated paste is the usual cause.")
        elif "model" in str(exc).lower():
            print(f"  Groq did not recognise {GROQ_MODEL!r}. Model ids change.\n"
                  "  Check https://console.groq.com/docs/models and then:\n"
                  "    export GROQ_MODEL=<id from that page>")
        elif "rate" in str(exc).lower() or "429" in str(exc):
            print("  Rate limited on the free tier. Wait 60 seconds and re-run.")
        return 1

    print(f"  model said: {resp.choices[0].message.content.strip()!r}")
    report_usage("groq sdk", resp.usage)

    # 3. The LangChain path, used by labs 02 and 03.
    print("\n  checking langchain-groq...")
    try:
        msg = chat_model().invoke("Reply with exactly: ready")
    except ImportError as exc:
        print(f"\nFAILED: {exc}\n  Run: uv sync")
        return 1
    except Exception as exc:
        print(f"\nFAILED on the LangChain path: {type(exc).__name__}: {exc}")
        return 1

    print(f"  langchain said: {msg.content.strip()!r}")
    report_usage("ChatGroq", msg.usage_metadata)

    banner("OK — you are ready for all three Session 1 labs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
