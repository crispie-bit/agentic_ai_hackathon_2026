"""
§3 · 04 — Two models, one prompt, one cost comparison.

    uv run section_3_bedrock/04_cost_and_optimisation.py

Sends the same prompt to two models and prints what each one cost. Credentials
come from bedrock_runtime() in _common, so the active SSO session is used —
run `aws sso login --profile <your-profile>` first if the token has expired.

Edit the settings block below to change the prompt or the pair being compared.

The output tells you the price difference. It does NOT tell you whether the
cheaper answer is good enough — that is a judgement you make by reading both,
which is why each answer is printed underneath.
"""

import json

import _bootstrap  # noqa: F401

from _common import bedrock_runtime

# ---- settings ------------------------------------------------------------
PROMPT = "Classify this ticket in one word: the office printer is offline."
COMPARE = ["haiku", "sonnet"]   # any two keys from MODELS
CALLS = 5000                    # projected number of calls
MAX_TOKENS = 500

# Cache rates, as a multiple of that model's input rate. Roughly consistent
# across models, unlike the absolute figures.
CACHE_READ_MULTIPLIER = 0.10    # a tenth of input
CACHE_WRITE_MULTIPLIER = 1.25   # a little more than input

# The unoptimised version of the same job: a bloated standing prompt, and an
# instruction that invites a long answer. Both are extremely common.
VERBOSE_SYSTEM = (
    "You are an IT service desk assistant for a large organisation.\n"
    "Follow these standing rules when classifying a ticket:\n"
    + "\n".join(
        f"{i}. Rule {i}: consider the affected system, the user's department, "
        f"the urgency, the likely root cause, and any related recent incidents "
        f"before deciding on a category."
        for i in range(1, 120)
    )
)
CHATTY_PROMPT = ("Classify this ticket: the office printer is offline. "
                 "Explain your reasoning in full before giving the category.")
TERSE_SYSTEM = "You classify IT tickets. Reply with one word."

# ==========================================================================
# IDS AND RATES ARE NOT PART OF THE CODE. Both differ per region and move
# over time. Copy from https://aws.amazon.com/bedrock/pricing and from the
# Bedrock console before relying on them. USD per MILLION tokens.
#
# NOTE THE PREFIX. Outside us-east-1 you generally cannot invoke a bare id
# like "anthropic.claude-sonnet-4-20250514-v1:0" — it returns "The provided
# model identifier is invalid." You need a cross-region INFERENCE PROFILE:
#
#     global.*   routed across regions worldwide
#     apac.*     routed within Asia-Pacific
#
# List the ones your account can use with:
#
#     aws bedrock list-inference-profiles --region ap-southeast-1 \
#       --query 'inferenceProfileSummaries[].inferenceProfileId'
#
# These three are verified working in ap-southeast-1. If you are elsewhere,
# run that command and paste your own ids in.
# ==========================================================================
MODELS = {
    "haiku": {
        "id": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "input": 1.00,
        "output": 5.00,
    },
    "sonnet": {
        "id": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "input": 3.00,
        "output": 15.00,
    },
    "opus": {
        "id": "global.anthropic.claude-opus-4-5-20251101-v1:0",
        "input": 5.00,
        "output": 25.00,
    },
}
# --------------------------------------------------------------------------


def run(client, model: dict) -> tuple[int, int, str]:
    """Invoke one model. Returns (input_tokens, output_tokens, answer)."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    })
    try:
        payload = json.loads(
            client.invoke_model(modelId=model["id"], body=body)["body"].read())
    except client.exceptions.ValidationException as exc:
        raise SystemExit(
            f"\n{model['id']}\nwas rejected: {exc}\n\n"
            "  Almost always the model id, not your credentials. Outside\n"
            "  us-east-1 you need a global.* or apac.* inference profile.\n"
            "  List what this account can invoke:\n\n"
            "    aws bedrock list-inference-profiles --region ap-southeast-1 \\\n"
            "      --query 'inferenceProfileSummaries[].inferenceProfileId'\n"
        ) from exc
    except client.exceptions.AccessDeniedException as exc:
        raise SystemExit(
            f"\n{model['id']}\nis a valid id, but this account cannot invoke it: {exc}\n\n"
            "  Enable it under Bedrock -> Model access, in this region.\n"
        ) from exc

    usage = payload["usage"]
    answer = "".join(b.get("text", "") for b in payload["content"]).strip()
    return usage["input_tokens"], usage["output_tokens"], answer


def cost_usd(model: dict, in_tokens: int, out_tokens: int,
             cache_read: int = 0, cache_write: int = 0) -> float:
    return (in_tokens * model["input"]
            + out_tokens * model["output"]
            + cache_read * model["input"] * CACHE_READ_MULTIPLIER
            + cache_write * model["input"] * CACHE_WRITE_MULTIPLIER) / 1_000_000


# --------------------------------------------------------------------------
# Optimisation. Same task, four ways, each one measured.
# --------------------------------------------------------------------------

def measure(client, model: dict, prompt: str, system: list | None = None,
            max_tokens: int = 500) -> dict:
    """Run one variant and return its token counts and cost."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if system:
        body["system"] = system

    payload = json.loads(
        client.invoke_model(modelId=model["id"], body=json.dumps(body))["body"].read())
    u = payload["usage"]
    fields = {
        "in": u.get("input_tokens", 0),
        "out": u.get("output_tokens", 0),
        "cache_read": u.get("cache_read_input_tokens", 0),
        "cache_write": u.get("cache_creation_input_tokens", 0),
        "answer": "".join(b.get("text", "") for b in payload["content"]).strip(),
    }
    fields["cost"] = cost_usd(model, fields["in"], fields["out"],
                              fields["cache_read"], fields["cache_write"])
    return fields


def row(label: str, f: dict) -> None:
    print(f"  {label:26}{f['in']:>7}{f['out']:>6}{f['cache_read']:>8}"
          f"{f['cost']:>12.6f}{f['cost'] * CALLS:>11.2f}")


def main() -> None:
    client = bedrock_runtime()

    print(f"prompt: {PROMPT}\n")

    results = []
    for name in COMPARE:
        model = MODELS[name]
        in_tok, out_tok, answer = run(client, model)
        results.append({"name": name, "in": in_tok, "out": out_tok,
                        "answer": answer,
                        "cost": cost_usd(model, in_tok, out_tok)})

    print(f"{'model':10}{'in':>8}{'out':>8}{'per call':>14}{f'x{CALLS}':>14}")
    print("-" * 54)
    for r in results:
        print(f"{r['name']:10}{r['in']:>8}{r['out']:>8}"
              f"{r['cost']:>14.6f}{r['cost'] * CALLS:>14.2f}")

    cheap, pricey = sorted(results, key=lambda r: r["cost"])
    ratio = pricey["cost"] / max(cheap["cost"], 1e-9)
    saved = (pricey["cost"] - cheap["cost"]) * CALLS
    print(f"\n{cheap['name']} is {ratio:.1f}x cheaper "
          f"— ${saved:.2f} per {CALLS} calls")

    # The number above is only half the decision. Print both answers so the
    # other half — is the cheap one still right? — is visible too.
    for r in results:
        print(f"\n--- {r['name']} ---\n{r['answer']}")
    print("\nCheaper only wins if that answer is still correct.")

    optimise(client)


def optimise(client) -> None:
    """The same classification job, optimised one lever at a time."""
    model = MODELS[COMPARE[0]]

    print(f"\n\noptimising the same job on {COMPARE[0]}, {CALLS} calls")
    print(f"  {'':26}{'in':>7}{'out':>6}{'cached':>8}{'per call':>12}"
          f"{f'x{CALLS}':>11}")
    print("  " + "-" * 70)

    # 0. The baseline: bloated system prompt, and an answer that rambles.
    base = measure(client, model, CHATTY_PROMPT,
                   [{"type": "text", "text": VERBOSE_SYSTEM}])
    row("baseline", base)

    # 1. Trim the standing prompt. 119 rules nobody reads -> one sentence.
    trimmed = measure(client, model, CHATTY_PROMPT,
                      [{"type": "text", "text": TERSE_SYSTEM}])
    row("+ trim the system prompt", trimmed)

    # 2. Constrain the OUTPUT. It bills at ~5x input, so this is the lever
    #    people most often forget.
    terse = measure(client, model, PROMPT,
                    [{"type": "text", "text": TERSE_SYSTEM}], max_tokens=10)
    row("+ ask for one word", terse)

    # 3. If the long prompt is genuinely needed, cache it instead of trimming.
    cached_system = [{"type": "text", "text": VERBOSE_SYSTEM,
                      "cache_control": {"type": "ephemeral"}}]
    measure(client, model, PROMPT, cached_system, max_tokens=10)   # warm it
    cached = measure(client, model, PROMPT, cached_system, max_tokens=10)
    row("(or: cache it instead)", cached)

    saving = base["cost"] / max(terse["cost"], 1e-9)
    print(f"\n  baseline   ${base['cost']:.6f}/call   ${base['cost'] * CALLS:.2f} "
          f"per {CALLS}")
    print(f"  optimised  ${terse['cost']:.6f}/call   ${terse['cost'] * CALLS:.2f} "
          f"per {CALLS}")
    print(f"  saving     {saving:.0f}x  — "
          f"${(base['cost'] - terse['cost']) * CALLS:.2f} per {CALLS} calls")


if __name__ == "__main__":
    main()