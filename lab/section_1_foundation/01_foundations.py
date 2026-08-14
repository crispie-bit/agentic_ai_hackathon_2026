"""
LAB 1 — what an LLM call actually is.  (8 minutes)

    uv run section_1_foundation/01_foundations.py

Needs GROQ_API_KEY. If this file exits telling you so, you skipped
`uv run 00_check_groq.py` — run that first, it explains how to get one.

No LangChain here on purpose. You are looking at the raw request and the raw
response, because every framework in the next two sessions is a wrapper around
exactly this: a list of messages in, one message and a token count out.

# ======================================================================
# TODO(student) 1 — part B, statelessness.  The second call has no idea who
#   you are, because the API kept nothing. Rebuild `second_turn` so it
#   carries the whole conversation: the first question, the model's reply,
#   then the new question. Two lines. Re-run.
#
#   That list IS the memory. There is no other kind.
#
# TODO(student) 2 — part C, truncation.  Set MAX_TOKENS below to 10 and
#   re-run. The JSON stops mid-structure and json.loads fails. Look at
#   finish_reason before you blame the model: it says "length". The model
#   did not produce bad JSON — you cut off good JSON.
#
# TODO(student) 3 — part D, sampling.  Set TEMPERATURE to 1.0 and re-run.
#   Compare the three answers with what you got at 0.0. Note that 0.0 is
#   very consistent but is NOT a guarantee — which is why extraction
#   pipelines validate output instead of trusting it (Lab 2, part C).
# ======================================================================

"""

import json

import _bootstrap  # noqa: F401

from _common import GROQ_MODEL, banner, groq_client, report_usage

MAX_TOKENS = 300      # TODO 2: make this 10
TEMPERATURE = 0.0     # TODO 3: make this 1.0


def ask(client, messages, max_tokens=None, temperature=None):
    """One API call. Returns (text, finish_reason, usage).

    This is the whole interface. Everything else in this course is built on it.
    """
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        # max_completion_tokens caps OUTPUT only — it is not a budget for the
        # whole call, and it does not limit how much you send.
        max_completion_tokens=MAX_TOKENS if max_tokens is None else max_tokens,
        temperature=TEMPERATURE if temperature is None else temperature,
    )
    choice = resp.choices[0]
    return choice.message.content, choice.finish_reason, resp.usage


# --------------------------------------------------------------------------
# A. The shape of a request and a response.
# --------------------------------------------------------------------------

def part_a_shape(client) -> None:
    banner("A. request in, response out")

    messages = [
        # `system` is a message with a role, like any other. Some providers
        # take it as a separate argument instead — you will meet that on
        # Bedrock tomorrow, and it is the single most common porting bug.
        {"role": "system", "content": "You are terse. Answer in one sentence."},
        {"role": "user", "content": "What is a context window?"},
    ]

    print("  what we send:")
    print(json.dumps({"model": GROQ_MODEL, "messages": messages,
                      "max_completion_tokens": MAX_TOKENS,
                      "temperature": TEMPERATURE}, indent=2)[:600])

    text, finish, usage = ask(client, messages)
    print(f"\n  content:       {text!r}")
    print(f"  finish_reason: {finish}      <- 'stop' means it finished on its own")
    report_usage("part A", usage)
    print("\n  Both numbers are billed. `input` is everything you sent, re-sent\n"
          "  in full on every call — which is what the next part is about.")


# --------------------------------------------------------------------------
# B. Statelessness. The one idea the rest of the course is built on.
# --------------------------------------------------------------------------

def part_b_statelessness(client) -> None:
    banner("B. the model remembers nothing")

    first_turn = [
        {"role": "user",
         "content": "My name is Alex and I work on the payments team."},
    ]
    reply, _, usage1 = ask(client, first_turn)
    print(f"  call 1 -> {reply.strip()[:100]!r}")
    report_usage("call 1", usage1)

    # A second, completely independent HTTP request.
    second_turn = [
        {"role": "user", "content": "What is my name, and which team am I on?"},
    ]
    # TODO 1: rebuild second_turn to carry the history. You need the first
    #   user message, then {"role": "assistant", "content": reply}, then the
    #   question above.

    answer, _, usage2 = ask(client, second_turn)
    print(f"\n  call 2 -> {answer.strip()[:160]!r}")
    report_usage("call 2", usage2)

    knows = "alex" in answer.lower()
    print(f"\n  Does call 2 know the name?  {'YES' if knows else 'NO'}")
    if not knows:
        print("  Nothing was kept between the two calls. That is not a bug and\n"
              "  not a missing feature — it is what the API is. TODO 1 fixes it\n"
              "  the only way it can be fixed: by re-sending the conversation.")
    else:
        print("  You re-sent the history, so the model can read it. Watch the\n"
              "  input token count on call 2 — memory is not free, you pay for\n"
              "  the whole transcript on every single turn.")


# --------------------------------------------------------------------------
# C. Truncation, and the JSON error that isn't one.
# --------------------------------------------------------------------------

def part_c_truncation(client) -> None:
    banner(f"C. max_completion_tokens = {MAX_TOKENS}")

    messages = [
        {"role": "system",
         "content": 'Reply with JSON only, exactly: '
                    '{"answer": "<two sentences>", "confidence": "high|medium|low"}'},
        {"role": "user", "content": "Why is an LLM API call stateless?"},
    ]
    text, finish, usage = ask(client, messages)
    print(f"  finish_reason: {finish}")
    print(f"  raw text:      {text!r}")

    body = text.strip()
    if body.startswith("```"):
        # Models wrap JSON in markdown fences even when told not to. Do not
        # fight it in the prompt; strip it and move on. You will re-write this
        # function in every extraction pipeline you ever build.
        body = body.split("\n", 1)[-1] if "\n" in body else body[3:]
        body = body.strip().removesuffix("```").strip()
        print("  note:          it fenced the JSON despite being told not to.")

    try:
        print(f"  parsed:        {json.loads(body)}")
    except json.JSONDecodeError as exc:
        print(f"  !! json.loads failed: {exc}")
        if finish == "length":
            print("     finish_reason is 'length'. The model did not return bad\n"
                  "     JSON — you truncated good JSON. That is TODO 2.")
        else:
            print("     finish_reason is not 'length', so this really is\n"
                  "     malformed output. Now it is worth changing the prompt.")
    report_usage("part C", usage)


# --------------------------------------------------------------------------
# D. Sampling.
# --------------------------------------------------------------------------

def part_d_sampling(client) -> None:
    banner(f"D. temperature = {TEMPERATURE}, same question three times")

    messages = [{"role": "user",
                 "content": "Name one everyday use for an AI agent. "
                            "Reply with the use case only, under eight words."}]
    for i in range(3):
        text, _, _ = ask(client, messages, max_tokens=32)
        print(f"  run {i + 1}: {text.strip()!r}")

    print("\n  temperature 0 asks for the most likely token every time. It is\n"
          "  consistent, not deterministic — same prompt, same settings, and you\n"
          "  can still get different text. Which is TODO 3, and the reason Lab 2\n"
          "  validates the model's output rather than trusting its shape.")


def main() -> None:
    client = groq_client()
    print(f"model: {GROQ_MODEL}")
    part_a_shape(client)
    part_b_statelessness(client)
    part_c_truncation(client)
    part_d_sampling(client)

    banner("What to take away")
    print("""  - a call is: messages in -> one message + a token count out
  - the API keeps NOTHING. Memory is you re-sending the transcript.
  - so every turn costs the whole conversation again, in input tokens
  - max tokens caps OUTPUT; too low truncates mid-JSON
  - check finish_reason BEFORE you blame the model for bad output
  - temperature 0 is consistent, not guaranteed""")


if __name__ == "__main__":
    main()
