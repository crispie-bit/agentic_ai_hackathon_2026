"""
LAB 1 — what an LLM call actually is.

    uv run section_1_foundation/01_foundations.py

Needs GROQ_API_KEY. If this exits telling you so, run 00_check_groq.py first.

No LangChain here on purpose. This is the raw request and the raw response,
because every framework in the sessions that follow wraps exactly this: a list
of messages in, one message and a token count out.

Three TODOs, marked in the code below. Parts A to D run in order.
"""

import json

import _bootstrap  # noqa: F401

from _common import GROQ_MODEL, banner, groq_client, report_usage

MAX_TOKENS = 300
TRUNCATION_MAX_TOKENS = 10
TEMPERATURE = 1.0     # TODO 3 changes this


def ask(client, messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    """One API call. Returns (text, finish_reason, usage).

    This is the whole interface. Everything else in the course is built on it.
    """
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        # Caps the OUTPUT only. Not a budget for the call, and no limit on
        # how much you send.
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    choice = resp.choices[0]
    return choice.message.content, choice.finish_reason, resp.usage


# --------------------------------------------------------------------------
# A. The shape of a request and a response.  No change needed — just read it.
# --------------------------------------------------------------------------

def part_a_shape(client) -> None:
    banner("A. request in, response out")

    messages = [
        # `system` is a message with a role, like any other. Some providers
        # take it as a separate top-level argument instead. That difference is
        # the most common porting bug.
        {"role": "system", "content": "You are terse. Answer in one sentence."},
        {"role": "user", "content": "What is a context window?"},
    ]

    print("  what we send:")
    print(json.dumps({"model": GROQ_MODEL,
                      "messages": messages,
                      "max_completion_tokens": MAX_TOKENS,
                      "temperature": TEMPERATURE}, indent=2))

    text, finish, usage = ask(client, messages)
    print(f"\n  content:       {text!r}")
    print(f"  finish_reason: {finish}   <- 'stop' means it finished on its own")
    report_usage("part A", usage)


# --------------------------------------------------------------------------
# B. Statelessness.
# --------------------------------------------------------------------------

def part_b_statelessness(client) -> None:
    banner("B. the model remembers nothing")

    question_1 = {"role": "user",
                  "content": "My name is Alex and I work on the payments team."}
    question_2 = {"role": "user",
                  "content": "What is my name, and which team am I on?"}

    reply, _, usage_1 = ask(client, [question_1])
    print(f"  call 1 -> {reply.strip()[:100]!r}")
    report_usage("call 1", usage_1)

    # A second, completely independent request. It carries only question_2, so
    # the model has never seen the name.
    #
    # TODO 1 -------------------------------------------------------------
    #   Rebuild second_turn so it carries the whole conversation: the first
    #   question, the model's reply, then the new question. One line.
    #
    #       second_turn = [question_1,
    #                      {"role": "assistant", "content": reply},
    #                      question_2]
    #
    #   That list IS the memory. There is no other kind.
    # ---------------------------------------------------------------------
    second_turn = [
        question_1,
        {"role": "assistant", "content": reply},
        question_2,
    ]

    answer, _, usage_2 = ask(client, second_turn)
    print(f"\n  call 2 -> {answer.strip()[:160]!r}")
    report_usage("call 2", usage_2)

    knows = "alex" in answer.lower()
    print(f"\n  Does call 2 know the name?  {'YES' if knows else 'NO'}")
    if knows:
        print("  You re-sent the history, so the model can read it. Compare the\n"
              "  input token count on call 2 against call 1.")
    else:
        print("  Nothing was kept between the calls. That is what the API is,\n"
              "  and TODO 1 fixes it the only way it can be fixed.")


# --------------------------------------------------------------------------
# C. Truncation, and the JSON error that isn't one.
# --------------------------------------------------------------------------

def part_c_truncation(client) -> None:
    banner(f"C. max_completion_tokens = {TRUNCATION_MAX_TOKENS}")

    messages = [
        {"role": "system",
         "content": 'Reply with JSON only, exactly: '
                    '{"answer": "<two sentences>", "confidence": "high|medium|low"}'},
        {"role": "user", "content": "Why is an LLM API call stateless?"},
    ]
    text, finish, usage = ask(client, messages, max_tokens=TRUNCATION_MAX_TOKENS)
    print(f"  finish_reason: {finish}")
    print(f"  raw text:      {text!r}")

    body = text.strip()
    if body.startswith("```"):
        # Models fence JSON even when told not to. Strip it in code rather
        # than fighting it in the prompt.
        body = body.split("\n", 1)[-1] if "\n" in body else body[3:]
        body = body.strip().removesuffix("```").strip()
        print("  note:          it fenced the JSON despite being told not to.")

    # TODO 2 -------------------------------------------------------------
    #   TRUNCATION_MAX_TOKENS is set to 10 above. The JSON stops mid-structure
    #   and json.loads fails.
    #
    #   Read finish_reason before blaming the model. It says "length": the
    #   model returned good JSON, and you cut it off.
    # ---------------------------------------------------------------------
    try:
        print(f"  parsed:        {json.loads(body)}")
    except json.JSONDecodeError as exc:
        print(f"  !! json.loads failed: {exc}")
        if finish == "length":
            print("     finish_reason is 'length' — you truncated good JSON.")
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

    # TODO 3 -------------------------------------------------------------
    #   Set TEMPERATURE at the top of this file to 1.0 and re-run. Compare
    #   the three answers with what you got at 0.0.
    #
    #   Temperature 0 asks for the most likely token every time. It is
    #   consistent, but not guaranteed identical.
    # ---------------------------------------------------------------------
    for i in range(3):
        text, _, _ = ask(client, messages, max_tokens=32)
        print(f"  run {i + 1}: {text.strip()!r}")


def main() -> None:
    client = groq_client()
    print(f"model: {GROQ_MODEL}")
    part_a_shape(client)
    part_b_statelessness(client)
    part_c_truncation(client)
    part_d_sampling(client)


if __name__ == "__main__":
    main()
