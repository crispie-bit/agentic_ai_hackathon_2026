"""
§2 · 06 — PROMPT LAB. Three TODOs, three scores to beat.

    uv run section_2_agentic_ai_basic/06_prompt_engineering.py
    uv run section_2_agentic_ai_basic/06_prompt_engineering.py play


WHAT THIS IS

    06a, 06b and 06c were demos you read. This is the same three techniques
    with the training wheels off. Each TODO is graded against a fixed test set
    whose answers are already written down, so you are never asking whether a
    prompt "reads well" — you are watching a number.

        TODO 1   zero-shot          8 tickets, you need 8/8
        TODO 2   few-shot           6 tickets, you need 6/6
        TODO 3   chain of thought   4 problems, you need 4/4

    Edit the EDIT ZONE, re-run, repeat. Every run costs 24 model calls, so
    think before you re-run.


THE ONE RULE

    You may only change what is inside the EDIT ZONE. The instruction in
    TODO 2 is locked on purpose: that challenge is about the examples, and
    nothing else.
"""

import re
import sys
import time

import _bootstrap  # noqa: F401

from _common import chat_model, model_label
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ##########################################################################
#                              EDIT ZONE
# ##########################################################################

MY_INSTRUCTION = """Classify each customer message as exactly one of: billing,
account, technical. Use billing for incorrect charges, prices, refunds, or
payment disputes; account for subscription cancellation and changes to users
or login details; technical for software errors, broken features, or anything
that fails to work. Classify by the primary problem: a broken invoice download
is technical even though it mentions an invoice. Reply with only the lowercase
label."""

MY_EXAMPLES = [
    ("My invoice has the wrong VAT number.", "billing"),
    ("I was billed in the wrong currency.", "billing"),
    ("There is a charge from a device I don't recognise.", "security"),
    ("The export button does nothing.", "engineering"),
    ("The dashboard is blank on Safari.", "engineering"),
]

MY_MATH_INSTRUCTION = """Solve the problem step by step. End your response with
the final amount on its own line in exactly this format: TOTAL: <number>"""

# ##########################################################################

LOCKED_INSTRUCTION = ("Route this ticket to one team: billing, security or "
                      "engineering. Reply with one word, lowercase.")

TRAINING_DATA = [
    ("My invoice has the wrong VAT number.", "billing"),
    ("The annual charge should have been prorated.", "billing"),
    ("I was billed in the wrong currency.", "billing"),
    ("There is a charge from a device I don't recognise.", "security"),
    ("The export button does nothing.", "engineering"),
    ("The dashboard is blank on Safari.", "engineering"),
    ("Charts render off the screen on mobile.", "engineering"),
]

TEST_1 = [
    ("The payment page throws a 500 error when I click pay.", "technical"),
    ("I can't afford this anymore, close my subscription.", "account"),
    ("You charged me $19 but the plan page says $9.", "billing"),
    ("My photos upload but never appear in the gallery.", "technical"),
    ("Change the email on my login to my work address.", "account"),
    ("The invoice PDF won't download.", "technical"),
    ("Refund the annual charge, I cancelled in June.", "billing"),
    ("Add my colleague as a second user on my plan.", "account"),
]

TEST_2 = [
    ("I see a payment I never made on my statement.", "security"),
    ("The tax on my receipt looks wrong.", "billing"),
    ("A device I don't own is in my active sessions.", "security"),
    ("Uploads fail silently on large files.", "engineering"),
    ("I was charged in dollars instead of euros.", "billing"),
    ("The search bar returns nothing at all.", "engineering"),
]

TEST_3 = [
    ("3 bags at $18 and 2 mugs at $12. A $15 coupon, then 10% off the "
     "rest. Shipping $6, never discounted. Total?", 62.70),
    ("A $240 order, 25% off, then $8 shipping added. Total?", 188.00),
    ("5 boxes at $14. Buy 4 get 1 free. Then 10% off. Total?", 50.40),
    ("A plan is $30 a month. Pay yearly and 2 months are free. There is a "
     "one-off $25 setup fee. What does year one cost?", 325.00),
]


def clean(text):
    return text.strip().strip(".!?\"'`").strip().lower()


def total_line(text):
    """Pull the number off the TOTAL: line.

    The leading [^A-Za-z0-9]* is deliberate: models love to answer with
    **TOTAL: 62.70**, and failing that would send you off rewriting a prompt
    that was already right. The line still has to be its own line.
    """
    found = re.findall(r"(?mi)^[^A-Za-z0-9]*TOTAL:\s*\$?([\d,]+(?:\.\d+)?)",
                       text)
    return round(float(found[-1].replace(",", "")), 2) if found else None


def build(instruction, ticket, examples=()):
    messages = [SystemMessage(instruction)]
    for text, label in examples:
        messages += [HumanMessage(text), AIMessage(label)]
    return messages + [HumanMessage(ticket)]


def ask(model, messages, attempts=3):
    """Invoke, retrying briefly. Returns None if it never got through.

    A room full of people shares one rate limit. Without this, one 429 in the
    middle of challenge 2 throws away every result you had already paid for.
    """
    for n in range(attempts):
        try:
            return model.invoke(messages).content
        except Exception as exc:
            if n == attempts - 1:
                print(f"  (gave up after {attempts}: {type(exc).__name__})")
                return None
            time.sleep(2 * (n + 1))


def score(model, title, instruction, tests, examples=(), parse=clean):
    print(f"\n--- {title} ---")
    hits = 0
    for item, want in tests:
        raw = ask(model, build(instruction, item, examples))
        got = parse(raw) if raw is not None else None
        hits += got == want
        mark = "PASS" if got == want else ("ERR " if raw is None else "FAIL")
        shown = " ".join(str(got).split())
        shown = shown[:22] + "..." if len(shown) > 22 else shown
        print(f"  {mark}  got={shown:<26} want={str(want):<12} {item[:38]}")
    total = len(tests)
    print(f"  [{'#' * hits}{'.' * (total - hits)}] {hits}/{total}")
    return hits == total


def check_examples():
    head = "\n--- 2b. FEW-SHOT, your examples ---"
    rows = [tuple(row) for row in MY_EXAMPLES]
    if len(rows) != 5:
        print(f"{head}\n  blocked: need exactly 5 rows, you have "
              f"{len(rows)}. Pick 5 of the 7 in TRAINING_DATA.")
        return False
    if len(set(rows)) != 5:
        print(f"{head}\n  blocked: 5 DIFFERENT rows — no duplicates.")
        return False
    for row in rows:
        if row not in TRAINING_DATA:
            print(f"{head}\n  blocked: rows must be copied from "
                  f"TRAINING_DATA unedited. This one is not:\n    {row}")
            return False
    return True


def challenge_two(model):
    """Baseline first, then your examples, so the lift is on screen.

    The whole point of few-shot is that examples reach what an instruction
    cannot. Printing only the second number would make that a claim; printing
    both makes it an observation.
    """
    score(model, "2a. FEW-SHOT, NO examples (the baseline to beat)",
          LOCKED_INSTRUCTION, TEST_2)
    if not check_examples():
        return False
    return score(model, "2b. FEW-SHOT, your examples", LOCKED_INSTRUCTION,
                 TEST_2, MY_EXAMPLES)


def play(model):
    print("\nType a ticket. Blank line to quit.\n")
    while True:
        try:
            ticket = input("ticket> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not ticket:
            break
        zero = ask(model, build(MY_INSTRUCTION, ticket))
        few = ask(model, build(LOCKED_INSTRUCTION, ticket, MY_EXAMPLES))
        print(f"  zero-shot: {clean(zero) if zero else '(failed)'}")
        print(f"  few-shot:  {clean(few) if few else '(failed)'}\n")


def main():
    print(f"model: {model_label()}")
    model = chat_model(temperature=0)

    if len(sys.argv) > 1 and sys.argv[1] == "play":
        play(model)
        return

    passed = [
        score(model, "1. ZERO-SHOT", MY_INSTRUCTION, TEST_1),
        challenge_two(model),
        score(model, "3. CHAIN OF THOUGHT", MY_MATH_INSTRUCTION, TEST_3,
              parse=total_line),
    ]
    print(f"\n=== {sum(passed)}/3 challenges beaten ===")


if __name__ == "__main__":
    main()
