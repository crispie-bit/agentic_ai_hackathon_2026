"""
§2 · 06 — PROMPT LAB. Three TODOs, three scores to beat.

    uv run section_2_agentic_ai_basic/06_prompt_engineering.py
    uv run section_2_agentic_ai_basic/06_prompt_engineering.py play

Zero-shot, few-shot and chain of thought — 06a, 06b and 06c — with the
training wheels off. Each TODO is graded against a fixed test set, so you are
not asking whether a prompt reads well; you are watching a number.

Edit the EDIT ZONE, re-run, repeat.
"""

import re
import sys

import _bootstrap  # noqa: F401

from _common import chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ##########################################################################
#                              EDIT ZONE
# ##########################################################################

# TODO 1 — get 8/8 on TEST_1. Zero-shot: the instruction is all you get.
# The label set is not written down anywhere. Read TEST_1 and work out what
# the three labels are, then name them here — a model cannot pick a category
# it has not been told exists. Two of the tickets sit on a boundary.
MY_INSTRUCTION = "Sort the message."

# TODO 2 — get 6/6 on TEST_2. The instruction is locked, so the only thing
# you can change is which examples the model sees. Exactly 5 rows, copied
# from TRAINING_DATA unedited. Which 5 is the whole exercise.
MY_EXAMPLES = [
    # ("ticket text", "label"),
]

# TODO 3 — get 4/4 on TEST_3. The grader is not reading your prose, only a
# line formatted:  TOTAL: <number>
# Getting the arithmetic right and getting it parsed are two problems.
MY_MATH_INSTRUCTION = "Reply with the final number only."

# ##########################################################################

LOCKED_INSTRUCTION = ("Route this ticket to one team: billing, security or "
                      "engineering. Reply with one word, lowercase.")

TRAINING_DATA = [
    ("My invoice has the wrong VAT number.", "billing"),
    ("The annual charge should have been prorated.", "billing"),
    ("I was billed in the wrong currency.", "billing"),
    ("There is a charge from a device I don't recognise.", "security"),
    ("Someone logged into my account from Brazil.", "security"),
    ("I got an email asking for my password, is it from you?", "security"),
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
    found = re.findall(r"(?mi)^\s*TOTAL:\s*\$?([\d,]+(?:\.\d+)?)", text)
    return round(float(found[-1].replace(",", "")), 2) if found else None


def build(instruction, ticket, examples=()):
    messages = [SystemMessage(instruction)]
    for text, label in examples:
        messages += [HumanMessage(text), AIMessage(label)]
    return messages + [HumanMessage(ticket)]


def score(model, title, instruction, tests, examples=(), parse=clean):
    print(f"\n--- {title} ---")
    hits = 0
    for item, want in tests:
        got = parse(model.invoke(build(instruction, item, examples)).content)
        hits += got == want
        mark = "PASS" if got == want else "FAIL"
        shown = " ".join(str(got).split())
        shown = shown[:22] + "..." if len(shown) > 22 else shown
        print(f"  {mark}  got={shown:<26} want={str(want):<12} {item[:38]}")
    total = len(tests)
    print(f"  [{'#' * hits}{'.' * (total - hits)}] {hits}/{total}")
    return hits == total


def check_examples():
    if len(MY_EXAMPLES) != 5:
        print(f"\n--- 2. FEW-SHOT ---\n  blocked: need exactly 5 rows, "
              f"you have {len(MY_EXAMPLES)}")
        return False
    if any(row not in TRAINING_DATA for row in MY_EXAMPLES):
        print("\n--- 2. FEW-SHOT ---\n  blocked: rows must be copied from "
              "TRAINING_DATA, unedited")
        return False
    return True


def play(model):
    print("\nType a ticket. Blank line to quit.\n")
    while True:
        try:
            ticket = input("ticket> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not ticket:
            break
        zero = clean(model.invoke(build(MY_INSTRUCTION, ticket)).content)
        few = clean(model.invoke(
            build(LOCKED_INSTRUCTION, ticket, MY_EXAMPLES)).content)
        print(f"  zero-shot: {zero}")
        print(f"  few-shot:  {few}\n")


def main():
    model = chat_model(temperature=0)

    if len(sys.argv) > 1 and sys.argv[1] == "play":
        play(model)
        return

    passed = [
        score(model, "1. ZERO-SHOT", MY_INSTRUCTION, TEST_1),
        check_examples() and score(model, "2. FEW-SHOT", LOCKED_INSTRUCTION,
                                   TEST_2, MY_EXAMPLES),
        score(model, "3. CHAIN OF THOUGHT", MY_MATH_INSTRUCTION, TEST_3,
              parse=total_line),
    ]
    print(f"\n=== {sum(passed)}/3 challenges beaten ===")


if __name__ == "__main__":
    main()