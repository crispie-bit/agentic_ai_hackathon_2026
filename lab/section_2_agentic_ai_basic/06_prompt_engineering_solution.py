"""
§2 · 06 — PROMPT LAB, WORKED.

    uv run section_2_agentic_ai_basic/06_prompt_engineering_solution.py

Read this AFTER you have had a real go at the lab.

This is not an answer key, and copying the three strings out of it will teach
you nothing. Challenge 1 is deliberately shown FAILING first, then fixed,
because the thing worth learning here is the revision loop — write, score,
read which row failed, change one clause — and not the string it happens to
end at. There are dozens of instructions that score 8/8. There is one method
for finding them.

Running this makes 32 model calls and prints all five scores.
"""

import re
import time

import _bootstrap  # noqa: F401

from _common import banner, chat_model, model_label
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ##########################################################################
#                    TODO 1 — ZERO-SHOT, WORKED
# ##########################################################################

# FIRST ATTEMPT. This is what almost everyone writes, and it is not a bad
# prompt: it names all three labels, which is the single most important thing
# a zero-shot instruction can do. The model cannot pick a category it has
# never been told exists.
#
# It scores 6/8 or 7/8, depending on which model you are pointed at.

INSTRUCTION_V1 = ("Route this ticket to one team: billing, account or "
                  "technical. Reply with one word, lowercase.")

# WHAT IT MISSES — and it is the same mistake twice:
#
#   FAIL  got=billing  want=account     I can't afford this anymore, close my
#                                       subscription.
#   FAIL  got=billing  want=technical   The invoice PDF won't download.
#
# Whether you see one of these or both depends on the model. The cause is
# identical either way, which is the part worth noticing.
#
# Read them the way the model did. The loudest word in the first is "afford";
# in the second it is "invoice". Both are money words, so `billing` is a
# perfectly sensible reading of the TOPIC. But neither customer is disputing an
# amount. One wants a subscription closed. The other has a download that does
# not work.
#
# V1 named three teams and never said what any of them DOES. So when a ticket
# mentions money but asks for something else, the model has nothing to decide
# on except the vocabulary — and the vocabulary says billing.
#
# The fix is not "be more specific": V1 is already specific about the label
# set. It is to key the decision on the ACTION, and then say out loud that the
# action beats the noun.

INSTRUCTION_V2 = ("Route this support ticket to one team: billing, account "
                  "or technical. Classify by the action needed, not the topic "
                  "mentioned. If something is broken or will not load it is "
                  "technical, even if it is an invoice. If an amount is wrong "
                  "it is billing. If it changes who may use the subscription "
                  "it is account. Reply with one word, lowercase.")

# 8/8, on both providers. Note what did NOT change: the same three labels, the
# same output format, the same tone. Almost all of V1 was right. Prompt
# engineering is usually this — finding the one boundary the instruction left
# undefined — and not rewriting from scratch.


# ##########################################################################
#                    TODO 2 — FEW-SHOT, WORKED
# ##########################################################################

# The instruction is locked, so there is exactly one lever: which 5 of the 7
# training rows the model sees. Before choosing, look at what the locked
# instruction manages ON ITS OWN — 5/6. Five of these six tickets never needed
# examples at all.
#
# So the whole challenge is this one row:
#
#   FAIL  got=billing  want=security   I see a payment I never made on my
#                                      statement.
#
# The instruction says "billing, security or engineering" and stops. A payment
# on a statement is, on its face, billing. Nothing in the instruction says that
# a payment you did not authorise is a break-in rather than a bookkeeping
# error, and no amount of rewording would say it — the instruction is locked.
#
# One training row draws that line explicitly:
#
#     ("There is a charge from a device I don't recognise.", "security")
#
# It is the only row where money and security collide, and it resolves the
# collision in favour of security. It is the row to reach for first.
#
# How much it matters depends on the model, and that is worth knowing rather
# than hiding. Every legal pick was tested against the full TEST_2:
#
#     Groq / gpt-oss     20 of the 21 possible picks score 6/6. Almost
#                        anything works, including picks without that row.
#     Bedrock / Haiku    13 of 21. Every single passing pick contains that
#                        row, and no pick without it ever passes.
#
# So on the workshop model you will probably pass on your first guess. Do not
# take that as proof your reasoning was right — it is a real result about this
# model, not a general law. The lesson that DOES hold on both is the one the
# baseline shows you: the instruction alone cannot reach that ticket, and
# examples can.
#
# The four rows below the key one keep all three labels represented, so the
# model is not nudged into over-using any of them.

EXAMPLES = [
    ("My invoice has the wrong VAT number.", "billing"),
    ("I was billed in the wrong currency.", "billing"),
    ("There is a charge from a device I don't recognise.", "security"),
    ("The export button does nothing.", "engineering"),
    ("The dashboard is blank on Safari.", "engineering"),
]

# 6/6. Worth being precise about what happened: the examples did not make the
# model "better at classifying". Five of the six answers were already right and
# stayed right. One boundary moved, and it was the boundary the instruction had
# no words for. That is what examples are for.


# ##########################################################################
#                    TODO 3 — CHAIN OF THOUGHT, WORKED
# ##########################################################################

# The starter prompt, "Reply with the final number only", scores 0/4 — and the
# reason is not the one most people expect.
#
# It is NOT that the model cannot do the arithmetic. Check for yourself: all
# four totals come back correct, on both providers.
#
# What each model does with the FORMAT instruction is where it gets
# interesting, because they do opposite things and both score zero:
#
#   Groq / gpt-oss    obeys perfectly. Replies "62.70". Nothing else.
#   Bedrock / Haiku   ignores it completely, and prints a page of markdown
#                     working with the right total at the bottom.
#
# Neither produces a line the grader can read, so both print got=None on every
# row. Four right answers, zero points — twice, for opposite reasons.
#
# Sit with that, because it is the most useful thing in this file: an answer
# your code cannot parse is not an answer. Note especially that the obedient
# model failed too. "Reply with the final number only" was honoured to the
# letter and still scored 0/4, because the format it was told to use was not
# the format the caller actually needed.
#
# The fix is not about making the model think harder — it is already doing
# that. It is about naming a shape you can rely on: let it reason as much as it
# likes, then pin down the last line and forbid anything after it. That
# contract survives the model changing underneath you. "Only" does not.

MATH_INSTRUCTION = ("Solve the problem. Work through it one step at a time, "
                    "showing each calculation, applying the steps in the "
                    "order the problem states them. Then end your reply with "
                    "a final line in exactly this format:\n"
                    "TOTAL: <number>\n"
                    "Write nothing after that line.")

# 4/4. "Write nothing after that line" is doing real work: without it the model
# likes to add a friendly sentence afterwards, and a grader anchored to the
# last line would read that instead of the number.


# ##########################################################################

LOCKED_INSTRUCTION = ("Route this ticket to one team: billing, security or "
                      "engineering. Reply with one word, lowercase.")

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
    found = re.findall(r"(?mi)^[^A-Za-z0-9]*TOTAL:\s*\$?([\d,]+(?:\.\d+)?)",
                       text)
    return round(float(found[-1].replace(",", "")), 2) if found else None


def build(instruction, ticket, examples=()):
    messages = [SystemMessage(instruction)]
    for text, label in examples:
        messages += [HumanMessage(text), AIMessage(label)]
    return messages + [HumanMessage(ticket)]


def ask(model, messages, attempts=3):
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
    return hits


def main():
    print(f"model: {model_label()}")
    model = chat_model(temperature=0)

    banner("TODO 1 — the revision loop, both halves")
    v1 = score(model, "1a. first attempt: three labels, no definitions",
               INSTRUCTION_V1, TEST_1)
    v2 = score(model, "1b. same prompt + one clause on what each team does",
               INSTRUCTION_V2, TEST_1)
    print(f"\n  {v1}/8 -> {v2}/8 once the instruction says what each team "
          f"does,\n  and that the action beats the noun.")

    banner("TODO 2 — what the examples actually bought you")
    base = score(model, "2a. locked instruction, no examples",
                 LOCKED_INSTRUCTION, TEST_2)
    few = score(model, "2b. locked instruction, 5 chosen examples",
                LOCKED_INSTRUCTION, TEST_2, EXAMPLES)
    print(f"\n  {base}/6 -> {few}/6. The instruction could not be changed, so "
          f"every point of that\n  difference came from the examples.")

    banner("TODO 3 — four right answers, zero points")
    cot = score(model, "3. chain of thought, forced final line",
                MATH_INSTRUCTION, TEST_3, parse=total_line)
    print(f"\n  {cot}/4. The starter prompt gets all four totals RIGHT and "
          f"still scores 0/4,\n  because not one of them lands on a line the "
          f"grader can read.")


if __name__ == "__main__":
    main()
