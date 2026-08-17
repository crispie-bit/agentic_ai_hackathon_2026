"""Chain-of-thought prompting: make the model show its work before answering."""

import re

import _bootstrap  # noqa: F401

from _common import chat_model
from langchain_core.messages import HumanMessage, SystemMessage

DIRECT = "Answer the question. Reply with the number only."

COT = (
    "Answer the question. Work through it one step at a time, showing each "
    "calculation. Then end with a final line in exactly this format:\n"
    "ANSWER: <number>"
)

PROBLEMS = [
    (
        "A customer bought 3 bags of beans at $18 each and 2 mugs at $12 each. "
        "They used a $15 coupon, then a 10% loyalty discount was applied to "
        "the remaining total. Shipping is $6 and is never discounted. "
        "What did they pay in total?",
        62.70,
    ),
    (
        "A support queue had 40 open tickets. On Monday 12 new tickets arrived "
        "and 18 were closed. On Tuesday twice as many arrived as on Monday, "
        "and 9 were closed. How many tickets are open now?",
        49,
    ),
]


def ask(model, instruction, question):
    reply = model.invoke([SystemMessage(instruction), HumanMessage(question)])
    return reply.content.strip()


def final_number(text):
    """Pull the answer out. With CoT the reasoning is text you must skip past."""
    match = re.search(r"ANSWER:\s*\$?([\d,]+(?:\.\d+)?)", text)
    if not match:
        numbers = re.findall(r"\$?([\d,]+(?:\.\d+)?)", text)
        if not numbers:
            return None
        match_value = numbers[-1]
    else:
        match_value = match.group(1)
    return float(match_value.replace(",", ""))


def main():
    model = chat_model(temperature=0)

    for question, expected in PROBLEMS:
        print(f"\nQ: {question}")
        print(f"   expected: {expected}")

        direct = ask(model, DIRECT, question)
        print(f"\n   DIRECT -> {direct[:80]!r}")
        print(f"      parsed: {final_number(direct)}")

        cot = ask(model, COT, question)
        print("\n   CHAIN OF THOUGHT ->")
        for line in cot.splitlines():
            if line.strip():
                print(f"      {line}")
        print(f"      parsed: {final_number(cot)}")


if __name__ == "__main__":
    main()