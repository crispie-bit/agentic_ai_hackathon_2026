"""Zero-shot prompting: an instruction, no examples."""

import _bootstrap  # noqa: F401

from _common import chat_model
from langchain_core.messages import HumanMessage, SystemMessage

INSTRUCTION = (
    "Classify the ticket as billing, technical or account. "
    "Reply with one word."
)

TICKETS = [
    "My card was charged twice this month.",
    "The app crashes every time I upload a photo.",
    "How do I change the email on my profile?",
    "I was billed annually but I picked the monthly plan.",
]


def classify(model, ticket):
    reply = model.invoke([
        SystemMessage(INSTRUCTION),
        HumanMessage(ticket),
    ])
    return reply.content.strip().strip(".").lower()


def main():
    model = chat_model(temperature=0)
    for ticket in TICKETS:
        print(f"{classify(model, ticket):<10} <- {ticket}")


if __name__ == "__main__":
    main()