"""Few-shot prompting: the same instruction, plus examples."""

import _bootstrap  # noqa: F401

from _common import chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

INSTRUCTION = (
    "Classify the ticket as billing, technical or account. "
    "Reply with one word."
)

# The examples are just fake conversation turns: you write both sides.
# These teach a convention the instruction never states — anything about
# logging in or verifying identity is 'account', not 'technical'.
EXAMPLES = [
    ("I can't log in, it says my password is wrong.", "account"),
    ("The dashboard won't load on Safari.", "technical"),
    ("Refund me for the duplicate charge please.", "billing"),
]

TICKETS = [
    "Two-factor codes never arrive on my phone.",
    "My card was charged twice this month.",
    "The app crashes every time I upload a photo.",
]


def build_messages(ticket):
    messages = [SystemMessage(INSTRUCTION)]
    for example_ticket, label in EXAMPLES:
        messages.append(HumanMessage(example_ticket))
        messages.append(AIMessage(label))
    messages.append(HumanMessage(ticket))
    return messages


def classify(model, ticket):
    reply = model.invoke(build_messages(ticket))
    return reply.content.strip().strip(".").lower()


def main():
    model = chat_model(temperature=0)
    for ticket in TICKETS:
        print(f"{classify(model, ticket):<10} <- {ticket}")


if __name__ == "__main__":
    main()