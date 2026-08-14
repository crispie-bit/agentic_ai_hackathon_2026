"""
§2 · 01 — Memory: short term and long term.  (slide 22)

    uv run section_2_agentic_ai_basic/01_memory.py

The first file with a real model call. Which provider it hits is decided by
lab/.env — nothing in this file names one:

    # lab/.env
    # LLM_PROVIDER=bedrock     <- commented out: runs on Groq's free tier
    LLM_PROVIDER=bedrock       <- uncommented: same file, same output, Bedrock

Slide 22, in one line: the model retains nothing between calls.

    SHORT TERM   the messages list, resent IN FULL on every request
    LONG TERM    a store outside the model, searched, results pasted into
                 that same list

Long-term memory is not memory the model has. It is a lookup you run, whose
output becomes short-term memory like everything else.
"""

import _bootstrap  # noqa: F401

from langchain_core.messages import HumanMessage, SystemMessage

from _common import banner, chat_model, model_label, report_usage

model = chat_model(temperature=0)


# ==========================================================================
# PART A — no memory. Each call is a fresh universe.
# ==========================================================================

def part_a_stateless() -> None:
    banner("A · NO MEMORY — each call starts from nothing")

    first = model.invoke([HumanMessage("My warehouse is in Penang. Reply in one sentence.")])
    print(f"  turn 1 -> {first.content.strip()}")

    # The obvious mistake: ask a follow-up without sending the history.
    second = model.invoke([HumanMessage("Which warehouse did I just name?")])
    print(f"  turn 2 -> {second.content.strip()}")


# ==========================================================================
# PART B — short-term memory. You resend the whole transcript.
# ==========================================================================

def part_b_short_term() -> list:
    banner("B · SHORT TERM — the transcript IS the memory")

    messages = [SystemMessage("You are terse. One sentence, no preamble.")]

    for question in ["My warehouse is in Penang.",
                     "Which warehouse did I just name?",
                     "And what country is that in?"]:
        messages.append(HumanMessage(question))

        reply = model.invoke(messages)             # <- the WHOLE list, every time
        messages.append(reply)                     # <- append, never replace

        usage = reply.usage_metadata or {}
        print(f"\n  sending {len(messages) - 1} messages")
        print(f"  ask  -> {question}")
        print(f"  got  -> {reply.content.strip()}")
        report_usage(f"turn {len(messages) // 2}", usage)

    return messages


# ==========================================================================
# PART C — long-term memory. A store, a search, an insert.
# ==========================================================================

# A "vector store" in four lines. Real ones embed the text and rank by cosine
# similarity; this ranks by word overlap. The retrieval STEP is the lesson —
# swap this dict for Pinecone or OpenSearch and the loop below is unchanged.
STORE = [
    "Penang warehouse (WH-PEN) closes for stocktake on the last Friday of each month.",
    "SKU-77 is sourced from a single supplier in Ipoh; lead time is 14 days.",
    "Kuala Lumpur warehouse (WH-KUL) handles all export paperwork.",
    "Purchase orders above RM 50,000 require a second approver.",
    "The cafeteria serves nasi lemak on Tuesdays.",
]


def search(query: str, k: int = 2) -> list[str]:
    """Rank stored entries by shared words. Stands in for an embedding search."""
    words = set(query.lower().replace("?", "").split())
    scored = [(len(words & set(e.lower().split())), e) for e in STORE]
    return [text for score, text in sorted(scored, reverse=True)[:k] if score]


def part_c_long_term() -> None:
    banner("C · LONG TERM — a store outside the model")

    question = "When can I not ship from the Penang warehouse?"

    # Without retrieval the model has no way to know. It is not in the weights.
    blind = model.invoke([HumanMessage(question + " One sentence.")])
    print(f"  without retrieval -> {blind.content.strip()}")

    # 1. search  2. insert what came back  3. ask normally
    hits = search(question)
    print("\n  search returned:")
    for h in hits:
        print(f"    - {h}")

    messages = [
        SystemMessage("Answer from the notes below. One sentence.\n\n" + "\n".join(hits)),
        HumanMessage(question),
    ]
    informed = model.invoke(messages)
    print(f"\n  with retrieval    -> {informed.content.strip()}")


if __name__ == "__main__":
    print(f"model: {model_label()}")

    part_a_stateless()
    transcript = part_b_short_term()
    part_c_long_term()

    banner("TRANSCRIPT")
    print(f"  {len(transcript)} messages carried forward from part B")
