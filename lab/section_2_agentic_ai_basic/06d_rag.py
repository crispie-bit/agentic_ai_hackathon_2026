"""
LAB — RAG: giving the model knowledge it does not have.

    uv run section_1_foundation/02_rag.py

Needs GROQ_API_KEY, same as 01_foundations.py.

Lab 1 showed that the model remembers nothing between calls, and that the only
memory is the list of messages you send. RAG is the same trick applied to
knowledge instead of conversation: you look up relevant text yourself, and put
it in the message list before you ask.

Retrieval is written from scratch here (no vector DB, no embedding model) so
nothing is hidden. Only the generation step is a real API call.

Two TODOs, marked in the code. Parts A to D run in order.
"""

import json
import math
import re
from collections import Counter

try:
    import _bootstrap  # noqa: F401

    from _common import GROQ_MODEL, banner, groq_client, report_usage
except ImportError:
    # Fallback so this file also runs outside the course repo.
    import os

    from groq import Groq

    GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    def groq_client():
        if not os.environ.get("GROQ_API_KEY"):
            raise SystemExit("GROQ_API_KEY is not set. Run 00_check_groq.py first.")
        return Groq(api_key=os.environ["GROQ_API_KEY"])

    def banner(text):
        print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")

    def report_usage(label, usage):
        print(f"  [{label}] in={usage.prompt_tokens}  "
              f"out={usage.completion_tokens}  total={usage.total_tokens}")


MAX_TOKENS = 300
TEMPERATURE = 0.0     # RAG answers should be reproducible, not creative
TOP_K = 3             # TODO 1 changes this
MIN_SCORE = 0.05      # TODO 2 changes this


def ask(client, messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE):
    """Identical to lab 1. RAG changes what goes IN the messages, nothing else."""
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    choice = resp.choices[0]
    return choice.message.content, choice.finish_reason, resp.usage


# --------------------------------------------------------------------------
# The external knowledge source.
# Made up on purpose: the model cannot possibly have seen it in training.
# --------------------------------------------------------------------------

DOCUMENTS = [
    """Nimbus Coffee was founded in 2019 by Mara Ellison in Portland, Oregon.
    The company started as a single cart outside the Hawthorne library. Today
    Nimbus operates eleven cafes across Oregon and Washington. The head office
    is still located above the original Hawthorne shop.""",

    """Our return policy allows unopened bags of coffee beans to be returned
    within 30 days of purchase for a full refund. Opened bags can be exchanged
    within 14 days if you are unhappy with the roast. Brewing equipment carries
    a one year warranty against manufacturing defects. Gift cards cannot be
    refunded or exchanged for cash.""",

    """Nimbus roasts every batch in small drums that hold twelve kilograms of
    green coffee. Light roasts are pulled at approximately 196 degrees Celsius,
    while dark roasts continue to 224 degrees. All beans rest for three days
    after roasting before they are bagged and shipped to stores.""",

    """The Cloud Club loyalty program gives members one point for every dollar
    spent. One hundred points can be redeemed for a free drink of any size.
    Members also receive a free pastry during their birthday month. There is no
    fee to join the Cloud Club and points never expire.""",

    """Most Nimbus cafes open at 6:30 in the morning on weekdays and close at
    7 in the evening. Weekend hours are shorter, running from 7:30 to 5. Free
    wifi is available in every location, and the network password is printed
    on the bottom of each receipt.""",
]


# --------------------------------------------------------------------------
# Retrieval. No API calls in this section — it is all local text matching.
# --------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "its", "of", "on", "or", "the", "to",
    "was", "what", "when", "where", "which", "who", "with", "you", "your", "my",
    "me", "that", "this", "there", "they", "we", "us", "our", "if", "so", "much",
}


def split_into_sentences(text):
    text = " ".join(text.split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_document(text, sentences_per_chunk=2):
    """Cut a document into pieces small enough to be worth retrieving alone."""
    sentences = split_into_sentences(text)
    return [" ".join(sentences[i:i + sentences_per_chunk])
            for i in range(0, len(sentences), sentences_per_chunk)]


def embed(text):
    """Stand-in for an embedding model: a normalised word-count vector.

    A real system calls an embedding API and gets back ~1000 floats that encode
    meaning, so "reset my password" matches "recover your login". This version
    only matches shared words. Swap this one function and the rest still works.
    """
    counts = Counter(w for w in re.findall(r"[a-z0-9]+", text.lower())
                     if w not in STOPWORDS)
    length = math.sqrt(sum(v * v for v in counts.values()))
    return {w: c / length for w, c in counts.items()} if length else {}


def cosine_similarity(a, b):
    small, large = (a, b) if len(a) < len(b) else (b, a)
    return sum(w * large.get(k, 0.0) for k, w in small.items())


def build_index(documents):
    """The 'vector database': every chunk stored beside its vector."""
    return [{"doc_id": i, "text": chunk, "vector": embed(chunk)}
            for i, doc in enumerate(documents)
            for chunk in chunk_document(doc)]


def retrieve(question, index, top_k=TOP_K, min_score=MIN_SCORE):
    qv = embed(question)
    scored = sorted(((cosine_similarity(qv, e["vector"]), e) for e in index),
                    key=lambda pair: pair[0], reverse=True)
    return [(s, e) for s, e in scored[:top_k] if s >= min_score]


def build_messages(question, retrieved):
    """The entire trick: retrieved text becomes part of the message list."""
    if not retrieved:
        context = "(no relevant sources found)"
    else:
        context = "\n\n".join(f"[Source {i + 1}] {e['text']}"
                              for i, (_s, e) in enumerate(retrieved))
    return [
        {"role": "system",
         "content": "Answer using only the sources provided by the user. "
                    "If the sources do not contain the answer, say you don't "
                    "know. Cite the source number you used. Be terse."},
        {"role": "user",
         "content": f"SOURCES:\n{context}\n\nQUESTION: {question}"},
    ]


# --------------------------------------------------------------------------
# A. Retrieval on its own. No model involved, no tokens spent.
# --------------------------------------------------------------------------

def part_a_retrieval(index) -> None:
    banner("A. retrieval is just text matching")

    question = "How many points do I need for a free drink?"
    print(f"  question: {question!r}\n")
    for score, entry in retrieve(question, index):
        preview = entry["text"][:66] + ("..." if len(entry["text"]) > 66 else "")
        print(f"  score {score:.3f}  doc {entry['doc_id']}  {preview}")

    print("\n  Nothing above touched the API. Retrieval is local, cheap, and\n"
          "  the part you control. The model only sees what this step selects.")


# --------------------------------------------------------------------------
# B. The same question, with and without the retrieved context.
# --------------------------------------------------------------------------

def part_b_with_and_without(client, index) -> None:
    banner("B. same question, context vs no context")

    question = "What is the Cloud Club and how many points is a free drink?"

    blind = [{"role": "user", "content": question}]
    text, _, usage_blind = ask(client, blind)
    print(f"  WITHOUT context -> {text.strip()[:220]!r}")
    report_usage("no context", usage_blind)

    retrieved = retrieve(question, index)
    text, _, usage_rag = ask(client, build_messages(question, retrieved))
    print(f"\n  WITH context    -> {text.strip()[:220]!r}")
    report_usage("with context", usage_rag)

    extra = usage_rag.prompt_tokens - usage_blind.prompt_tokens
    print(f"\n  RAG cost {extra} extra INPUT tokens for {len(retrieved)} chunks.")
    print("  That is the whole trade: you pay input tokens to buy accuracy.\n"
          "  Retrieving more chunks costs more and can bury the useful one.")

    # TODO 1 -------------------------------------------------------------
    #   Set TOP_K at the top of this file to 8 and re-run part B. Watch the
    #   input token count rise. Then check whether the answer actually got
    #   better. Usually it does not.
    # ---------------------------------------------------------------------


# --------------------------------------------------------------------------
# C. The full loop.
# --------------------------------------------------------------------------

def part_c_full_loop(client, index) -> None:
    banner("C. the full RAG loop")

    questions = [
        "Can I return a bag of beans after I opened it?",
        "What temperature are dark roasts pulled at?",
        "Who founded the company, and where?",
    ]

    for question in questions:
        retrieved = retrieve(question, index)
        messages = build_messages(question, retrieved)
        text, finish, usage = ask(client, messages)

        print(f"\n  Q: {question}")
        print(f"     sources used: {[e['doc_id'] for _s, e in retrieved]}")
        print(f"     A: {text.strip()}")
        if finish == "length":
            print("     (truncated — raise MAX_TOKENS, the model did fine)")
        report_usage("rag", usage)


# --------------------------------------------------------------------------
# D. The failure mode worth knowing about.
# --------------------------------------------------------------------------

def part_d_failure(client, index) -> None:
    banner("D. when retrieval finds nothing useful")

    question = "Does Nimbus Coffee sell tea?"
    retrieved = retrieve(question, index)

    print(f"  question: {question!r}")
    print("  tea is never mentioned in any document, yet retrieval returns:")
    for score, entry in retrieved:
        print(f"    score {score:.3f}  doc {entry['doc_id']}  "
              f"{entry['text'][:56]}...")

    text, _, usage = ask(client, build_messages(question, retrieved))
    print(f"\n  A: {text.strip()}")
    report_usage("part D", usage)

    print("\n  Search always returns a ranked list — something is always 'best',\n"
          "  even when nothing is relevant. Two defences: a score floor, and a\n"
          "  system prompt that permits 'I don't know'. Both are in this file.")

    # TODO 2 -------------------------------------------------------------
    #   Set MIN_SCORE at the top of this file to 0.9 and re-run. Retrieval
    #   now returns nothing at all, and build_messages sends the model an
    #   empty source list. Confirm it still refuses rather than inventing an
    #   answer — that is the system prompt doing the work, not retrieval.
    # ---------------------------------------------------------------------


def main() -> None:
    client = groq_client()
    index = build_index(DOCUMENTS)
    print(f"model: {GROQ_MODEL}")
    print(f"indexed {len(index)} chunks from {len(DOCUMENTS)} documents")

    part_a_retrieval(index)
    part_b_with_and_without(client, index)
    part_c_full_loop(client, index)
    part_d_failure(client, index)

    banner("what to take away")
    print(json.dumps({
        "memory (lab 1)": "re-send the conversation in messages",
        "knowledge (this lab)": "re-send the documents in messages",
        "the API": "unchanged — still stateless, still messages in, text out",
    }, indent=2))


if __name__ == "__main__":
    main()