"""
§5 · 02 — What DeepAgents adds.  (slide 49)

    uv run section_5_deepagents/02_deepagent.py

A ReAct agent with three things already attached:

    to-do tool    the model writes a plan before acting, and reads it back
    filesystem    work is passed as FILES, not through the context window
    sub-agents    named specialists to delegate to        (file 03)

Nothing here is new behaviour. It is the planning and memory work from §2,
already wired in — with the prompt that drives it written by someone else.

The to-do tool is the point: forcing a plan to be written down externalises
it, so later steps consult the plan instead of trying to remember it. That is
the context-rot mitigation from §2, handed over as a default.
"""

import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from _common import chat_model, model_label

WORKSPACE = Path(__file__).resolve().parent / "workspace"

FACTS = {
    "state": "LangGraph state is a typed dict every node reads and writes. A "
             "node returns only the keys it changed.",
    "reducer": "A reducer is a function (old, new) -> merged attached to a state "
               "key, so concurrent writers combine instead of clobbering. "
               "add_messages is the built-in one for message history.",
    "checkpointer": "A checkpointer persists graph state between invocations, "
                    "keyed by thread_id — which is what gives an agent memory "
                    "across turns and lets a crashed run resume.",
}

TASK = ("Write a short briefing note explaining LangGraph state, reducers and "
        "checkpointers to a new engineer. Save it as briefing.md.")

INSTRUCTIONS = """You are a technical writer.

1. Use your to-do tool FIRST to write the plan before doing anything else.
2. Use `lookup` for each concept. Do not rely on memory.
3. Write the note to `briefing.md` — about 150 words.
"""


def lookup(topic: str) -> str:
    """Look up a short factual note about a LangGraph concept.

    Args:
        topic: one of state, reducer, checkpointer.
    """
    print(f"  [tool] lookup({topic!r})")
    return FACTS.get(topic.strip().lower(),
                     f"No note on {topic!r}. Known: {', '.join(FACTS)}")


def main() -> None:
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
    except ImportError:
        sys.exit("deepagents is not installed.\n  Run: uv sync --extra takehome")

    shutil.rmtree(WORKSPACE, ignore_errors=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    agent = create_deep_agent(
        model=chat_model(),
        tools=[lookup],
        system_prompt=INSTRUCTIONS,
        # Agents hand work to each other through FILES rather than context.
        #
        # virtual_mode=True is load-bearing. Without it, root_dir only affects
        # RELATIVE paths — and the model writes to "/briefing.md", which is an
        # absolute path, so it lands on your real filesystem root (or /tmp)
        # instead of here. With it, "/" means this directory and nothing the
        # agent does can escape it.
        backend=FilesystemBackend(root_dir=str(WORKSPACE), virtual_mode=True),
    )

    print(f"model: {model_label()}")
    print(f"\nIN\n  {TASK}\n  tools you supplied: [lookup]")
    print("\nTRACE")

    result = agent.invoke({"messages": [{"role": "user", "content": TASK}]})

    for m in result["messages"]:
        for call in getattr(m, "tool_calls", None) or []:
            args = str(call.get("args", ""))[:70].replace("\n", " ")
            print(f"  {call['name']:16} {args}")

    print("\nOUT")
    print(f"  messages:  {len(result['messages'])}")
    files = [f for f in sorted(WORKSPACE.rglob("*")) if f.is_file()]
    for f in files:
        print(f"  file:      {f.relative_to(WORKSPACE)}  ({f.stat().st_size} bytes)")
    if not files:
        print("  file:      (none written — the model skipped step 3)")

    print(f"\n  {result['messages'][-1].content.strip()[:400]}")


if __name__ == "__main__":
    main()
