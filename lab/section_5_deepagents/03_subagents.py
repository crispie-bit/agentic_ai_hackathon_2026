"""
§5 · 03 — Defining sub-agents.  (slide 50)

    uv run section_5_deepagents/03_subagents.py

No class to inherit and no graph to wire. A sub-agent is a DICT:

    name          what the orchestrator calls it
    description   what the orchestrator READS to decide when to delegate
    system_prompt the sub-agent's own instructions
    tools         optional — see the warning below

Two things worth watching in the trace:

  1. `task` calls. That is the orchestrator handing work to a named
     sub-agent. Its internal turns do not appear in this transcript — the
     sub-agent has its OWN context window, which is the point. Delegation
     is a context-management strategy before it is anything else.

  2. The reviewer is given "tools": [] so it can read, think and reply but
     not act. Capability restriction here is config, not a sandbox.

OMITTING `tools` DOES NOT MEAN "NO TOOLS"

    # deepagents/graph.py
    # Inherit parent tools unless the subagent declares its own.
    raw_subagent_tools = spec.get("tools") if "tools" in spec else tools

Leave the key out and the sub-agent inherits EVERY tool the orchestrator has.
Measured on this script: the reviewer re-ran all three `lookup` calls for
itself — 6 calls instead of 3 — because it inherited them. To restrict a
sub-agent you must say so explicitly with "tools": [].

That is a security-relevant default. A "read-only reviewer" that silently
inherits a write tool is not read-only.

The `description` is a routing prompt. Make it vague and the orchestrator
stops delegating — exactly like a vague tool description in §3.
"""

import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from _common import chat_model, model_label

WORKSPACE = Path(__file__).resolve().parent / "workspace_subagents"

FACTS = {
    "state": "LangGraph state is a typed dict every node reads and writes. A "
             "node returns only the keys it changed.",
    "reducer": "A reducer is (old, new) -> merged, attached to a state key, so "
               "concurrent writers combine instead of clobbering.",
    "checkpointer": "A checkpointer persists graph state between invocations, "
                    "keyed by thread_id — memory across turns, and resume "
                    "after a crash.",
}

ORCHESTRATOR = """You are a technical writer producing a short briefing note.

1. Use your to-do tool first to write the plan.
2. Use `lookup` to gather facts. Do not rely on memory.
3. Write the note to `briefing.md` — about 150 words.
4. Ask the `reviewer` sub-agent to critique it.
5. Revise once based on the critique, then reply with a one-line summary.
"""

REVIEWER = """You review short technical briefings for accuracy and clarity.

Read `briefing.md`. Reply with at most three specific, actionable improvements,
or exactly ACCEPT if it is genuinely good. Do NOT modify the file yourself.
"""

TASK = ("Write a briefing note explaining LangGraph state, reducers and "
        "checkpointers to a new engineer.")


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

    # A sub-agent is a plain dict. No class, no graph wiring.
    reviewer = {
        "name": "reviewer",
        # This sentence is a ROUTING PROMPT. The orchestrator reads it to
        # decide whether to delegate. Make it vague ("helps with things") and
        # delegation stops happening.
        "description": ("Critiques a drafted technical briefing for accuracy, "
                        "clarity and missing concepts. Use after writing a draft."),
        "system_prompt": REVIEWER,
        # EXPLICITLY empty. Delete this line and the reviewer inherits every
        # tool the orchestrator has — including, in a real app, the ones that
        # write. Verified: with the key removed, `lookup` runs 6 times instead
        # of 3, because the reviewer repeats the research for itself.
        "tools": [],
    }

    agent = create_deep_agent(
        model=chat_model(),
        tools=[lookup],
        system_prompt=ORCHESTRATOR,
        subagents=[reviewer],
        backend=FilesystemBackend(root_dir=str(WORKSPACE), virtual_mode=True),
    )

    print(f"model: {model_label()}")
    print(f"\nIN\n  {TASK}")
    print("  orchestrator tools: [lookup]        sub-agents: [reviewer]")
    print("\nTRACE")

    result = agent.invoke({"messages": [{"role": "user", "content": TASK}]})

    delegations = 0
    for m in result["messages"]:
        for call in getattr(m, "tool_calls", None) or []:
            args = str(call.get("args", ""))[:66].replace("\n", " ")
            marker = ""
            if call["name"] == "task":
                delegations += 1
                marker = "   <- DELEGATION"
            print(f"  {call['name']:16} {args}{marker}")

    print("\nOUT")
    print(f"  messages:     {len(result['messages'])}")
    print(f"  delegations:  {delegations}")
    for f in sorted(WORKSPACE.rglob("*")):
        if f.is_file():
            print(f"  file:         {f.relative_to(WORKSPACE)} "
                  f"({f.stat().st_size} bytes)")

    print(f"\n  {result['messages'][-1].content.strip()[:300]}")


if __name__ == "__main__":
    main()
