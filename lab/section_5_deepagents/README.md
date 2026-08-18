# §5 — From graph to agent framework  (slides 47–52)

Two scripts. Both demos — nothing here is a hands-on lab.

```bash
cd agentic_teaching/lab
uv sync --extra takehome            # deepagents
uv run section_5_deepagents/01_deepagent.py
uv run section_5_deepagents/02_subagents.py
```

| | File | Slide |
|---|---|---|
| 1 | `01_deepagent.py` | 49 — todo tool, filesystem |
| 2 | `02_subagents.py` | 50 — sub-agents are dicts |

In §4 you wired nodes and edges by hand. `create_deep_agent` hands you a
planner, a filesystem and sub-agent delegation for four lines of setup — and
charges you for all three on every call, whether the question needed them or
not. That trade is the whole section.

## ⚠ Use Bedrock for this section

```
LLM_PROVIDER=bedrock        # in lab/.env
```

DeepAgents attaches a large harness — a planner, a filesystem toolset,
sub-agent machinery — and the Groq free-tier models **cannot reliably emit tool
calls against it**. Measured failure:

```
groq.BadRequestError: tool_use_failed
  failed_generation: '<function=write_file {"content": ...} </function>'
```

Both files need Claude.

## 01 — what DeepAgents adds

Four lines of setup buy a planner, a filesystem and delegation:

```python
agent = create_deep_agent(
    model=chat_model(), tools=[lookup],
    system_prompt="1. Use to-do tool first to plan. ...",
    backend=FilesystemBackend(root_dir="./workspace", virtual_mode=True),
)
```

Watch the trace: the agent writes a **plan** before it writes the file, then
re-reads it to mark the work done.

```
  write_todos      {'todos': [{'content': 'Look up LangGraph state concept'...
  lookup           {'topic': 'state'}
  task             -> delegates to a sub-agent
  write_file       {'file_path': '/briefing.md', ...}
  write_todos      ...status updated to completed...
```

The plan is written down before acting, then re-read. That is the context-rot
mitigation from §2, handed over as a default.

The `task` call is the surprise: **nothing in this script asks for a sub-agent.**
DeepAgents ships delegation as a default tool, so the orchestrator spawns a
nested agent — with its own loop of model calls — whenever it judges the work
worth handing off. File 02 is where that becomes deliberate.

**`virtual_mode=True` is load-bearing.** Without it, `root_dir` only affects
*relative* paths — and the model writes to `/briefing.md`, an absolute path,
which lands on your real filesystem (we caught it writing to `/tmp`). With it,
`/` means the workspace directory and the agent cannot escape it.

**`root_dir="./workspace"` is relative to your shell, not the script.** Run
from `lab/` and the file appears in `lab/workspace/`; run from this folder and
you get a second workspace here. File 02 pins it with
`Path(__file__).parent / "workspace_subagents"` — do that in anything real.

### Why it takes ~90 seconds

Expect a long silence before any output. One `agent.invoke()` is not one API
call — measured on Bedrock (Haiku 4.5):

```
  model calls        5 (sequential, plus a nested sub-agent run)
  input tokens      46,714   (~9k per call — harness prompt + tool schemas)
  output tokens      4,855   (one write_file message was 4,187 of them)
  wall clock          ~90s
  a trivial call       3.3s  <- for comparison
```

Output generation dominates: tokens emerge at roughly 50–90/sec, so ~4.9k of
them is most of the 90 seconds on its own. Nothing in the prompt bounds the
document length, so the model wrote eight sections with code examples.

If the wait is awkward in front of a room:

- **bound the output** — `"Write briefing.md, max 300 words."` is the single
  biggest win
- **say "do not delegate"**, or drop to `create_agent`, to cut the sub-agent hop
- **stream it** — `agent.stream(..., stream_mode="values")` is no faster, but
  the class watches steps arrive instead of a dead terminal

That input number is the point: ~9k tokens of harness ride along on every call,
for a question that needed one fact and one file.

## 02 — sub-agents

A sub-agent is a dict: `name`, `description`, `system_prompt`, and optionally
`tools`. No class to inherit, no graph to wire.

The script runs a writer/reviewer pair — draft, critique, revise once — and
prints the trace with each `task` call marked `<- DELEGATION`.

The `description` is a **routing prompt**. The orchestrator reads it to decide
when to delegate; make it vague and delegation stops.

A sub-agent's internal turns do not appear in the transcript, because it has
its **own context window**. Delegation is a context-management strategy before
it is anything else.

**Omitting `tools` does NOT mean "no tools".** From `deepagents/graph.py`:

```python
# Inherit parent tools unless the subagent declares its own.
raw_subagent_tools = spec.get("tools") if "tools" in spec else tools
```

Leave the key out and the sub-agent inherits **every** tool the orchestrator
has. Measured on this script: the reviewer re-ran all three `lookup` calls for
itself, 6 instead of 3. To restrict it you must pass `"tools": []` explicitly,
as the reviewer here does.

A "read-only reviewer" that silently inherits a write tool is not read-only.
That default is worth saying out loud.

## The through-line

Every rung — the hand-wired graph in §4, `create_agent`, `create_deep_agent` —
expresses the same four decisions: **which model, which tools, what shape the
answer takes, and when to stop.** Start at the highest rung that solves the
problem, and check what the rung costs before it ships.
