# §5 — From graph to agent framework  (slides 47–52)

Four scripts. All demos — nothing here is a hands-on lab.

```bash
cd agentic_teaching/lab
uv sync --extra takehome            # deepagents + claude-agent-sdk
```

| | File | Slide |
|---|---|---|
| 1 | `01_abstraction_levels.py` | 48 — the six rungs |
| 2 | `02_deepagent.py` | 49 — todo tool, filesystem |
| 3 | `03_subagents.py` | 50 — sub-agents are dicts |
| 4 | `04_agent_sdk.py` | 51 + 52 — MCP server and options |

## ⚠ Use Bedrock for this section

```
LLM_PROVIDER=bedrock        # in lab/.env
```

DeepAgents attaches a large harness — a planner, a filesystem toolset,
sub-agent machinery — and `llama-3.3-70b` on Groq **cannot reliably emit tool
calls against it**. Measured failure:

```
groq.BadRequestError: tool_use_failed
  failed_generation: '<function=write_file {"content": ...} </function>'
```

Files 02 and 03 need Claude. File 01 runs on either, but is slower on Groq.

## 01 — abstraction levels

The same question, at three rungs. Measured on Bedrock (Haiku 4.5):

```
  rung                  you write                 msgs  input tok    secs
  3  StateGraph         nodes and edges              4       1281     3.9
  4  create_agent       tools + prompt               4       1281     3.1
  5  create_deep_agent  tools + subagent dicts       4      14035     3.6
```

Same answer, same message count, **11× the input tokens** — every call carries
a planner, a filesystem toolset and sub-agent machinery this question never
needed.

Rungs 3 and 4 send *identical* token counts, which is the cleanest statement of
what `create_agent` is: the graph you wrote by hand in §4, typed for you.

Latency is a wash here (3.1s vs 3.6s), so lead with the token number. On Groq
the same script showed 34× and a 53s rung-5 run — the ratio depends on the
model and the harness, so re-measure rather than quoting these figures.

Start at the highest rung that solves the problem — and check what the rung
costs before it ships.

## 02 — what DeepAgents adds

Watch the trace order: `write_todos` comes **first**, before any research.

```
  write_todos      {'todos': [{'content': 'Look up LangGraph state concept'...
  lookup           {'topic': 'state'}
  lookup           {'topic': 'reducer'}
  lookup           {'topic': 'checkpointer'}
  write_todos      ...status updated to completed...
  write_file       {'file_path': '/briefing.md', ...}
```

The plan is written down before acting, then re-read. That is the context-rot
mitigation from §2, handed over as a default.

**`virtual_mode=True` is load-bearing.** Without it, `root_dir` only affects
*relative* paths — and the model writes to `/briefing.md`, an absolute path,
which lands on your real filesystem (we caught it writing to `/tmp`). With it,
`/` means the workspace directory and the agent cannot escape it.

## 03 — sub-agents

A sub-agent is a dict: `name`, `description`, `system_prompt`, and optionally
`tools`. The `description` is a routing prompt — the orchestrator reads it to
decide when to delegate. Make it vague and delegation stops.

**Omitting `tools` does NOT mean "no tools".** From `deepagents/graph.py`:

```python
# Inherit parent tools unless the subagent declares its own.
raw_subagent_tools = spec.get("tools") if "tools" in spec else tools
```

Leave the key out and the sub-agent inherits **every** tool the orchestrator
has. Measured on this script: the reviewer re-ran all three `lookup` calls for
itself, 6 instead of 3. To restrict it you must pass `"tools": []` explicitly.

A "read-only reviewer" that silently inherits a write tool is not read-only.
That default is worth saying out loud.

## 04 — the Claude Agent SDK

**Needs the Claude Code CLI**, because the SDK drives it as a subprocess:

```bash
npm install -g @anthropic-ai/claude-code
```

Without it the script exits with instructions rather than a stack trace. If
you cannot install it, read the source — slides 51 and 52 are about the
*shape* of the configuration, which is all visible without running it.

What to point at:

- `@tool(name, description, schema)` — all three **explicit**, unlike
  LangChain which infers them from the function
- tools are **async** and return **content blocks**, not strings
- `is_error=True` reports failure to the model without ending the run
- `create_sdk_mcp_server` runs **in-process** — no subprocess, no network —
  yet speaks the same protocol as an external MCP server
- `allowed_tools` is an **allow-list and a security boundary**, matched on
  exact names (`mcp__<server>__<tool>`). Get the string wrong and the tool
  silently does not exist.
- `CLAUDE_CODE_USE_BEDROCK=1` routes the whole SDK through Bedrock

## The through-line

Every rung expresses the same four decisions: **which model, which tools, what
shape the answer takes, and when to stop.** `max_turns` here is the same step
cap you wrote by hand in §2, under a different name.
