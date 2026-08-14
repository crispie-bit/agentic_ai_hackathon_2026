# §4 — LangGraph  (slides 44–47)

Two files. **You write the first one; the second you watch.**

```bash
cd agentic_teaching/lab
uv run section_4_langgraph/01_graph_lab.py       # the lab — 15 min
uv run section_4_langgraph/02_tool_writes_state.py   # the demo — watch
```

| | File | Slides | Format |
|---|---|---|---|
| 1 | `01_graph_lab.py` | 44, 45, 46 | **hands-on**, 3 TODOs |
| 2 | `02_tool_writes_state.py` | 47 | **explained**, no TODOs |

## What changes from §3

Nothing about the model call. §3 ended with an agent that was a `for` loop;
LangGraph gives that loop a runtime — named steps, explicit routing, one state
object between them.

```
START -> generate -> should_continue -> evaluate -> (back to generate)
                          |
                          +-> END
```

Four parts, and that is the library:

| | |
|---|---|
| **State** | the shared object every node reads and writes |
| **Nodes** | one unit of work: a model call, a tool, a check |
| **Edges** | which node runs next |
| **Runtime** | checkpointing, streaming, retries, replay — from `compile()` |

## The lab (01)

A support-ticket reply that drafts, gets critiqued, and redrafts — bounded.

- **TODO 1** register the second node *and* its edge. Forgetting the edge is
  the classic mistake: the node exists and never runs.
- **TODO 2** add the loop bound. A back-edge runs forever on its own.
- **TODO 3** delete `add_messages` and watch history get clobbered.

Note what happens before you fix TODO 1: it fails at **compile time**, naming
the node it cannot find. A graph validates its own wiring before it costs you
a single token — which is one of the honest arguments for using one.

The router is ordinary Python with no model call in it, so your control flow
is deterministic and testable without an API key.

## The demo (02)

A tool that receives the graph state by injection and returns a state update
instead of a string. Three annotations do it:

```python
state:        Annotated[dict, InjectedState]
tool_call_id: Annotated[str, InjectedToolCallId]
-> Command(update={...})
```

The point: the ticket text lives in state and **never enters the prompt**. The
tool reaches into state, computes, and writes structured fields back. Swap the
ticket for a 40MB parsed PDF and nothing in the file changes.

Injected parameters are hidden from the model — not in the schema it sees — so
it cannot fill them in, cannot get them wrong, and cannot be talked into
overriding them.

**The rule**, and it bites everyone once: a tool returning `Command(update=...)`
must both write its keys **and** emit a `ToolMessage` carrying the
`tool_call_id`. Returning a plain string does the second for you; the moment
you return a `Command`, you own it.

## Same thing on Groq

Both files call `chat_model()`, so they run on either provider. Comment out
`LLM_PROVIDER=bedrock` in `lab/.env` and they run on Groq's free tier
unchanged.


## LangChain 1.x, not 0.3.x

Some names moved. If an import breaks, check whether the symbol *moved* before
assuming it was removed:

| 0.3.x | 1.x |
|---|---|
| `langgraph.prebuilt.create_react_agent` | `langchain.agents.create_agent` |
| `create_react_agent(prompt=...)` | `create_agent(system_prompt=...)` |
| `langgraph.prebuilt.InjectedToolCallId` | `langchain_core.tools.InjectedToolCallId` |
| `langgraph.prebuilt.InjectedState` | unchanged |
