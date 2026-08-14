"""
§5 · 01 — Abstraction levels.  (slide 48, "the ladder")

    uv run section_5_deepagents/01_abstraction_levels.py

The same question, the same tool, three rungs. Not competing frameworks —
different levels of abstraction over the same model call.

    rung 3   StateGraph          you wire nodes and edges          ~35 lines
    rung 4   create_agent        you supply tools and a prompt      3 lines
    rung 5   create_deep_agent   you supply tools and dicts         3 lines

WHAT TO READ IN THE TABLE

  All three give the same answer in the same number of messages. The cost is
  in the last two columns: rung 5 carries a planner, a filesystem toolset and
  sub-agent machinery on every call, for a question that needed none of it.

  Higher rungs write more of your code and show less of what is happening, so
  when one misbehaves you are debugging somebody else's prompt. Start at the
  highest rung that solves the problem; drop down when the framework's
  behaviour becomes the thing you need to see. Then check what the rung costs
  before it ships.
"""

import time
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import _bootstrap  # noqa: F401

from _common import banner, chat_model, model_label

QUESTION = "How many units of SKU-77 are in stock, and is that below the reorder level of 50?"

STOCK = {"SKU-77": 12, "SKU-14": 240}


@tool
def check_stock(sku: str) -> str:
    """Get the units currently in stock for one SKU."""
    print(f"      [tool] check_stock({sku!r})")
    return f"{sku}: {STOCK.get(sku.upper(), 0)} units in stock."


def total_input_tokens(messages) -> int:
    """Sum input tokens across every AI message in a run."""
    return sum((getattr(m, "usage_metadata", None) or {}).get("input_tokens", 0)
               for m in messages)


TOOLS = [check_stock]
SYSTEM = "You answer stock questions using your tools. One sentence."


# ==========================================================================
# RUNG 3 — StateGraph. You wire the nodes and the edges yourself.
# ==========================================================================

def rung_3_stategraph() -> tuple[str, int, int]:
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    model = chat_model(temperature=0).bind_tools(TOOLS)
    by_name = {t.name: t for t in TOOLS}

    def call_model(state: State) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    def call_tools(state: State) -> dict:
        out = []
        for call in state["messages"][-1].tool_calls:
            result = by_name[call["name"]].invoke(call["args"])
            out.append(ToolMessage(result, tool_call_id=call["id"]))
        return {"messages": out}

    def should_continue(state: State) -> str:
        return "tools" if state["messages"][-1].tool_calls else "end"

    builder = StateGraph(State)
    builder.add_node("model", call_model)
    builder.add_node("tools", call_tools)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", should_continue,
                                  {"tools": "tools", "end": END})
    builder.add_edge("tools", "model")
    graph = builder.compile()

    final = graph.invoke({"messages": [HumanMessage(SYSTEM + "\n\n" + QUESTION)]})
    msgs = final["messages"]
    return msgs[-1].content, len(msgs), total_input_tokens(msgs)


# ==========================================================================
# RUNG 4 — create_agent. The same graph, prebuilt.
# ==========================================================================

def rung_4_create_agent() -> tuple[str, int, int]:
    agent = create_agent(model=chat_model(temperature=0), tools=TOOLS,
                         system_prompt=SYSTEM)
    final = agent.invoke({"messages": [HumanMessage(QUESTION)]})
    msgs = final["messages"]
    return msgs[-1].content, len(msgs), total_input_tokens(msgs)


# ==========================================================================
# RUNG 5 — create_deep_agent. Adds planning, a filesystem and sub-agents,
# whether or not this task needs them.
# ==========================================================================

def rung_5_deep_agent() -> tuple[str, int, int]:
    from deepagents import create_deep_agent

    agent = create_deep_agent(model=chat_model(temperature=0), tools=TOOLS,
                              system_prompt=SYSTEM)
    final = agent.invoke({"messages": [HumanMessage(QUESTION)]})
    msgs = final["messages"]
    return msgs[-1].content, len(msgs), total_input_tokens(msgs)


# (rung, what YOU hand-write, what the rung hands YOU, runner)
RUNGS = [
    ("3  StateGraph", "state, nodes, edges, router",
     "checkpoints, streaming, replay", rung_3_stategraph),
    ("4  create_agent", "tools, prompt",
     "the ReAct loop, tool dispatch", rung_4_create_agent),
    ("5  create_deep_agent", "tools, prompt, subagent dicts",
     "planning, filesystem, sub-agents", rung_5_deep_agent),
]


def main() -> None:
    print(f"model: {model_label()}")
    print(f"question: {QUESTION}")

    results = []
    for label, you_write, you_get, fn in RUNGS:
        banner(f"RUNG {label}")
        started = time.time()
        answer, messages, tokens = fn()
        elapsed = time.time() - started
        print(f"  -> {answer.strip()[:150]}")
        results.append((label, you_write, you_get, messages, tokens, elapsed))

    banner("THE LADDER")
    print(f"  {'rung':21}{'you hand-write':31}{'you get free':34}"
          f"{'input tok':>10}{'secs':>7}")
    print("  " + "-" * 101)
    for label, you_write, you_get, messages, tokens, elapsed in results:
        print(f"  {label:21}{you_write:31}{you_get:34}{tokens:>10}{elapsed:>7.1f}")

    msgs = {r[3] for r in results}
    base = results[1][4] or 1
    print(f"\n  same answer from all three, {msgs.pop() if len(msgs) == 1 else '?'} "
          f"messages each.")
    print(f"  rung 5 sends {results[2][4] / base:.0f}x the input tokens of rung 4.")


if __name__ == "__main__":
    main()
