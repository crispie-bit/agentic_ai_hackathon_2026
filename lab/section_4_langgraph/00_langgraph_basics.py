"""
§4 · 00 — The agent loop as a graph.

    uv run section_4_langgraph/00_langgraph_basics.py

The whole of LangGraph in one file:

    STATE    the shared object passed between steps
    NODES    plain functions: state in, partial update out
    EDGES    which node runs next
    RUNTIME  what compile() gives you

The graph is the §3 ReAct `for` loop with names on it: the model node decides,
a router looks at the last message, and the tool node runs whatever was asked
for and loops back.

    START -> chatbot -> should_continue -> tools -> (back to chatbot)
                              |
                              +-> END

The model comes from chat_model(), so the same file runs on Groq (free tier)
or on Bedrock — the only difference is LLM_PROVIDER in lab/.env.
"""

from typing import Annotated

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

import _bootstrap  # noqa: F401

from _common import banner, chat_model, model_label


# ---------------------------------------------------------------- 1. STATE
class State(TypedDict):
    # add_messages is a reducer: it APPENDS new messages instead of
    # overwriting the list. Drop it and each node clobbers the history.
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------- 2. TOOLS
@tool
def calculate_sum(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b


tools = [calculate_sum]

# The one model line. No provider name in this file — _common decides.
llm = chat_model().bind_tools(tools)


# ---------------------------------------------------------------- 3. NODES
# state in, PARTIAL dict out. Return only what you changed.

def chatbot_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)      # runs every tool_call on the last message


# ---------------------------------------------------------------- 4. EDGES
# A router is a PURE function of state returning a label. No model call.

def should_continue(state: State):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


builder = StateGraph(State)

builder.add_node("chatbot", chatbot_node)       # register each node by name
builder.add_node("tools", tool_node)

builder.add_edge(START, "chatbot")              # where to begin
builder.add_conditional_edges("chatbot", should_continue, ["tools", END])
builder.add_edge("tools", "chatbot")            # loop back after a tool ran

# ---------------------------------------------------------------- 5. RUNTIME
graph = builder.compile()      # <- validates the wiring, returns a runnable


if __name__ == "__main__":
    banner(f"§4 · 00 — the agent loop as a graph   [{model_label()}]")
    print(graph.get_graph().draw_ascii())

    user_input = {"messages": [("user", "What is 42 + 58?")]}
    output = graph.invoke(user_input)

    print()
    for msg in output["messages"]:
        print(f"{msg.type.upper()}: {msg.content}")
