"""
The agent that runs IN the cloud. This is the file AgentCore deploys.

It is lab 03's agent — a LangGraph ReAct agent whose tool writes into graph state
via InjectedState — with **four lines added** to make it deployable:

    from bedrock_agentcore.runtime import BedrockAgentCoreApp   # 1. import
    app = BedrockAgentCoreApp()                                 # 2. initialise
    @app.entrypoint                                             # 3. decorate
    app.run()                                                   # 4. serve

That is the entire delta between "an agent on my laptop" and "an agent with an
HTTPS endpoint, session isolation, and CloudWatch traces". Everything else in
this file is the agent you already wrote.

This directory is self-contained on purpose: AgentCore uploads the whole source
path, so the agent must not import from the lab's other files.

Run it locally first — `agentcore deploy --local` or `python agent.py` — then
deploy. Debugging in the cloud is slow.
"""

import os
from typing import Annotated, TypedDict

import boto3
from botocore.config import Config
from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from bedrock_agentcore.runtime import BedrockAgentCoreApp

MODEL_ID = os.environ.get(
    "BEDROCK_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
)
REGION = os.environ.get("AWS_REGION") or os.environ.get(
    "AWS_DEFAULT_REGION", "ap-southeast-1"
)

# 2. Initialise. Do this at import time, not inside a function.
app = BedrockAgentCoreApp()


def _model() -> ChatBedrockConverse:
    """Same tuned-client pattern as the rest of the labs.

    Note there are no credentials here. In the cloud the runtime assumes its
    execution role, so boto3's default chain finds them automatically — which is
    exactly why the deploy script has to give that role bedrock:InvokeModel.
    """
    client = boto3.client(
        "bedrock-runtime",
        config=Config(region_name=REGION, read_timeout=300, connect_timeout=120,
                      retries={"max_attempts": 1}),
    )
    return ChatBedrockConverse(model_id=MODEL_ID, client=client, temperature=0)


class DocState(TypedDict):
    messages: Annotated[list, add_messages]
    remaining_steps: int
    doc_text: str          # seeded per request; never sent to the model
    word_count: int        # written by count_words
    top_words: list        # written by count_words


@tool
def count_words(
    min_length: int,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Count words in the loaded document and report the most frequent ones.

    Args:
        min_length: ignore words shorter than this many characters.

    The document is already in state — you do not pass it.
    """
    text = state.get("doc_text", "")
    if not text:
        return Command(update={"messages": [ToolMessage(
            "ERROR: no document is loaded in state.", tool_call_id=tool_call_id)]})

    words = [w.strip(".,;:()").lower() for w in text.split()]
    words = [w for w in words if len(w) >= min_length]
    freq: dict = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

    return Command(update={
        "word_count": len(words),
        "top_words": [w for w, _ in top],
        "messages": [ToolMessage(
            f"Counted {len(words)} words. Most frequent: "
            + ", ".join(f"{w} ({c})" for w, c in top),
            tool_call_id=tool_call_id,
        )],
    })


SYSTEM = ("You analyse documents using your tools. The document is already "
          "loaded — call count_words with a sensible min_length, then summarise "
          "what you found in two sentences.")


# 3. Decorate. The decorated function IS the HTTP handler: AgentCore turns the
#    POSTed JSON body into `payload` and serialises whatever you return.
@app.entrypoint
def handler(payload: dict) -> dict:
    """One invocation. `payload` is the JSON body sent to the runtime.

    Returning a dict (rather than a bare string) is worth doing from the start:
    it gives you somewhere to put the structured state the tool wrote, alongside
    the model's prose, without changing the contract later.
    """
    prompt = payload.get("prompt") or "Summarise the document."
    doc = payload.get("doc_text") or DEFAULT_DOC

    agent = create_agent(
        model=_model(),
        tools=[count_words],
        state_schema=DocState,
        system_prompt=SYSTEM,
    )
    final = agent.invoke({
        "messages": [HumanMessage(content=prompt)],
        "doc_text": doc,
    })

    return {
        "answer": final["messages"][-1].content,
        # The custom state keys the TOOL wrote — the whole point of lab 03,
        # surviving the trip through a deployed runtime.
        "word_count": final.get("word_count"),
        "top_words": final.get("top_words"),
        "model": MODEL_ID,
    }


DEFAULT_DOC = """
Quarterly Operations Note

Throughput rose fourteen percent against the prior quarter, driven mainly by the
new intake process. Two sites reported handling delays; both were traced to a
single upstream dependency and are now resolved. Headcount was unchanged.

Outlook: we expect throughput to hold, with modest upside if the intake changes
are rolled out to the remaining three sites before the end of the period.
"""


# 4. Serve. Locally this starts an HTTP server on :8080; in the cloud AgentCore
#    runs the same server behind its own endpoint.
if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", "8080")))
