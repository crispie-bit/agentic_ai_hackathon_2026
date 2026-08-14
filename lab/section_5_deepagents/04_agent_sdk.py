"""
§5 · 04 — The Claude Agent SDK.  (slides 51 and 52)

    npm install -g @anthropic-ai/claude-code     # once, and it is required
    uv run section_5_deepagents/04_agent_sdk.py

The top rung. The SDK drives the same engine as Claude Code, so you get files,
bash and skills for free — and almost no visibility into the loop.

Two halves, and they only work together:

  SLIDE 51  tools as an in-process MCP server
            @tool(name, description, schema) + create_sdk_mcp_server(...)
  SLIDE 52  ClaudeAgentOptions
            which model, which tools, what shape the answer takes, when to stop

READ THIS BEFORE RUNNING

  This script needs the `claude` CLI on your PATH, because the SDK runs it as
  a SUBPROCESS. That is a real dependency and the most common reason this file
  does not run. If it is missing, the script says so and exits.

HOW THIS DIFFERS FROM LANGCHAIN'S @tool

  LangChain infers the name, description and schema from your function's name,
  docstring and type hints. Here all three are EXPLICIT arguments, tools are
  async, and they return content blocks rather than strings — the same block
  shape you saw coming back from Bedrock in §3.

  is_error=True reports a failure to the model without ending the run.
"""

import asyncio
import os
import shutil
import sys

import _bootstrap  # noqa: F401

from _common import MODEL_ID, REGION, banner

PROMPT = ("Analyse this sentence and report the word count and longest word: "
          "'Context engineering is the practice of deciding what information "
          "goes into a model context so it performs reliably.'")


def build():
    from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool

    # ---- SLIDE 51: tools as an MCP server ------------------------------
    # Name, description and schema are EXPLICIT — not inferred.
    @tool("word_stats", "Compute word statistics for a piece of text.",
          {"text": str})
    async def word_stats(args: dict) -> dict:            # tools are async
        words = args.get("text", "").split()
        longest = max(words, key=len) if words else ""
        print(f"  [tool] word_stats: {len(words)} words")
        # Content blocks, not a string.
        return {"content": [{
            "type": "text",
            "text": f"word_count={len(words)} longest_word={longest!r}",
        }]}

    @tool("always_fails", "Always fails. Shows tool error reporting.", {})
    async def always_fails(args: dict) -> dict:
        # is_error tells the MODEL the call failed, so it can read the message
        # and try something else. An exception would kill the run instead.
        return {"content": [{"type": "text", "text": "Deliberate failure."}],
                "is_error": True}

    # In-process: no subprocess, no network — but the same protocol an
    # external MCP server speaks, so tools move between the two unchanged.
    server = create_sdk_mcp_server(name="my_tools",
                                   tools=[word_stats, always_fails])

    # ---- SLIDE 52: ClaudeAgentOptions ----------------------------------
    return ClaudeAgentOptions(
        model=MODEL_ID,
        max_turns=8,                       # the step cap from §2, renamed
        system_prompt=("You analyse text using your tools. Always call "
                       "word_stats rather than counting yourself."),
        env={
            # This one line routes the entire SDK through Bedrock.
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_DEFAULT_REGION": REGION,
            "AWS_PROFILE": os.environ.get("AWS_PROFILE", ""),
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_SESSION_TOKEN": os.environ.get("AWS_SESSION_TOKEN", ""),
            "API_TIMEOUT_MS": "600000",
        },
        mcp_servers={"my_tools": server},
        # An ALLOW-LIST, and therefore a security boundary. The convention is
        # mcp__<server>__<tool>. Get the string wrong and the tool silently
        # does not exist — no error, the model just never sees it.
        allowed_tools=["mcp__my_tools__word_stats",
                       "mcp__my_tools__always_fails"],
        disallowed_tools=["Agent"],        # stop it spawning more agents
    )


async def run() -> None:
    from claude_agent_sdk import (AssistantMessage, ResultMessage, TextBlock,
                                  ToolUseBlock, query)

    banner(f"RUNNING  ({MODEL_ID} via Bedrock)")
    # `query()` is the interface. There is no client object to construct.
    async for message in query(prompt=PROMPT, options=build()):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    print(f"  model -> {block.name}({str(block.input)[:60]})")
                elif isinstance(block, TextBlock):
                    print(f"  model: {block.text.strip()[:200]}")
        elif isinstance(message, ResultMessage):
            print(f"\n  result: {message.result}")
            if getattr(message, "usage", None):
                print(f"  usage:  {message.usage}")


def main() -> None:
    if shutil.which("claude") is None:
        sys.exit(
            "The `claude` CLI is not on your PATH.\n\n"
            "  The Agent SDK drives it as a subprocess, so it is required:\n"
            "    npm install -g @anthropic-ai/claude-code\n\n"
            "  If you cannot install it, read this file instead — slides 51\n"
            "  and 52 are about the SHAPE of the configuration, and that is\n"
            "  all visible in the source above."
        )
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        sys.exit("claude-agent-sdk is not installed.\n"
                 "  Run: uv sync --extra takehome")
    asyncio.run(run())


if __name__ == "__main__":
    main()
