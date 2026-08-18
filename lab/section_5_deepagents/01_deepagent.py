import _bootstrap  # noqa: F401

from _common import chat_model
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# 1. Define a tool
def lookup(topic: str) -> str:
    """Look up factual notes about a topic."""
    facts = {"state": "LangGraph state is a typed dict each node updates."}
    return facts.get(topic.lower(), "Topic not found.")

# 2. Initialize the Deep Agent (Supervisor + Filesystem + Auto-Planning)
agent = create_deep_agent(
    model=chat_model(),
    tools=[lookup],
    system_prompt="1. Use to-do tool first to plan.\n2. Look up facts.\n3. Write briefing.md.",
    backend=FilesystemBackend(root_dir="./workspace", virtual_mode=True),
)

# 3. Execute the single high-level goal
result = agent.invoke({
    "messages": [{"role": "user", "content": "Write briefing.md explaining LangGraph state."}]
})

print(result["messages"][-1].content)
