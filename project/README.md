# Ruiheng Project Starter

This folder is the shared project base for the `Ruiheng` branch. It is separate
from the workshop labs so other branches can build product code without
rewriting the training material.

The starter implements a small support-triage agent:

- retrieves relevant policy/context articles
- routes a ticket to `billing`, `security`, `technical`, `account`, or `general`
- flags cases that need human handoff
- returns structured JSON for later UI, API, or AgentCore integration

It uses only the Python standard library, so it can run before Groq, Bedrock,
or AWS dependencies are configured.

## Run

From the repository root:

```bash
PYTHONPATH=project/src python -m hackathon_agent \
  "I see a payment I never made on my statement."
```

Run the tests:

```bash
PYTHONPATH=project/src python -m unittest discover -s project/tests
```

## Extend

Good branch points from here:

- replace `tools.py` routing with an LLM-backed classifier
- load `KnowledgeBase` articles from files, a database, or vector search
- expose `ProjectAgent.handle()` through FastAPI or AgentCore
- add project-specific tools with side effects behind explicit approvals
- persist `AgentDecision` records for audit and evaluation

