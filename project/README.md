# Ruiheng Project Starter

This folder is the shared project base for the `Ruiheng` branch. It is separate
from the workshop labs so other branches can build product code without
rewriting the training material.

The starter implements a small support-triage agent:

- retrieves relevant policy/context articles
- routes a ticket to `billing`, `security`, `technical`, `account`, or `general`
- flags cases that need human handoff
- returns structured JSON for later UI, API, or AgentCore integration
- can persist decisions to JSONL
- can evaluate changes against labeled cases

It uses only the Python standard library, so it can run before Groq, Bedrock,
or AWS dependencies are configured.

## Run

From the repository root:

```bash
PYTHONPATH=project/src python -m hackathon_agent \
  "I see a payment I never made on my statement."
```

Run with custom knowledge and an audit log:

```bash
PYTHONPATH=project/src python -m hackathon_agent \
  --knowledge project/data/knowledge_articles.jsonl \
  --history project/runs/decisions.jsonl \
  "The invoice PDF won't download."
```

Evaluate routing quality:

```bash
PYTHONPATH=project/src python -m hackathon_agent --evaluate
```

Start the local HTTP API:

```bash
PYTHONPATH=project/src python -m hackathon_agent --serve
curl -s http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/handle \
  -H 'content-type: application/json' \
  -d '{"ticket":"A device I do not own is in my active sessions."}'
```

Optional LLM-backed classification is behind the same `ProjectAgent` contract.
Set `HACKATHON_LLM_URL`, `HACKATHON_LLM_API_KEY`, and `HACKATHON_LLM_MODEL`,
then pass `--llm-classifier`. If the LLM returns unusable JSON, the agent falls
back to the deterministic classifier.

Run the tests:

```bash
PYTHONPATH=project/src python -m unittest discover -s project/tests
```

## Extend

Good branch points from here:

- replace `tools.py` routing with a richer LLM-backed classifier
- load `KnowledgeBase` articles from a database or vector search
- expose `ProjectAgent.handle()` through FastAPI or AgentCore
- add project-specific tools with side effects behind explicit approvals
- add richer evaluation datasets per domain
