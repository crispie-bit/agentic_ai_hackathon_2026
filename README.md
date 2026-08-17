# agentic_ai_hackathon_2026

Runnable lab for the Agentic AI hackathon workshop: LLM foundations → agents →
Amazon Bedrock → LangGraph → DeepAgents & the Claude Agent SDK → deploying to
Bedrock AgentCore.

Two sessions. **Day 1 needs no AWS account** — it runs on Groq's free tier.
Day 2 moves the same code to Bedrock by changing one line.

## Quick start

```bash
cd lab
uv sync
echo 'GROQ_API_KEY=gsk_...' > .env          # console.groq.com -> API Keys
uv run 00_check_env.py                      # offline preflight, no key needed
uv run section_1_foundation/00_check_groq.py
```

Then work through the sections in order. Each folder has its own README.

| | Folder | Covers | Runs on |
|---|---|---|---|
| §1 | `lab/section_1_foundation/` | what an LLM call actually is | Groq |
| §2 | `lab/section_2_agentic_ai_basic/` | memory, tools, planning, the loop, prompting & RAG | Groq |
| §3 | `lab/section_3_bedrock/` | the raw call, tools, cost, multimodal | Bedrock |
| §4 | `lab/section_4_langgraph/` | state, nodes, edges, tools × state | either |
| §5 | `lab/section_5_deepagents/` | the abstraction ladder, sub-agents, Agent SDK | Bedrock |
| §6 | `lab/section_6_agentcore/` | deploying the agent as an HTTPS service | Bedrock |

Full setup, keys and troubleshooting: [`lab/README.md`](lab/README.md).
AWS CLI and SSO setup: [`lab/AWS_SETUP.md`](lab/AWS_SETUP.md).

## Hands-on labs

Most files are demos to run and read. These are exercises with `TODO(student)`
blocks — the file runs before you touch it and gets the answer wrong in a
useful way:

- `lab/section_2_agentic_ai_basic/05_agent_lab.py` — the agent loop, 4 TODOs
- `lab/section_2_agentic_ai_basic/06d_rag.py` — retrieval from scratch, 2 TODOs
- `lab/section_2_agentic_ai_basic/prompt_engineering.py` — 4 prompt experiments to break
- `lab/section_4_langgraph/01_graph_lab.py` — state, nodes, edges

## ⚠ §6 creates real, billable AWS resources

Everything else is an API call that finishes. Deploying to AgentCore leaves a
runtime, an IAM role and an S3 upload in your account, and a runtime in READY
bills whether or not anything calls it.

```bash
uv run section_6_agentcore/03_teardown.py          # dry run
uv run section_6_agentcore/03_teardown.py --yes    # delete
uv run section_6_agentcore/03_teardown.py --list   # everything in the region
```

Run `--list` at the end of the session.

## Keys are never committed

`lab/.env` holds your Groq key and AWS profile, and is gitignored. If you fork
or clone this, create your own — there are no credentials in this repository.

## No client data

Everything is synthetic: fictional staff, fictional amounts, a fictional
company. The code patterns come from real projects; the data never does.
