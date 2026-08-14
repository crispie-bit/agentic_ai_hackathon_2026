# Lab — Agentic AI practitioner track

Six sections, one per part of the deck. **Day 1 needs no AWS account** — it
runs on Groq's free tier. Day 2 moves the same code to Bedrock.

```bash
cd agentic_teaching/lab
uv sync                              # Session 1 only, ~20s
echo 'GROQ_API_KEY=gsk_...' > .env   # see "Keys" below
uv run section_1_foundation/00_check_groq.py
```

| | Folder | Covers | Runs on |
|---|---|---|---|
| §1 | `section_1_foundation/` | what an LLM call is | Groq |
| §2 | `section_2_agentic_ai_basic/` | memory, tools, planning, the loop | Groq |
| §3 | `section_3_bedrock/` | the raw call, tools, cost, multimodal | Bedrock |
| §4 | `section_4_langgraph/` | state, nodes, edges, tools × state | either |
| §5 | `section_5_deepagents/` | the abstraction ladder, sub-agents, Agent SDK | Bedrock |
| §6 | `section_6_agentcore/` | deploying the agent as an HTTPS service | Bedrock |

Each folder has its own `README.md` with what to run and in what order.

## Which sections are hands-on

| | |
|---|---|
| **Labs** (students type) | `section_2/05_agent_lab.py`, `section_4/01_graph_lab.py` |
| **Solutions** | the `*_solution.py` next to each lab |
| **Everything else** | demos you run and narrate |

## Keys

One file, `lab/.env`, read by `_common.py` on import. It is gitignored.

```
# Session 1 — free, no card. console.groq.com -> API Keys
GROQ_API_KEY=gsk_...

# Session 2 — comment out to send every lab back to Groq
LLM_PROVIDER=bedrock
AWS_PROFILE=sentia
AWS_DEFAULT_REGION=ap-southeast-1
```

A real environment variable always beats `.env`. Check which provider is live:

```bash
uv run python -c "from _common import model_label; print(model_label())"
```

AWS setup (CLI, SSO, model access) is in **[AWS_SETUP.md](AWS_SETUP.md)** —
twenty minutes, and it must be done *before* Day 2, not during it.

## The one line that matters

`_common.py` has a single switch:

```python
PROVIDER = os.environ.get("LLM_PROVIDER", "groq")   # "groq" | "bedrock"
```

No lab file below it names a provider. That is the point of the Day 2 demo:
your Session 1 agent runs against Claude on Bedrock with nothing changed but
that line.

## Installing

```bash
uv sync                      # §1, §2, §4 on Groq
uv sync --extra aws          # + §3, §4, §5, §6 on Bedrock
uv sync --extra takehome     # + deepagents, Agent SDK, PDF tooling
uv sync --extra agentcore    # + the AgentCore deploy toolkit
```

`requirements.txt` mirrors this for pip users. Two things pip cannot install:

- the **`claude` CLI** (`npm install -g @anthropic-ai/claude-code`), needed
  only by `section_5/04_agent_sdk.py`
- the **AWS CLI**, for SSO

## ⚠ §6 creates real, billable AWS resources

Every other section is an API call that finishes. `section_6/02_deploy.py`
leaves a runtime, an IAM role and an S3 upload in your account until you
delete them, and a runtime in READY bills whether or not anything calls it.

```bash
uv run section_6_agentcore/03_teardown.py          # dry run — read the list
uv run section_6_agentcore/03_teardown.py --yes    # actually delete
uv run section_6_agentcore/03_teardown.py --list   # everything in the region
```

Run `--list` at the end of the session. The local `.bedrock_agentcore.yaml` is
just a file and can be out of date; the account is the truth.

## LangChain 1.x, not 0.3.x

Names moved. When an import breaks, check whether the symbol *moved* before
assuming it was removed:

| 0.3.x | 1.x |
|---|---|
| `langgraph.prebuilt.create_react_agent` | `langchain.agents.create_agent` |
| `create_react_agent(prompt=...)` | `create_agent(system_prompt=...)` |
| `langgraph.prebuilt.InjectedToolCallId` | `langchain_core.tools.InjectedToolCallId` |
| `langgraph.prebuilt.InjectedState` | unchanged |

## Sample data

`make_sample_pdf.py` generates `sample_docs/expenses.pdf`, used by
`section_3/05_multimodal.py` and `section_3/06_extraction.py`:

```bash
uv run make_sample_pdf.py
```

Everything here is synthetic — fictional staff, fictional amounts, a fictional
company. The code patterns come from real projects; the data never does.
