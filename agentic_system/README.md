# Agentic Workday OS

This project is a setup-first scaffold for a personal productivity system with a single lead AI and focused specialist agents.

## Design goals

- The lead agent is the decision-maker and coordinator.
- Specialist agents stay narrow and low-token.
- AWS and Bedrock are ready in configuration, but intentionally disabled until the setup is tested.
- Speech is available as an optional capability, not a default runtime dependency.

## Agents

### 1) Lead agent
- Purpose: decide routing, priority, and summary
- Recommended model: Claude 3.5 Sonnet
- Why: broad reasoning and orchestration
- Good for: planning, delegation, synthesis

### 2) Outlook agent
- Purpose: monitor email and rank urgency
- Recommended model: Claude 3.5 Haiku
- Why: small, fast, good for short summaries and filtering
- Good for: inbox triage, action-item extraction

### 3) NTULearn agent
- Purpose: monitor course announcements, deadlines, and assignment updates
- Recommended model: Claude 3.5 Haiku
- Why: narrow task, short structured output
- Good for: deadline reminders, assignment tracking

### 4) Voice agent
- Purpose: handle speech interface and short spoken responses
- Recommended model: Nova Micro
- Why: low-latency, cheaper voice workflow
- Good for: quick spoken summaries and digests

## Tools to prefer


## AWS strategy


## Setup

```bash
cd agentic_system
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Then add your AWS keys to `.env` (private file) and enable only the features you need.

## Latest feature update

The Agentic Workday OS now includes a runnable Streamlit interface and staged first-run setup:


### Teammate task summary

Completed: Outlook Graph integration, Azure configuration validation, AWS credential validation, NTULearn session handling, shared workspace storage, staged onboarding UI, chat retrieval, voice controls, prerequisite installation, regression tests, and application startup validation.

Remaining integration task: confirm NTULearn's live course-page selectors and download behavior against an authenticated student account so every course file can be downloaded and extracted automatically.

### GitHub feature description

**feat: add staged Agentic Workday OS onboarding and workspace assistant**

This feature adds the first runnable end-user workflow for the Agentic Workday OS. Users can launch the Streamlit interface, authenticate NTULearn in a visible browser, sign in to Outlook through Microsoft Graph, ingest source data into a local SQLite workspace, and query the indexed content through a preparation-gated assistant. Optional voice input and spoken responses are included, and the prerequisite installer now provisions the UI, voice dependencies, and Playwright Chromium browser.

The workflow requires real AWS credentials, Microsoft Entra app registration values, Microsoft consent, and an NTULearn SSO session. Secrets remain environment-only and are not committed.

## Product progression

This `agentic_system` is the canonical product path for the hackathon team. It
builds the student workday assistant on top of the workshop lessons:

1. Ingest NTULearn course information; keep Outlook as an optional adapter.
2. Store grounded source records locally and protect runtime data.
3. Extract structured tasks with due times, effort, priority, and source.
4. Show the student what matters next and answer grounded questions.
5. Run the lead model/tool/observation loop for prioritisation and replanning
	after the deterministic flow is validated.
6. Use Bedrock when the account policy permits `bedrock:InvokeModel`.
7. Add LangGraph/AgentCore only after the core workflow is reliable.

The parallel support-triage project on another team branch is useful as a
source of evaluation, structured-output, and deterministic-tool patterns. It
is not merged wholesale because its domain and UI are different from this
student workday product.

### Answer engine

The assistant uses Groq `openai/gpt-oss-20b` when `GROQ_API_KEY` is configured
in the private `.env`. It sends the student question and ranked workspace tasks
to a concise lead-agent prompt. Without a key, the same interface uses the
deterministic offline planner, so the demo remains runnable without cloud
access. Bedrock can replace Groq after `bedrock:InvokeModel` is permitted.
