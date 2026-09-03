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

- Outlook agent: Microsoft Graph API for email queries
- NTULearn agent: direct fetcher or scraper with strict filtering
- Voice agent: speech-to-text + text-to-speech wrapper
- Lead agent: router and planner only, not a general-purpose scraper

## AWS strategy

- Keep AWS disabled initially.
- Use Bedrock only when the workflows are validated and you are ready to spend tokens.
- Start with narrow prompts and short outputs.

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

- Added a preparation gate that keeps assistant questions locked until platform setup is complete.
- Added visible NTULearn SSO login with persisted browser storage state.
- Added Outlook sign-in and Microsoft Graph mailbox ingestion into a local searchable workspace.
- Added a SQLite-backed workspace index and grounded keyword retrieval for course and email content.
- Added optional microphone transcription and local text-to-speech responses.
- Added `run_app.ps1` for Windows launch and expanded `setup_prereqs.py` to install Streamlit, voice packages, and Chromium.
- Added `MICROSOFT_REDIRECT_URI` to `.env.example` for the Graph authentication configuration.

### Teammate task summary

Completed: Outlook Graph integration, Azure configuration validation, AWS credential validation, NTULearn session handling, shared workspace storage, staged onboarding UI, chat retrieval, voice controls, prerequisite installation, regression tests, and application startup validation.

Remaining integration task: confirm NTULearn's live course-page selectors and download behavior against an authenticated student account so every course file can be downloaded and extracted automatically.

### GitHub feature description

**feat: add staged Agentic Workday OS onboarding and workspace assistant**

This feature adds the first runnable end-user workflow for the Agentic Workday OS. Users can launch the Streamlit interface, authenticate NTULearn in a visible browser, sign in to Outlook through Microsoft Graph, ingest source data into a local SQLite workspace, and query the indexed content through a preparation-gated assistant. Optional voice input and spoken responses are included, and the prerequisite installer now provisions the UI, voice dependencies, and Playwright Chromium browser.

The workflow requires real AWS credentials, Microsoft Entra app registration values, Microsoft consent, and an NTULearn SSO session. Secrets remain environment-only and are not committed.
