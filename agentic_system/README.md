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
