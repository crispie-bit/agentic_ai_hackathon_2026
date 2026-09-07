# CogniFlow OS ⚡

An intelligent, multi-agent academic and workday operating system featuring cascading LLM orchestration and instant local failsafe intelligence.

![CogniFlow](agentic_os_v2/static/cogniflow_logo.png)

---

## 🌟 Key Features

### 1. **Cascading Multi-Agent AI Orchestration**
CogniFlow automatically routes requests across a 4-tier model hierarchy to optimize for quality, speed, and 100% uptime:
- **Priority 1: AWS Bedrock** — Anthropic Claude 3.5 / 4.5 Sonnet for deep reasoning, schedule optimization, and multi-agent coordination.
- **Priority 2: Groq Cloud** — Ultra-fast (~300 tokens/sec) open-weights fallback (`qwen/qwen3.8-27b` / `llama-3.3-70b`).
- **Priority 3: Google Gemini** — High-context (1M tokens) secondary cloud failover (`gemini-3.6-flash`).
- **Priority 4: Local Specialist Engine** — Instant deterministic SQLite course dossier & document search (<10ms, 0 cloud cost, 100% offline).

### 2. **Real-Time AI Token & Quota Usage Monitor**
- Live token tracking modal accessible directly via the top navbar **AI** pill badge.
- Per-model daily free quotas, remaining allowances, inference latencies, and real-time diagnostic test pings.

### 3. **Course-Sorted Local Intelligence Hub**
- When operating offline or without cloud tokens, CogniFlow renders an interactive multi-module dossier instead of a chat mode.
- Interactive course tabs grouping:
  - Direct clickable PDF links to syllabus schedules, active tutorials, and past exam mock papers.
  - Prioritized assessment milestones and task deadlines with urgency scores.
  - Official Blackboard / NTULearn announcements.

### 4. **Dynamic Time-Blocking & Timetable Engine**
- Visual weekly class schedule and calendar planner with exam dates and venue mapping.
- Algorithmic prioritization engine for deadlines and daily study blocks.

---

## 🚀 Quick Start

### Windows (One-Click Launch)
```powershell
cd agentic_os_v2
.\run.ps1
```

### Manual Setup
```bash
cd agentic_os_v2
python -m venv .venv

# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
playwright install chromium

# Configure (optional for cloud models):
cp .env.example .env

# Launch:
python app.py
```

Open your browser to: **`http://127.0.0.1:8765`**

---

## 📂 Project Structure

```
crispie-bit/agentic_ai_hackathon_2026/
├── agentic_os_v2/           # ⭐ CogniFlow OS (Main Project)
│   ├── app.py               # FastAPI Backend & REST Endpoints
│   ├── run.ps1              # 1-Click Launch (Windows)
│   ├── run.sh               # 1-Click Launch (macOS/Linux)
│   ├── requirements.txt     # Core Dependencies
│   ├── core/                # Config, Paths, Database
│   ├── services/            # LLM Cascade, Multi-Agent Reasoning, Course Parsing
│   └── static/              # Glassmorphic Dashboard UI
│
├── agentic_system/          # Agentic Workday OS (Parallel Development)
│   └── README.md            # Multi-agent coordination with Outlook/NTULearn
│
└── lab/                     # 🎓 Agentic AI Hackathon Workshop Labs
    ├── section_1_foundation/          # LLM Fundamentals (Groq)
    ├── section_2_agentic_ai_basic/    # Agents, Memory, Tools, RAG
    ├── section_3_bedrock/             # AWS Bedrock Integration
    ├── section_4_langgraph/           # LangGraph State Machines
    ├── section_5_deepagents/          # DeepAgents & Claude Agent SDK
    └── section_6_agentcore/           # Bedrock AgentCore Deployment
```

---

## 🎓 Agentic AI Hackathon Workshop Labs

This repository includes a comprehensive 6-section hands-on lab curriculum that provided the foundations for CogniFlow. The labs progress from LLM fundamentals through multi-agent systems to production deployment:

| Section | Folder | Covers | Platform |
|---------|--------|--------|----------|
| §1 | `lab/section_1_foundation/` | LLM call fundamentals | Groq (Free Tier) |
| §2 | `lab/section_2_agentic_ai_basic/` | Memory, Tools, Planning, RAG | Groq |
| §3 | `lab/section_3_bedrock/` | AWS Bedrock API, Multimodal | Bedrock |
| §4 | `lab/section_4_langgraph/` | State Machines, Workflows | Either |
| §5 | `lab/section_5_deepagents/` | DeepAgents, Agent SDK | Bedrock |
| §6 | `lab/section_6_agentcore/` | AgentCore Deployment | Bedrock |

### Lab Features
- **No AWS account required for Day 1** — runs on Groq's free tier
- **Day 2 uses Bedrock** — swap one line of code to move to production
- **Hands-on exercises** with `TODO` blocks in:
  - `lab/section_2_agentic_ai_basic/05_agent_lab.py` — agent loop (4 TODOs)
  - `lab/section_2_agentic_ai_basic/06d_rag.py` — retrieval patterns (2 TODOs)
  - `lab/section_2_agentic_ai_basic/06_prompt_engineering.py` — prompt grading (3 challenges)
  - `lab/section_4_langgraph/01_graph_lab.py` — state & graph design

**Full lab documentation:** [`lab/README.md`](lab/README.md)  
**AWS setup guide:** [`lab/AWS_SETUP.md`](lab/AWS_SETUP.md)

### ⚠️ AWS Resource Warning (§6 Only)
Section 6 creates real, billable AWS resources (runtime, IAM role, S3). Cleanup:
```bash
cd lab/section_6_agentcore
uv run 03_teardown.py --yes   # Delete resources
```

---

## 🔐 Privacy & Security

- **Keys are never committed** — `.env` is gitignored. Create your own for Groq/AWS credentials.
- **No client data** — Everything uses synthetic, fictional data (names, amounts, companies).
- **Runtime data is local** — NTULearn sessions, tokens, and databases are kept local and ignored by Git.

---

## 📄 License

MIT License. Created by students for students and professionals.

---

## 🎯 Getting Started

**New to CogniFlow?** Start here:
1. Launch CogniFlow OS: [`agentic_os_v2/README.md`](agentic_os_v2/README.md)

**Want to learn the concepts?** Take the labs:
2. Follow the workshop curriculum: [`lab/README.md`](lab/README.md)

**Building a workday assistant?** Check the parallel project:
3. Explore multi-agent patterns: [`agentic_system/README.md`](agentic_system/README.md)
