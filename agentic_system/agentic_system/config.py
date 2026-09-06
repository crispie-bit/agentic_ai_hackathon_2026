import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"
NTULEARN_SESSION_PATH = ROOT_DIR / "ntulearn_session.json"

load_dotenv(ENV_FILE, override=False)

APP_NAME = os.getenv("APP_NAME", "agentic-workday-os")
APP_MODE = os.getenv("APP_MODE", "setup")
NTULEARN_BASE_URL = os.getenv("NTULEARN_BASE_URL", "https://ntulearn.ntu.edu.sg/")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "true").lower() in {"1", "true", "yes"}
# us.anthropic.* cross-region inference profiles are hosted in us-east-1.
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", "")

# When explicit keys are provided, remove AWS_PROFILE to avoid botocore ProfileNotFound.
# This matches the fix Harvey applied after the hackathon credentials were distributed.
if AWS_ACCESS_KEY_ID and os.environ.get("AWS_PROFILE") == "default":
    os.environ.pop("AWS_PROFILE", None)

ENABLE_SPEECH = os.getenv("ENABLE_SPEECH", "false").lower() in {"1", "true", "yes"}
ENABLE_OUTLOOK = os.getenv("ENABLE_OUTLOOK", "false").lower() in {"1", "true", "yes"}
ENABLE_NTU_LEARN = os.getenv("ENABLE_NTU_LEARN", "false").lower() in {"1", "true", "yes"}
ENABLE_AWS = (
    os.getenv("ENABLE_AWS", "false").lower() in {"1", "true", "yes"}
    or bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)
)

AWS_READY = APP_MODE.lower() in {"aws", "live", "production"} or ENABLE_AWS

# Use us.anthropic.* cross-region inference profile IDs — these route to whichever
# AWS region has capacity and avoid the 'model not found' error on hackathon accounts.
AGENT_MODEL_IDS = {
    "lead_agent": os.getenv("LEAD_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
    "outlook_agent": os.getenv("OUTLOOK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    "ntulearn_agent": os.getenv("NTULEARN_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
    "voice_agent": os.getenv("VOICE_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
}

AGENT_TOOLS = {
    "lead_agent": ["task_router", "priority_planner", "response_summarizer"],
    "outlook_agent": ["microsoft_graph_email_query", "message_ranker", "summary_shortener"],
    "ntulearn_agent": ["ntulearn_fetcher", "deadline_parser", "announcement_filter"],
    "voice_agent": ["speech_to_text", "voice_response_generator"],
}

AGENT_PROMPTS = {
    "lead_agent": "Co-ordinate work, decide what needs action, and delegate to specialist agents only when needed.",
    "outlook_agent": "Scan for urgent emails, ignore noise, and return a compact action summary.",
    "ntulearn_agent": "Check announcements, assignments and deadlines; summarise only the items that matter.",
    "voice_agent": "Answer in short, natural spoken language and keep responses concise.",
}

LOW_TOKEN_GUIDANCE = {
    "max_output_tokens": 256,
    "temperature": 0.2,
    "system_style": "concise, structured, no unnecessary verbosity",
}
