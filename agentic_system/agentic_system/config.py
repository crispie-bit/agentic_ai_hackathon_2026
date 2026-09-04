import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"

load_dotenv(ENV_FILE, override=False)

APP_NAME = os.getenv("APP_NAME", "agentic-workday-os")
APP_MODE = os.getenv("APP_MODE", "setup")
NTULEARN_BASE_URL = os.getenv("NTULEARN_BASE_URL", "https://ntulearn.ntu.edu.sg/")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", "")
ENABLE_SPEECH = os.getenv("ENABLE_SPEECH", "false").lower() in {"1", "true", "yes"}
ENABLE_OUTLOOK = os.getenv("ENABLE_OUTLOOK", "false").lower() in {"1", "true", "yes"}
ENABLE_NTU_LEARN = os.getenv("ENABLE_NTU_LEARN", "false").lower() in {"1", "true", "yes"}
ENABLE_AWS = (
    os.getenv("ENABLE_AWS", "false").lower() in {"1", "true", "yes"}
    or bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)
)

AWS_READY = APP_MODE.lower() in {"aws", "live", "production"} or ENABLE_AWS

AGENT_MODEL_IDS = {
    "lead_agent": os.getenv("LEAD_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
    "outlook_agent": os.getenv("OUTLOOK_MODEL_ID", "anthropic.claude-3-5-haiku-20241022-v1:0"),
    "ntulearn_agent": os.getenv("NTULEARN_MODEL_ID", "anthropic.claude-3-5-haiku-20241022-v1:0"),
    "voice_agent": os.getenv("VOICE_MODEL_ID", "amazon.nova-micro-v1:0"),
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
