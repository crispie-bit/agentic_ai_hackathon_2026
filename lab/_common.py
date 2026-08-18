"""
Shared setup for every lab script: which model, which provider, and the two
helpers (banner, report_usage) that every file prints with.

  ==================================================================
  ONE LINE decides which provider every lab talks to:

      PROVIDER = "groq"      <- Session 1. Free tier, no AWS, no cost.
      PROVIDER = "bedrock"   <- Session 2, Demo A. Same files, same output.

  No lab file below this one mentions a provider. That is deliberate, and
  it is the point of Demo A: the agent you write on Day 1 is the agent that
  runs on Bedrock on Day 2, and the diff is this line.
  ==================================================================

Import this first in each lab file. It does no network I/O on import, and on
the Groq path it never imports boto3 — so Session 1 runs with no AWS anything
installed.
"""

import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    """Read lab/.env into the environment, if it exists.

    `export GROQ_API_KEY=...` lasts until you close the terminal, which is how
    people end up keyless halfway through a lab. A .env file next to this one
    survives new tabs, editor restarts and Windows.

    A real env var always wins — this only fills in what is missing. Deliberately
    hand-rolled rather than pulling in python-dotenv: one less thing to install
    at the start of a session, and it is eight lines.
    """
    path = Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

# The one line. Override without editing the file:  export LLM_PROVIDER=bedrock
PROVIDER = os.environ.get("LLM_PROVIDER", "groq")

# --------------------------------------------------------------------------
# Model ids. Never string-build a model id — copy it from the provider's
# console or a constant like these. At least one Bedrock id in our own tree
# breaks the usual -YYYYMMDD-vN:0 convention, so a "helpful" f-string that
# assembles ids will work for months and then fail on one model.
# --------------------------------------------------------------------------

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# Kept as MODEL_ID because labs 01-08 (Session 2 and take-home) import that name.
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
)
REGION = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get(
    "AWS_REGION", "ap-southeast-1"
)


def model_label() -> str:
    """What to print at the top of a lab, so the trace says what produced it."""
    return f"{PROVIDER}:{GROQ_MODEL if PROVIDER == 'groq' else MODEL_ID}"


# --------------------------------------------------------------------------
# Groq — the Session 1 path.
# --------------------------------------------------------------------------

def require_groq_key() -> None:
    """Exit with a sentence instead of a stack trace if the key is missing."""
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit(
            "No GROQ_API_KEY found.\n\n"
            "  1. Sign in at https://console.groq.com  (free, no card)\n"
            "  2. API Keys -> Create API Key -> copy it NOW (shown once)\n"
            "  3. either:\n"
            "       export GROQ_API_KEY=gsk_...        (this terminal only)\n"
            "     or, better:\n"
            "       echo 'GROQ_API_KEY=gsk_...' > lab/.env\n\n"
            "  The .env file survives new terminal tabs. It is gitignored —\n"
            "  do not commit a key, and do not paste one in the chat."
        )


def groq_client():
    """The raw Groq SDK client. Lab 1 uses this to see the wire format."""
    require_groq_key()
    from groq import Groq

    return Groq()


# --------------------------------------------------------------------------
# Bedrock — the Session 2 path. boto3 is imported inside the functions so the
# Groq path never needs it installed.
# --------------------------------------------------------------------------

def require_aws_credentials() -> None:
    import boto3

    creds = boto3.Session().get_credentials()
    if creds is None or not creds.access_key:
        sys.exit(
            "No AWS credentials found.\n\n"
            "  boto3 looked at: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY,\n"
            "  ~/.aws/credentials, and any SSO or instance role.\n\n"
            "  Fix with ONE of:\n"
            "    export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...\n"
            "    aws sso login --profile <your-profile>\n\n"
            f"  Region currently resolves to: {REGION}\n"
            "  Set it with: export AWS_DEFAULT_REGION=ap-southeast-1\n\n"
            "  Session 1 needs none of this. If you are on Day 1, you have\n"
            "  LLM_PROVIDER=bedrock set by mistake — unset it."
        )


# Kept for labs 01-08, which were written against boto3 directly.
def bedrock_runtime():
    import boto3
    from botocore.config import Config

    require_aws_credentials()
    # read_timeout=300 because a long generation blows through the 60s default
    # and fails after you have already paid for the tokens. max_attempts=1 so
    # throttling reaches our own retry logic instead of being retried (and
    # hidden) down at the botocore layer.
    return boto3.client(
        "bedrock-runtime",
        config=Config(
            region_name=REGION,
            read_timeout=300,
            connect_timeout=120,
            retries={"max_attempts": 1},
        ),
    )


# --------------------------------------------------------------------------
# The one factory every LangChain lab calls.
# --------------------------------------------------------------------------

def chat_model(temperature: float = 0.0, **kwargs):
    """A LangChain chat model for whichever provider PROVIDER names.

    Both objects implement the same BaseChatModel interface, which is the
    entire argument for using LangChain at all: .invoke(), .bind_tools(),
    .with_structured_output() work identically on either side of this if.
    """
    if PROVIDER == "groq":
        require_groq_key()
        from langchain_groq import ChatGroq

        return ChatGroq(model=GROQ_MODEL, temperature=temperature, **kwargs)

    if PROVIDER == "bedrock":
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(
            model_id=MODEL_ID,
            client=bedrock_runtime(),
            temperature=temperature,
            **kwargs,
        )

    sys.exit(f"Unknown PROVIDER {PROVIDER!r}. Use 'groq' or 'bedrock'.")


# --------------------------------------------------------------------------
# Printing.
# --------------------------------------------------------------------------

def report_usage(label: str, usage) -> None:
    """Print token usage from any of the three shapes the labs encounter.

    Measure from day one. It is the only cost number that means anything for
    YOUR workload — multiply by the rate on the provider's pricing page, and
    do not trust a number in someone's comment (including this one).

      raw Groq / OpenAI-compatible   usage.prompt_tokens / .completion_tokens
      raw Bedrock                    inputTokens / outputTokens
      LangChain usage_metadata       input_tokens / output_tokens
    """
    if usage is None:
        return
    if not isinstance(usage, dict):                     # the raw Groq object
        usage = getattr(usage, "model_dump", lambda: vars(usage))()

    def pick(*names):
        for n in names:
            if usage.get(n):
                return usage[n]
        return 0

    inp = pick("input_tokens", "inputTokens", "prompt_tokens")
    out = pick("output_tokens", "outputTokens", "completion_tokens")
    cr = pick("cache_read_input_tokens")
    cc = pick("cache_creation_input_tokens")
    extra = f"  cache_read={cr} cache_write={cc}" if (cr or cc) else ""
    print(f"  [{label}] input={inp} output={out} total={inp + out}{extra}")


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
