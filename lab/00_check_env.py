"""
Run this FIRST, before either session.  (10 seconds, no key needed)

    cd agentic_teaching/lab
    uv sync
    uv run 00_check_env.py

Four stages, in the order things go wrong:

    1 · PYTHON      new enough to run the labs
    2 · VENV        uv sync has run, and you are inside the venv it built
    3 · LIBRARIES   every section's imports, §1 to §6
    4 · KEYS        which provider is configured, and whether it has a key

Stages 1-3 decide the exit code, but only the Session 1 libraries are
required: the rest arrive with a `uv sync` extra and are reported as [next]
until the session that needs them. Stage 4 never fails the run — you are not
expected to have a key yet, so "not set" there is a normal result before the
session.

Nothing here touches the network, so it cannot prove a key is VALID. That is
what section_1_foundation/00_check_groq.py does, once you have one.

Fixes for every failure below are in README.md, under "Setup problems".
"""

import importlib.util
import os
import sys
from pathlib import Path

from _common import GROQ_MODEL, MODEL_ID, PROVIDER, REGION, banner

MIN_PYTHON = (3, 11)          # pyproject: requires-python = ">=3.11"
LAB = Path(__file__).resolve().parent

# Every section's imports, grouped the way pyproject groups them.
#
# Only the first row is required: it is the base `uv sync`, and without it
# nothing runs. The rest arrive with an extra and are [next] until the session
# that needs them — a Day 1 laptop is SUPPOSED to be missing them.
#
#   (label, required, [(import name, what installs it), ...])
SECTIONS = [
    ("§1 §2 §4  core", True, [
        ("groq", "groq"),
        ("langchain_groq", "langchain-groq"),
        ("langchain", "langchain"),
        ("langgraph", "langgraph"),
        ("pydantic", "pydantic"),
        ("grandalf", "grandalf"),
    ]),
    ("§3 §4 §6  Bedrock", False, [
        ("boto3", "boto3"),
        ("langchain_aws", "langchain-aws"),
    ]),
    ("§3        PDF labs", False, [
        ("fitz", "pymupdf"),
        ("pypdf", "pypdf"),
        ("reportlab", "reportlab"),
    ]),
    ("§5        DeepAgents", False, [
        ("deepagents", "deepagents"),
        ("claude_agent_sdk", "claude-agent-sdk"),
    ]),
    ("§6        AgentCore", False, [
        ("bedrock_agentcore", "bedrock-agentcore"),
        ("bedrock_agentcore_starter_toolkit", "bedrock-agentcore-starter-toolkit"),
    ]),
]

SEE_README = "See README.md, \"Setup problems\"."


def step(number: int, title: str) -> None:
    print(f"\n  {number} · {title}\n  {'-' * 60}")


def ok(label: str, detail: str = "") -> bool:
    print(f"  [ ok ] {label}{'   ' + detail if detail else ''}")
    return True


def fail(label: str, fix: str) -> bool:
    print(f"  [FAIL] {label}\n         {fix}")
    return False


def later(label: str, fix: str) -> None:
    """Expected to be missing now. There is a step for it later."""
    print(f"  [next] {label}\n         {fix}")


def have(module: str) -> bool:
    """True if the module can be found, without importing (and running) it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def check_python() -> bool:
    got = ".".join(str(n) for n in sys.version_info[:3])
    want = ".".join(str(n) for n in MIN_PYTHON)
    if sys.version_info[:3] < MIN_PYTHON:
        return fail(f"Python {got} — the labs need {want} or later",
                    f"uv installs its own: uv python install {want}")
    return ok("Python", got)


def check_venv() -> bool:
    """Almost every "the library is missing but I installed it" report is this:
    `python 00_check_env.py` runs system Python, which never sees what uv
    installed. `uv run` selects the venv for you."""
    venv = LAB / ".venv"
    if not venv.exists():
        return fail("lab/.venv does not exist — uv sync has not run here",
                    "cd agentic_teaching/lab && uv sync")
    if Path(sys.prefix).resolve() != venv.resolve():
        return fail(f"running outside the project venv ({sys.prefix})",
                    "Use: uv run 00_check_env.py")
    return ok("venv", str(venv))


def check_libraries() -> bool:
    """One row per section. Only the required row can fail the run."""
    core_ok = True
    incomplete = False

    width = max(len(label) for label, _r, _m in SECTIONS)

    for label, required, mods in SECTIONS:
        missing = [pkg for mod, pkg in mods if not have(mod)]
        if not missing:
            ok(f"{label:<{width}}", f"{len(mods)} libs")
        elif required:
            core_ok = fail(f"{label:<{width}}   missing: {', '.join(missing)}",
                           "uv sync")
        else:
            incomplete = True
            print(f"  [next] {label:<{width}}   missing: {', '.join(missing)}")

    if incomplete:
        # ONE command, deliberately. `uv sync --extra aws` on a machine that
        # already has the takehome extra uninstalls it again: uv sync makes the
        # venv match exactly what you name. Installing one extra at a time is
        # how people arrive at Day 2 with boto3 missing and a green log from
        # yesterday proving they installed it.
        print("         Not needed for Session 1. Before Session 2, in one command:")
        print("             uv sync --all-extras")
        print("         Adding extras one at a time REMOVES the ones you leave out.")

    return core_ok


def report_groq() -> bool:
    """Session 1. Free tier, no card."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        # Missing only matters if the labs are pointed at Groq. On Day 2 the
        # provider has moved to Bedrock and this is simply no longer used.
        hint = ("console.groq.com -> API Keys, then: echo 'GROQ_API_KEY=gsk_...' > .env"
                if PROVIDER == "groq"
                else "Not used while LLM_PROVIDER=bedrock.")
        later("GROQ_API_KEY not set", hint)
        return False
    if not key.startswith("gsk_"):
        later(f"GROQ_API_KEY starts {key[:4]!r}, expected 'gsk_'",
              "Usually a truncated paste, or quotes around the value.")
        return False
    return ok("GROQ_API_KEY", f"{key[:7]}...{key[-4:]}")


def report_aws() -> bool:
    """Session 2 and the take-home labs. Not needed for Session 1."""
    profile = os.environ.get("AWS_PROFILE")
    if not profile and not os.environ.get("AWS_ACCESS_KEY_ID"):
        # Absent AWS only matters if the labs are pointed at Bedrock. Say so,
        # because someone on Day 1 with LLM_PROVIDER=bedrock left in .env will
        # otherwise fail every lab with a credentials error and not know why.
        hint = ("Remove LLM_PROVIDER from lab/.env to run on Groq instead."
                if PROVIDER == "bedrock"
                else "Only needed for Session 2. Setup: AWS_SETUP.md")
        later("AWS credentials not configured", hint)
        return False
    return ok("AWS credentials", f"profile={profile or '(static keys)'}")


def report_credentials() -> bool:
    """Both are optional here. Returns whether the SELECTED provider is ready."""
    model = GROQ_MODEL if PROVIDER == "groq" else MODEL_ID
    where = f" in {REGION}" if PROVIDER == "bedrock" else ""
    print(f"         the labs will call:  {PROVIDER} -> {model}{where}\n")

    groq_ready = report_groq()
    aws_ready = report_aws()

    if PROVIDER == "groq":
        return groq_ready
    if PROVIDER == "bedrock":
        return aws_ready

    later(f"LLM_PROVIDER is {PROVIDER!r}, expected 'groq' or 'bedrock'",
          "Fix the LLM_PROVIDER line in lab/.env.")
    return False


def main() -> int:
    banner("ENVIRONMENT CHECK — offline, no key needed yet")

    step(1, "PYTHON")
    setup = [check_python()]

    step(2, "VENV")
    setup.append(check_venv())

    step(3, "LIBRARIES")
    setup.append(check_libraries())

    step(4, "CREDENTIALS  (optional — nothing here fails the check)")
    provider_ready = report_credentials()

    if not all(setup):
        banner("NOT READY — fix the [FAIL] lines, then run this again")
        print(f"  {SEE_README}  Stuck after two tries? Raise your hand.\n")
        return 1

    # Whichever provider is selected, the live check for it is the next step.
    live_check = ("section_1_foundation/00_check_bedrock.py"
                  if PROVIDER == "bedrock"
                  else "section_1_foundation/00_check_groq.py")

    if not provider_ready:
        banner("MACHINE READY — credentials are the only thing left")
        print(f"  Everything Session 1 needs is installed. Once {PROVIDER} is\n"
              f"  configured (stage 4 above says how), confirm it reaches the\n"
              f"  model:\n\n      uv run {live_check}\n")
        return 0

    banner("OK — ready for hands-on lab")
    return 0


if __name__ == "__main__":
    sys.exit(main())