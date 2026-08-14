"""
Shared helpers for labs 07 (deploy) and 08 (teardown).

Exists mainly to hold ONE piece of unpleasantness in a single documented place:
the starter toolkit's `Runtime` object keeps its state in memory, so in a fresh
process it has no idea which agent your `.bedrock_agentcore.yaml` refers to.

    def __init__(self):
        self._config_path = None
        self.name = None

`configure()` sets those. Everything else — `status()`, `invoke()`, `destroy()` —
raises "Must configure first" without them. So a script that only wants to check
or delete an existing deployment has to attach to the config by hand. There is no
public API for it as of toolkit 0.3.11; `attach_runtime()` below is the shim, and
it is the reason these two labs share a module.

(The `agentcore` CLI does not have this problem — it reads the yaml itself. If
you would rather shell out than use the Python API, `agentcore status` /
`agentcore destroy` are equivalent.)
"""

import os
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  — puts lab/ on sys.path

# Importing _common runs its .env loader, so AWS_PROFILE / AWS_DEFAULT_REGION
# from lab/.env reach boto3 here too. Without this the toolkit reports
# "NoCredentialsError" even though every other lab script works.
import _common  # noqa: F401

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE / "agent"
ENTRYPOINT = AGENT_DIR / "agent.py"
CONFIG = AGENT_DIR / ".bedrock_agentcore.yaml"

REGION = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get(
    "AWS_REGION", "ap-southeast-1"
)

# One runtime per student. Two people deploying the same name collide on the
# runtime and the second gets a confusing conflict, so default to something
# unique per machine and let it be overridden explicitly.
DEFAULT_NAME = "agentcore_lab_" + (
    os.environ.get("USER") or os.environ.get("USERNAME") or "student"
).replace("-", "_").replace(".", "_")[:24]


def banner(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def require_toolkit():
    try:
        from bedrock_agentcore_starter_toolkit import Runtime
    except ImportError:
        sys.exit("bedrock-agentcore-starter-toolkit is not installed.\n"
                 "  Run: uv sync --extra agentcore")
    return Runtime


def require_creds() -> None:
    import boto3
    if boto3.Session().get_credentials() is None:
        sys.exit("No AWS credentials found.\n"
                 "  export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...\n"
                 "  or: aws sso login --profile <your-profile>\n"
                 "  See lab/README.md.")


def new_runtime():
    """A fresh Runtime, cwd moved to the agent dir so the toolkit finds its yaml."""
    Runtime = require_toolkit()
    require_creds()
    os.chdir(AGENT_DIR)
    return Runtime()


def attach_runtime(name: str):
    """Return a Runtime bound to the already-configured agent `name`.

    See the module docstring: this reaches into two private attributes because
    the toolkit exposes no public way to load an existing config. If a future
    version adds one, replace this function and nothing else changes.
    """
    require_configured(name)
    rt = new_runtime()
    rt._config_path = CONFIG      # noqa: SLF001 — no public setter exists
    rt.name = name
    return rt


def require_configured(name: str) -> None:
    """Exit readably when there is nothing deployed.

    Students hit this straight after the teardown, which deletes the yaml once
    the last agent is gone. A raw "ValueError: Must configure first" traceback
    there is a guaranteed five minutes of confusion.
    """
    if CONFIG.exists():
        return
    sys.exit(
        f"Nothing is configured for '{name}'.\n\n"
        f"  {CONFIG.name} does not exist. Either you have not deployed yet, or\n"
        f"  the teardown removed it — it deletes the file once the last agent\n"
        f"  is gone, which is expected.\n\n"
        f"  Deploy with:    uv run section_6_agentcore/02_deploy.py --agent {name}\n"
        f"  Check AWS with: uv run section_6_agentcore/03_teardown.py --list"
    )


def configured_agents() -> list:
    """Agent names present in the local config."""
    if not CONFIG.exists():
        return []
    try:
        import yaml
        data = yaml.safe_load(CONFIG.read_text()) or {}
        return sorted((data.get("agents") or {}).keys())
    except Exception:
        return []


def fetch_runtimes() -> list:
    import boto3
    c = boto3.client("bedrock-agentcore-control", region_name=REGION)
    return c.list_agent_runtimes().get("agentRuntimes", [])


def list_remote() -> None:
    """List runtimes actually in the account/region.

    Deliberately separate from the local config: the yaml is just a file. A
    runtime deployed from another machine, or left behind after a `git clean`,
    will not appear in it and will still be charged for.
    """
    banner(f"AGENTCORE RUNTIMES IN {REGION}")
    try:
        runtimes = fetch_runtimes()
    except Exception as exc:
        print(f"  could not list runtimes: {type(exc).__name__}: {exc}")
        print("  (needs bedrock-agentcore:ListAgentRuntimes)")
        return
    if not runtimes:
        print("  none — nothing is costing you anything here.")
        return
    for r in runtimes:
        print(f"  {r.get('agentRuntimeName', '?'):40} {r.get('status', '?'):10} "
              f"{r.get('agentRuntimeId', '')}")
    print(f"\n  {len(runtimes)} runtime(s). Anything you do not recognise is "
          f"worth investigating.")


def wait_gone(name: str, timeout_s: int = 180) -> bool:
    """Poll until the runtime disappears.

    Deletion is asynchronous — the API returns straight away and the runtime
    sits in DELETING for a few seconds. Reporting success while it is still
    DELETING would teach the wrong habit: always confirm a teardown finished.
    """
    print(f"  waiting for '{name}' to disappear...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            names = [r.get("agentRuntimeName") for r in fetch_runtimes()]
        except Exception as exc:
            print(f"  could not confirm: {type(exc).__name__}: {exc}")
            return False
        if name not in names:
            print("  confirmed gone.")
            return True
        time.sleep(5)
    print(f"  still present after {timeout_s}s — check the console.")
    return False


def teardown_hint(name: str) -> None:
    print(f"""
{'!' * 72}
LEAVING THIS DEPLOYED COSTS MONEY. When you are done:

    cd {HERE.parent}
    uv run section_6_agentcore/03_teardown.py --agent {name} --yes

Run it without --yes first to see exactly what would be deleted.
{'!' * 72}""")
