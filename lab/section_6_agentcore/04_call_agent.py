"""
§6 · 04 — Call the deployed agent, the way an application would.

    uv run section_6_agentcore/04_call_agent.py
    uv run section_6_agentcore/04_call_agent.py "Which words repeat most?"
    uv run section_6_agentcore/04_call_agent.py --doc mydoc.txt
    uv run section_6_agentcore/04_call_agent.py --session-id alice
    uv run section_6_agentcore/04_call_agent.py --arn arn:aws:bedrock-agentcore:...

No toolkit, no deployment machinery — just boto3 and the ARN. This is what you
would paste into a Lambda, an API handler or another service.

WHY NOT curl

The endpoint requires SigV4-signed requests, so a plain curl gets a 403. boto3
signs with whatever credentials your session has, which is why this works with
your SSO login and nothing else is needed.

SESSIONS

    runtimeSessionId gives you per-session isolation. Pass different ids and
    the runtime keeps the callers apart; pass the same id twice and you are
    talking to the same session. Must be at least 33 characters, so short
    names given with --session-id are padded below.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import boto3

import _bootstrap  # noqa: F401

from _common import REGION, banner

CONFIG = Path(__file__).resolve().parent / "agent" / ".bedrock_agentcore.yaml"

DEFAULT_PROMPT = "What are the most common substantive words in the document?"
DEFAULT_DOC = (
    "Quarterly Operations Note\n\n"
    "Throughput rose fourteen percent against the prior quarter, driven mainly "
    "by the new intake process. Two sites reported handling delays; both were "
    "traced to a single upstream dependency and are now resolved. Headcount "
    "was unchanged.\n\n"
    "Outlook: we expect throughput to hold, with modest upside if the intake "
    "changes are rolled out to the remaining three sites."
)


def arn_from_config() -> str:
    """Read the deployed agent's ARN out of .bedrock_agentcore.yaml.

    That file is written by `configure` and is just a local file — if you
    deployed from another machine it will not be here. Pass --arn instead, or
    find it with:  uv run section_6_agentcore/03_teardown.py --list
    """
    if not CONFIG.exists():
        sys.exit(f"No {CONFIG.name} — nothing is deployed from this checkout.\n"
                 "  Deploy:  uv run section_6_agentcore/02_deploy.py\n"
                 "  Or pass an ARN:  --arn arn:aws:bedrock-agentcore:...")
    import yaml

    data = yaml.safe_load(CONFIG.read_text()) or {}
    agents = data.get("agents") or {}
    for cfg in agents.values():
        arn = (cfg.get("bedrock_agentcore") or {}).get("agent_arn")
        if arn:
            return arn
    sys.exit(f"No agent_arn in {CONFIG}. Has `launch` finished?")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    ap.add_argument("--doc", help="path to a text file to analyse")
    ap.add_argument("--arn", help="agent runtime ARN (default: read from yaml)")
    ap.add_argument("--session-id", default=None,
                    help="runtimeSessionId — different ids are isolated")
    args = ap.parse_args()

    arn = args.arn or arn_from_config()
    doc = Path(args.doc).read_text() if args.doc else DEFAULT_DOC

    payload = {"prompt": args.prompt, "doc_text": doc}

    banner("CALLING THE DEPLOYED AGENT")
    print(f"  arn      {arn}")
    print(f"  region   {REGION}")
    print(f"  prompt   {args.prompt}")
    print(f"  doc      {len(doc.split())} words"
          f"{f'  from {args.doc}' if args.doc else '  (built-in sample)'}")

    client = boto3.client("bedrock-agentcore", region_name=REGION)

    kwargs = {
        "agentRuntimeArn": arn,
        "qualifier": "DEFAULT",
        "contentType": "application/json",
        "payload": json.dumps(payload),
    }
    if args.session_id:
        # The API requires >= 33 characters, so pad short, human-friendly names.
        kwargs["runtimeSessionId"] = args.session_id.ljust(33, "-")
        print(f"  session  {kwargs['runtimeSessionId']}")

    started = time.time()
    try:
        response = client.invoke_agent_runtime(**kwargs)
    except client.exceptions.ResourceNotFoundException:
        sys.exit(f"\nNo runtime with that ARN in {REGION}.\n"
                 "  It may have been torn down. Check:\n"
                 "    uv run section_6_agentcore/03_teardown.py --list")
    except client.exceptions.AccessDeniedException as exc:
        sys.exit(f"\nAccess denied: {exc}\n"
                 "  Your credentials are valid but lack "
                 "bedrock-agentcore:InvokeAgentRuntime.")
    elapsed = time.time() - started

    # `response` is a STREAM, exactly like InvokeModel's body in §3.
    body = json.loads(response["response"].read())

    banner("RESPONSE")
    print(json.dumps(body, indent=2)[:1500])
    print(f"\n  {elapsed:.1f}s   HTTP {response.get('statusCode')}")
    if response.get("runtimeSessionId"):
        print(f"  session: {response['runtimeSessionId']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
