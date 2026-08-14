"""
TAKE-HOME 4, part 1 — deploy the agent to Amazon Bedrock AgentCore.

    uv sync --extra agentcore
    uv run section_6_agentcore/02_deploy.py               # deploy + smoke test
    uv run section_6_agentcore/02_deploy.py --status      # check an existing deployment
    uv run section_6_agentcore/02_deploy.py --invoke-only # invoke without redeploying

>>> THIS LAB CREATES REAL AWS RESOURCES THAT COST MONEY UNTIL DELETED. <<<
    When you are finished:  uv run section_6_agentcore/03_teardown.py --yes

The agent is `agent/agent.py` — lab 03's agent plus four lines. This
script is the control plane: configure, launch, invoke.

# ======================================================================
# TODO(student) 1 — Run with --status after deploying. Note the agent ARN,
#   the execution role, and the endpoint status. Then find the agent in the
#   Bedrock AgentCore console and match them up.
#
# TODO(student) 2 — Deploy, then edit the SYSTEM prompt in
#   agent/agent.py and deploy again. Time both. The second is much
#   faster — work out what was cached and what wasn't.
#
# TODO(student) 3 — Invoke twice with the same --session-id and twice with
#   different ones. This agent is stateless, so the answers won't differ —
#   but check CloudWatch and see that sessions are traced separately. Then
#   reason about what you'd need to add for the checkpointer from §4 to work
#   here.
# ======================================================================

"""

import argparse
import json
import os
import sys
import time

import _bootstrap  # noqa: F401

from _agentcore import (DEFAULT_NAME, ENTRYPOINT, REGION, attach_runtime, banner,
                        new_runtime, teardown_hint)

SAMPLE_PAYLOAD = {
    "prompt": "What are the most common substantive words in the document?",
}


def do_configure(rt, name: str):
    banner(f"1/3  CONFIGURE  ({name})")
    print(f"  entrypoint      {ENTRYPOINT.name}")
    print(f"  region          {REGION}")
    print("  deployment_type direct_code_deploy   <- no Docker needed")
    result = rt.configure(
        entrypoint=str(ENTRYPOINT),
        agent_name=name,
        requirements_file="requirements.txt",
        region=REGION,
        # direct_code_deploy uploads the source to S3 and runs it on a managed
        # Python runtime. The alternative, "container", builds an ARM64 image via
        # CodeBuild — more control, more moving parts, and a Dockerfile.
        deployment_type="direct_code_deploy",
        runtime_type="PYTHON_3_13",
        auto_create_s3=True,
        # Let the toolkit make the IAM execution role. Without this you must
        # pre-create a role with bedrock:InvokeModel plus the AgentCore trust
        # policy, which is a lab in itself.
        auto_create_execution_role=True,
        protocol="HTTP",
        memory_mode="NO_MEMORY",
        non_interactive=True,
    )
    cfg = getattr(result, "config_path", None) or "(see .bedrock_agentcore.yaml)"
    print(f"\n  wrote {cfg}")
    return result


def do_launch(rt):
    banner("2/3  LAUNCH")
    print("  Uploading source, provisioning the runtime, waiting for READY.")
    print("  Measured on this lab: ~35s when the S3 source bucket already exists,")
    print("  longer the very first time in an account (it creates the bucket and")
    print("  the IAM role too).\n")
    t0 = time.time()
    result = rt.launch(auto_update_on_conflict=True)
    print(f"\n  launched in {time.time() - t0:.0f}s")
    for attr in ("agent_arn", "agent_id", "ecr_uri"):
        val = getattr(result, attr, None)
        if val:
            print(f"  {attr:10} {val}")
    return result


def do_invoke(rt, payload: dict, session_id: str | None = None):
    banner("3/3  INVOKE")
    print(f"  payload {json.dumps(payload)[:110]}")
    if session_id:
        print(f"  session {session_id}")
    t0 = time.time()
    resp = rt.invoke(payload, session_id=session_id)
    print(f"  responded in {time.time() - t0:.1f}s\n")

    # The runtime wraps the handler's return value. Unwrap defensively: the shape
    # has changed across toolkit versions, and a lab should not break on that.
    body = resp
    if isinstance(resp, dict):
        for key in ("response", "output", "result", "body"):
            if key in resp:
                body = resp[key]
                break
    if isinstance(body, (bytes, bytearray)):
        body = body.decode()
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, (bytes, bytearray)):
        body = body.decode()
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            pass

    if isinstance(body, dict) and "answer" in body:
        print(f"  answer      {body['answer'][:300]}")
        print(f"  word_count  {body.get('word_count')}   <- written by the TOOL, "
              f"in the cloud")
        print(f"  top_words   {body.get('top_words')}")
        print(f"  model       {body.get('model')}")
    else:
        print(f"  raw response: {str(body)[:600]}")
    return resp


def do_status(rt):
    banner("STATUS")
    st = rt.status()
    raw = st.model_dump() if hasattr(st, "model_dump") else dict(st)
    for section in ("config", "agent", "endpoint"):
        blob = raw.get(section)
        if not blob:
            continue
        print(f"\n  [{section}]")
        for k in ("name", "agent_arn", "agent_id", "status", "execution_role",
                  "deployment_type", "runtime_type", "region", "lastUpdatedAt"):
            if isinstance(blob, dict) and blob.get(k) is not None:
                print(f"    {k:16} {blob[k]}")
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy the lab agent to AgentCore.")
    ap.add_argument("--agent", default=os.environ.get("AGENTCORE_AGENT_NAME",
                                                      DEFAULT_NAME),
                    help=f"runtime name (default: {DEFAULT_NAME})")
    ap.add_argument("--status", action="store_true", help="show status and exit")
    ap.add_argument("--invoke-only", action="store_true",
                    help="invoke the existing deployment without redeploying")
    ap.add_argument("--no-invoke", action="store_true",
                    help="deploy but skip the smoke test")
    ap.add_argument("--session-id", default=None, help="reuse a session id")
    ap.add_argument("--prompt", default=SAMPLE_PAYLOAD["prompt"])
    args = ap.parse_args()

    rt = (attach_runtime(args.agent) if (args.status or args.invoke_only)
          else new_runtime())

    if args.status:
        do_status(rt)
        teardown_hint(args.agent)
        return 0

    if args.invoke_only:
        do_invoke(rt, {"prompt": args.prompt}, args.session_id)
        teardown_hint(args.agent)
        return 0

    do_configure(rt, args.agent)
    do_launch(rt)
    if not args.no_invoke:
        do_invoke(rt, {"prompt": args.prompt}, args.session_id)
    do_status(rt)

    banner("What just happened")
    print("""  configure  wrote .bedrock_agentcore.yaml, created an IAM execution
             role and an S3 bucket for the source upload
  launch     zipped agent/, uploaded it, provisioned a managed
             Python runtime, installed requirements.txt, waited for READY
  invoke     POSTed your JSON to the runtime's HTTPS endpoint, which called
             the @app.entrypoint handler and returned its dict

  You now have: an HTTPS endpoint, per-session isolation, CloudWatch logs and
  traces, and autoscaling. The agent code did not change to get any of it.""")
    teardown_hint(args.agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
