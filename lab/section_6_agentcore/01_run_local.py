"""
§6 · 01 — Run the agent locally, before deploying anything.  (slide 53)

    uv run section_6_agentcore/01_run_local.py

Free. No AWS resources are created. Nothing is billed except the model call
the agent makes.

agent/agent.py is the §4 graph with FOUR LINES added:

    from bedrock_agentcore.runtime import BedrockAgentCoreApp   # 1. import
    app = BedrockAgentCoreApp()                                 # 2. initialise
    @app.entrypoint                                             # 3. decorate
    app.run(port=8080)                                          # 4. serve

That is the whole delta between "an agent on my laptop" and "an agent with an
HTTPS endpoint". The graph, the state schema and the tools are unchanged.

This script starts that server as a subprocess, POSTs one request to
/invocations, prints the response, and shuts it down. Debugging in the cloud
costs ~30 seconds a round trip; debugging here costs none, so do it here.
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401

from _common import REGION, banner

AGENT_DIR = Path(__file__).resolve().parent / "agent"
PORT = 8080
URL = f"http://localhost:{PORT}/invocations"

PAYLOAD = {
    "prompt": "What are the most common substantive words in the document?",
    "doc_text": (
        "Quarterly Operations Note\n\n"
        "Throughput rose fourteen percent against the prior quarter, driven "
        "mainly by the new intake process. Two sites reported handling delays; "
        "both were traced to a single upstream dependency and are now resolved. "
        "Headcount was unchanged.\n\n"
        "Outlook: we expect throughput to hold, with modest upside if the "
        "intake changes are rolled out to the remaining three sites."
    ),
}


def wait_for_server(proc: subprocess.Popen, timeout_s: int = 40) -> bool:
    """Poll until the entrypoint answers, or the subprocess dies."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/ping", timeout=2)
            return True
        except urllib.error.HTTPError:
            return True            # answered, even if /ping is not a route
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.5)
    return False


def main() -> int:
    if not (AGENT_DIR / "agent.py").exists():
        sys.exit(f"No agent at {AGENT_DIR / 'agent.py'}")

    # --serve leaves the server up so you can curl it yourself. Without it the
    # script sends one request and shuts the server down again.
    serve = "--serve" in sys.argv

    banner("STARTING agent/agent.py ON localhost:8080")
    print(f"  region: {REGION}")

    proc = subprocess.Popen(
        [sys.executable, "agent.py"],
        cwd=AGENT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        if not wait_for_server(proc):
            out = proc.stdout.read() if proc.stdout else ""
            print("  the server did not start:\n")
            print(out[-2000:])
            return 1
        print("  up.")

        if serve:
            banner("SERVER RUNNING — curl it from another terminal")
            print(f"""  curl -s -X POST {URL} \\
    -H "Content-Type: application/json" \\
    -d '{{"prompt":"Which words repeat most?","doc_text":"Throughput rose. The
intake process improved throughput. Two sites reported delays."}}'

  Ctrl-C to stop.""")
            try:
                proc.wait()
            except KeyboardInterrupt:
                pass
            return 0

        banner("POST /invocations")
        print(f"  prompt:   {PAYLOAD['prompt']}")
        print(f"  doc_text: {len(PAYLOAD['doc_text'].split())} words")

        request = urllib.request.Request(
            URL,
            data=json.dumps(PAYLOAD).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.time()
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read())
        elapsed = time.time() - started

        banner("RESPONSE")
        print(json.dumps(body, indent=2)[:1200])
        print(f"\n  {elapsed:.1f}s, locally. In the cloud this is the same JSON,")
        print("  over HTTPS, with session isolation and CloudWatch traces.")
        return 0

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("\n  server stopped. Nothing was deployed, nothing to tear down.")


if __name__ == "__main__":
    sys.exit(main())
