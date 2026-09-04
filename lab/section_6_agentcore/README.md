The same LangGraph agent from §4, behind a managed HTTPS endpoint. **The graph
does not change.**

```bash
cd agentic_teaching/lab
uv sync --extra agentcore --extra aws
```

| | File | Creates AWS resources? |
|---|---|---|
| 1 | `01_run_local.py` | **no** — free, do this first |
| 2 | `02_deploy.py` | **YES — billable** |
| 3 | `03_teardown.py` | no (removes them) |
| | `agent/agent.py` | the file that gets deployed |

## The four lines

`agent/agent.py` is a normal LangGraph agent plus:

```python
from bedrock_agentcore.runtime import BedrockAgentCoreApp   # 1. import
app = BedrockAgentCoreApp()                                 # 2. initialise

@app.entrypoint                                             # 3. decorate
def handler(payload: dict) -> dict: ...

app.run(port=8080)                                          # 4. serve
```

That is the entire delta between an agent on your laptop and an agent with an
HTTPS endpoint, session isolation, autoscaling and CloudWatch traces. The
graph, the state schema and the tools are the same objects from §4.

The decorated function **is** the HTTP handler: AgentCore turns the POSTed
JSON body into `payload` and serialises whatever you return.

## 01 — local first

```bash
uv run section_6_agentcore/01_run_local.py
```

Starts `agent/agent.py` on `localhost:8080`, POSTs one request to
`/invocations`, prints the response, stops the server. Measured: **4.2s**, no
AWS resources created.

```json
{
  "answer": "...the most common substantive words are \"intake\", \"sites\"...",
  "word_count": 34,
  "top_words": ["intake", "sites", "throughput", "against", "changes"],
  "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0"
}
```

Do all your debugging here. In the cloud each round trip costs ~30 seconds.

## The lifecycle (slide 54)

| command | what it does | time |
|---|---|---|
| `configure` | writes `.bedrock_agentcore.yaml`, creates the IAM role and S3 bucket | seconds |
| `launch` | uploads source, builds the image, waits for READY | ~30s |
| `invoke` | POSTs JSON to the endpoint, returns the handler's dict | ~8s |
| `status` | reports runtime and endpoint states separately | instant |
| `destroy` | removes the runtime, S3 artefacts and any auto-created role | ~15s |

```bash
uv run section_6_agentcore/02_deploy.py             # configure + launch + invoke
uv run section_6_agentcore/02_deploy.py --status    # check it
uv run section_6_agentcore/02_deploy.py --invoke-only

uv run section_6_agentcore/03_teardown.py           # DRY RUN — read the list
uv run section_6_agentcore/03_teardown.py --yes     # actually delete
uv run section_6_agentcore/03_teardown.py --list    # every runtime in the region
```

## ⚠ This section creates real, billable AWS resources

Every other lab in this course is an API call that finishes. `02_deploy.py`
leaves an AgentCore runtime, an IAM role and an S3 upload sitting in your
account until you delete them. **A runtime left in READY bills whether or not
anything is calling it.**

- The runtime is named per-user (`agentcore_lab_$USER`) so two people on one
  account do not collide. Override with `--agent` or `AGENTCORE_AGENT_NAME`.
- `03_teardown.py` is a **dry run unless you pass `--yes`**. Run it that way
  first and read the list.
- `02_deploy.py` prints the exact teardown command as its last line.
- `--list` shows every AgentCore runtime in the region, including ones this
  checkout does not know about. That is the command to check when you suspect
  something was left behind — `.bedrock_agentcore.yaml` is just a file and can
  be out of date.

**Teardown is not complete.** `destroy` removes the runtime and its deployment
resources. Shared infrastructure — S3 buckets, ECR repositories, CloudWatch log
groups — can survive depending on configuration. Check the console at the end
of the session.

## Two CLIs exist

The verbs above are the **starter toolkit**, which matches the
`.bedrock_agentcore.yaml` file these scripts write. The newer **AgentCore CLI**
uses `create`, `dev`, `deploy` and `invoke` instead. Confirm which one is
installed before following any runbook you find online — the file layouts are
not interchangeable.

## agent/ is self-contained on purpose

AgentCore uploads the whole source path, so the agent must not import from the
lab's other files. That is why `agent/agent.py` repeats the model factory
instead of importing `_common`, and why `agent/requirements.txt` is separate
and minimal — every line is install time on every deploy, and anything unused
is attack surface in something holding an execution role.

Note what is **absent** from it: the starter toolkit. That is a deploy-time
tool that runs on your laptop; the agent in the cloud never needs it.

## No credentials in the deployed agent

`agent/agent.py` contains no keys. In the cloud the runtime assumes its
execution role and boto3's default chain finds them — which is exactly why the
deploy step has to grant that role `bedrock:InvokeModel`.
