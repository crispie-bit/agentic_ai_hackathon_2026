# §3 — Bedrock, from the raw call up

Five files, in order. Each adds exactly one idea to the file before it, and
every file runs on its own.

```bash
cd agentic_teaching/lab
uv sync --extra aws
aws sso login --profile <your-profile>      # see ../AWS_SETUP.md
uv run 00_check_bedrock.py                  # do this before anything below
```

Then, in order:

| | File | Adds |
|---|---|---|
| 1 | `01_invoke_model.py` | the four required body keys; the streaming response |
| 2 | `02_tool_definition.py` | a tool is a JSON Schema plus a sentence — against a live API |
| 3 | `03_tool_call_end_to_end.py` | the four steps of one complete tool call |
| 4 | `04_react_loop.py` | wrap it in a `for`, and you have an agent |
| 5 | `05_cost_and_optimisation.py` | measure, estimate, price, then four ways to cut it |

## The through-line

```
01  one call                     messages in, one message out
02  ...with a tool defined       stop_reason becomes "tool_use"
03  ...and the tool actually run  YOUR code runs it, then you append the result
04  ...in a loop                 the model chains tools until it is done
05  ...and paid for              what it cost, and the four levers that cut it
```

By file 04 you have written an agent with no framework at all. That is the
point of the section: §4's LangGraph replaces the loop in 04 with a graph, and
nothing conceptual changes.

## The tools are real

Files 02 and 03 call the public [Open-Meteo](https://open-meteo.com) API — no
key, no signup — and return the actual temperature right now. The model
supplies the coordinates from its own knowledge and gets the temperature from
the tool, which is the division of labour in one line: **knowledge from the
model, facts from the tool.**

That needs outbound internet. If you are behind a proxy that blocks it, the
tool returns its error as text rather than raising, so the files still run and
still teach the mechanism — you just get an error string where a temperature
should be.

File 04 uses an invented warehouse instead, deliberately: its `place_order`
tool has a **side effect**, which is what makes the "the model asked, your code
acted" point land.

## Two things that will bite you

**`system` is not a message.** On Groq (Session 1) the system prompt is
`{"role": "system", ...}` inside `messages`. On Bedrock it is a **top-level
`system` key**, and sending the Groq shape returns:

```
ValidationException: messages: Unexpected role "system". The Messages API
accepts a top-level `system` parameter, not "system" as an input message role.
```

`messages` carries the conversation, whose roles alternate user/assistant. A
standing instruction is not a turn in it. Where the system prompt goes:

| API | Where |
|---|---|
| InvokeModel (Bedrock) | `"system": "..."` — top-level key |
| Converse (Bedrock) | `system=[{"text": "..."}]` — separate argument |
| OpenAI / Groq | `{"role": "system", ...}` — a message |

**`stop_reason` drives the loop, not the text.** `"tool_use"` means the model
is asking for a tool and is not finished. `"end_turn"` means it is. Never
decide by string-matching the reply.

## InvokeModel vs Converse

These files use `InvokeModel`, which passes Anthropic's own JSON through
untouched. Bedrock also has `Converse`, which normalises the shape across
vendors — same client, different `modelId`, whether it is Anthropic, Meta,
Mistral or Amazon Nova.

| | InvokeModel | Converse |
|---|---|---|
| body | a JSON string you build | normal Python kwargs |
| `anthropic_version` | required | not needed |
| system prompt | top-level `"system"` | `system=[{"text": ...}]` |
| limits | `"max_tokens"` | `inferenceConfig.maxTokens` |
| reading the reply | `["body"].read()` | plain dict |
| stop field | `stop_reason` | `stopReason` |
| changing vendor | rewrite the body | change `modelId` |

Prefer Converse in your own work. The reason to learn InvokeModel is that
error messages, older code and most vendor documentation are written in its
shape — and `ChatBedrockConverse` in §4 is a wrapper over Converse, so knowing
what it wraps is how you debug it.

## Cost

Every file costs real money — cents, not zero. File 05 prints an estimate for
its own run and extrapolates to a workload. Rates live in one marked constant
at the top of that file and are probably out of date; check
<https://aws.amazon.com/bedrock/pricing> before quoting a number to anyone.

Set an AWS Budget alert before the hackathon, not after.

## No client data

The warehouse, product codes and stock figures in file 04 are invented. The
weather is real.
