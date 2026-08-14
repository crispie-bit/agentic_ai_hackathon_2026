# §2 — Agentic AI Basics  (slides 15–25)

Six files, easiest first. §1 established the single model call; this section
adds the four things that turn a call into an agent — **tools, memory,
reasoning, action** — and the loop that coordinates them.

Nothing here is a framework. Every loop is a `for` you can read in one screen.
§3 rewrites the last one against the raw Bedrock Converse API, and §4 hands it
to LangGraph.

```bash
cd agentic_teaching/lab
uv run section_2_agentic_ai_basic/00_anatomy_of_an_agent.py   # no key needed
uv run section_2_agentic_ai_basic/01_memory.py
uv run section_2_agentic_ai_basic/02_tools.py
uv run section_2_agentic_ai_basic/03_tool_composition.py
uv run section_2_agentic_ai_basic/04_planning.py
uv run section_2_agentic_ai_basic/05_agent_lab.py             # the lab
```

| | File | Slides | Difficulty | Format |
|---|---|---|---|---|
| 0 | `00_anatomy_of_an_agent.py` | 15, 16, 17 | ⬤◯◯◯◯ | read & run — **no API key, no cost** |
| 1 | `01_memory.py` | 22 | ⬤⬤◯◯◯ | read & run |
| 2 | `02_tools.py` | 23 | ⬤⬤◯◯◯ | read & run |
| 3 | `03_tool_composition.py` | 24 | ⬤⬤⬤◯◯ | read & run |
| 4 | `04_planning.py` | 18, 19, 20, 21 | ⬤⬤⬤⬤◯ | read & run |
| 5 | `05_agent_lab.py` | 25 | ⬤⬤⬤⬤⬤ | **hands-on**, 4 TODOs |

Files 0–4 you run and read. File 5 you write, and it assembles every piece
from the ones before it.

> The slide order puts planning (18–21) before memory and tools. The files
> reorder it, because planning is much easier to follow once you have seen a
> tool call and a transcript. Run them in file order; the slide numbers are on
> each file so you can jump back.

## Which provider

**One line in `lab/.env` decides**, and no file in this folder names a
provider. Everything below goes through `chat_model()` in `lab/_common.py`.

```bash
# lab/.env

# --- Groq (default: free tier, no AWS) -------------------------------
GROQ_API_KEY=gsk_...
# GROQ_MODEL=llama-3.3-70b-versatile      # only if the default id breaks
# LLM_PROVIDER=bedrock                    # <- commented out = Groq

# --- Bedrock ---------------------------------------------------------
# Uncomment the line above, then:
AWS_PROFILE=sentia
AWS_DEFAULT_REGION=ap-southeast-1
# BEDROCK_MODEL=global.anthropic.claude-haiku-4-5-20251001-v1:0
```

Switch for one run without editing anything:

```bash
LLM_PROVIDER=bedrock uv run section_2_agentic_ai_basic/02_tools.py
```

Bedrock also needs the extras and a live session:

```bash
uv sync --extra aws
aws sso login --profile sentia
```

Same files, same output, different bill. That the diff is one line is the
point — it is what `chat_model()` buys you.

## What each file is for

**00 · Anatomy of an agent** — the five stages of slide 16, printed as they
happen. The "model" is scripted Python, so there is no key, no network and no
cost; the only thing on screen is the shape of the loop. Establishes the
sentence the rest of the section rests on: **the model chooses, your code does
everything else.**

**01 · Memory** — three parts. No memory (the model was never told, it did not
forget), short-term memory (the transcript, resent whole, and the token count
climbing to prove it), long-term memory (a store, a search, an insert). The
takeaway: long-term memory is a *retrieval* problem, and whatever the search
misses may as well not exist.

**02 · Tools** — the two halves of slide 23. Prints the actual JSON schema the
model receives, then walks one round trip by hand: the reply carries a
*request*, you execute it, you append a `ToolMessage` with the matching id.
Ends on the half nobody demos — what happens when the arguments are wrong.

**03 · Tool composition** — sequential, parallel, conditional. Each tool sleeps
0.8s so the latency claims on the slide become numbers: parallel execution
halves the wall clock and changes the token bill by exactly zero. The
conditional branch is a plain `if`, which is the recommendation.

**04 · Planning** — the same goal three ways, with a comparison table at the
end. Decomposition plans first and never looks back. Reactive never plans.
Hierarchical does both, and is what production agents converge on. Watch the
token column: on a small goal, planning is pure overhead.

**05 · The lab** — four TODOs, one line each. The file runs before you touch
it and gets the answer wrong in a useful way: nothing is appended to the
transcript, so the model is asked the identical question six times and the cap
stops it. Fix them in order and re-run between each.

- **TODO 1** append the reply — without it the model never sees its own request
- **TODO 2** stop when no tool was requested — the *normal* exit
- **TODO 3** execute the tool and append the result **with its `tool_call_id`**
- **TODO 4** report *why* the loop ended — "finished" and "was stopped" both
  return text, and only one of them is an answer

`05_agent_lab_solution.py` is the finished version. Read it after you have
made each TODO fail at least once — it also adds the two things the TODOs
leave out: a try/except that turns a crash into an observation, and a check
for a tool name that does not exist.

## Things worth arguing about in the room

- **The cap is not a detail.** The structure of an agent guarantees nothing
  about termination. Either the model stops asking for tools, or `MAX_STEPS`
  stops it. There is no third option.
- **Docstrings are prompt.** Change one in `02_tools.py` and tool selection
  gets worse without a line of logic changing.
- **A tool error the model can read is a tool error it can recover from.**
  Return the failure as a string; let an exception escape and the run just dies.
- **You pay for the transcript on every turn.** Print `len(messages)` at each
  step of file 05 and multiply.

## After this section

§3 writes the file-05 loop against `bedrock-runtime` directly, where the same
three ideas show up as `stopReason`, `toolUse` and `toolResult`. §4 hands the
loop to LangGraph and you stop writing it at all.
