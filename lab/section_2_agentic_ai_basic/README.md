# §2 — Agentic AI Basics  (slides 15–25)

Easiest first. §1 established the single model call; this section adds the four
things that turn a call into an agent — **tools, memory, reasoning, action** —
the loop that coordinates them, and then the prompting that decides what the
loop actually does.

Nothing here is a framework. Every loop is a `for` you can read in one screen.
§3 rewrites the last one against the raw Bedrock Converse API, and §4 hands it
to LangGraph.

```bash
cd agentic_teaching/lab
uv run section_2_agentic_ai_basic/00_anatomy_of_an_agent.py   # no key needed
uv run section_2_agentic_ai_basic/01_memory.py
uv run section_2_agentic_ai_basic/02_tools.py
uv run section_2_agentic_ai_basic/03_tool_composition.py
uv run section_2_agentic_ai_basic/04a_planning_decomposition.py
uv run section_2_agentic_ai_basic/04b_planning_reactive.py
uv run section_2_agentic_ai_basic/04_planning.py
uv run section_2_agentic_ai_basic/05_agent_lab.py             # the lab
uv run section_2_agentic_ai_basic/06a_zero_shot.py
uv run section_2_agentic_ai_basic/06b_few_shot.py
uv run section_2_agentic_ai_basic/06c_cot.py
uv run section_2_agentic_ai_basic/06d_rag.py                  # the second lab
uv run section_2_agentic_ai_basic/prompt_engineering.py
```

| | File | Slides | Difficulty | Format |
|---|---|---|---|---|
| 0 | `00_anatomy_of_an_agent.py` | 15, 16, 17 | ⬤◯◯◯◯ | read & run — **no API key, no cost** |
| 1 | `01_memory.py` | 22 | ⬤⬤◯◯◯ | read & run |
| 2 | `02_tools.py` | 23 | ⬤⬤◯◯◯ | read & run |
| 3 | `03_tool_composition.py` | 24 | ⬤⬤⬤◯◯ | read & run |
| 4a | `04a_planning_decomposition.py` | 18, 19 | ⬤⬤◯◯◯ | read & run |
| 4b | `04b_planning_reactive.py` | 20 | ⬤⬤◯◯◯ | read & run |
| 4 | `04_planning.py` | 18, 19, 20, 21 | ⬤⬤⬤⬤◯ | read & run |
| 5 | `05_agent_lab.py` | 25 | ⬤⬤⬤⬤⬤ | **hands-on**, 4 TODOs |
| 6a | `06a_zero_shot.py` | — | ⬤◯◯◯◯ | read & run |
| 6b | `06b_few_shot.py` | — | ⬤⬤◯◯◯ | read & run |
| 6c | `06c_cot.py` | — | ⬤⬤◯◯◯ | read & run |
| 6d | `06d_rag.py` | — | ⬤⬤⬤⬤◯ | **hands-on**, 2 TODOs |
| 6 | `prompt_engineering.py` | — | ⬤⬤⬤◯◯ | read & run, then 4 TODOs |

Files 0–4 you run and read. File 5 you write, and it assembles every piece
from the ones before it. The 06 files come after, because prompting only
becomes interesting once you have seen a tool call, a transcript and a loop.

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
# GROQ_MODEL=openai/gpt-oss-120b          # only if the default id breaks
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

**04a · Decomposition, on its own** — the smallest possible version. One model
call produces the whole step list before anything runs; the rest is a `for`
loop over that list. The output is in two halves, `PLAN` then `EXECUTE`, so the
split is impossible to miss.

**04b · Reactive, on its own** — same goal, same tools, no plan. Each step is
chosen from the transcript as it stands, and the number of steps is not known
until the run ends. Read it directly after 04a; the diff between the two files
*is* the lesson.

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

The two things the TODOs leave out are worth adding yourself once it runs: a
try/except that turns a crash into an observation, and a check for a tool name
that does not exist.

## 06 · Prompting — four techniques, then the agent version

The first four files are one technique each, on the same shape of task, so you
can diff them against one another:

**06a · Zero-shot** — an instruction and nothing else. Four support tickets,
one word back each. The baseline everything below is measured against.

**06b · Few-shot** — the *identical* instruction plus three fake conversation
turns you wrote both sides of. The examples teach a convention the instruction
never states (anything about logging in is `account`, not `technical`), and the
classification changes without a word of the instruction changing.

**06c · Chain of thought** — two arithmetic word problems with known answers,
asked twice: answer-only, then "show each calculation, end with
`ANSWER: <number>`". The parsed number is printed against the expected one, so
the accuracy difference is on screen rather than asserted. The forced final
line is the part that matters in production — reasoning you cannot parse is
reasoning you cannot use.

**06d · RAG** *(hands-on, 2 TODOs)* — knowledge instead of conversation. The
retrieval half is written from scratch — sentence chunking, a bag-of-words
vector, cosine similarity — so no vector DB or embedding model hides anything;
only generation is a real API call. Parts A–D run in order: retrieval alone,
same question with and without context, the full loop, then the failure case
where retrieval finds nothing useful. **TODO 1** changes `TOP_K`, **TODO 2**
changes `MIN_SCORE` — the two knobs that decide what the model is allowed to
see.

**prompt_engineering.py** — the same lesson, but inside an agent, where a
prompt is not only what you ask:

| the tool docstring | decides **which** tool gets called |
|---|---|
| the system prompt | decides **how much** the model does at once |
| the error string | decides **whether** the agent can recover |
| the examples | decide **what form** the answer takes |

Four experiments, four tasks, each run twice — working, then broken by a
one-line change — so the difference is visible rather than described. Four
`TODO(student)` blocks at the bottom ask you to break them yourself.

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
