# §2 — Agentic AI Basics  (slides 15–26, then 27–34 for prompting)

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
uv run section_2_agentic_ai_basic/06_prompt_engineering.py    # the third lab
```

| | File | Slides | Difficulty | Format |
|---|---|---|---|---|
| 0 | `00_anatomy_of_an_agent.py` | 15, 16 | ⬤◯◯◯◯ | read & run — **no API key, no cost** |
| 1 | `01_memory.py` | 21 | ⬤⬤◯◯◯ | read & run |
| 2 | `02_tools.py` | 23 | ⬤⬤◯◯◯ | read & run |
| 3 | `03_tool_composition.py` | 22 | ⬤⬤⬤◯◯ | read & run |
| 4a | `04a_planning_decomposition.py` | 18 | ⬤⬤◯◯◯ | read & run |
| 4b | `04b_planning_reactive.py` | 19 | ⬤⬤◯◯◯ | read & run |
| 4 | `04_planning.py` | 17, 18, 19, 20 | ⬤⬤⬤⬤◯ | read & run |
| 5 | `05_agent_lab.py` | 24, 25, 26 | ⬤⬤⬤⬤⬤ | **hands-on**, 4 TODOs |
| 5s | `05_agent_lab_solution.py` | 24 | — | the four TODOs filled in |
| 6a | `06a_zero_shot.py` | 28 | ⬤◯◯◯◯ | read & run |
| 6b | `06b_few_shot.py` | 29 | ⬤⬤◯◯◯ | read & run |
| 6c | `06c_cot.py` | 30 | ⬤⬤◯◯◯ | read & run |
| 6d | `06d_rag.py` | 31 | ⬤⬤⬤⬤◯ | **hands-on**, 2 TODOs |
| 6 | `06_prompt_engineering.py` | 33, 34 | ⬤⬤⬤⬤◯ | **hands-on**, 3 TODOs |
| 6s | `06_prompt_engineering_solution.py` | 34 | — | worked, with the reasoning |

Files 0–4 you run and read. File 5 you write, and it assembles every piece
from the ones before it. The 06 files come after, because prompting only
becomes interesting once you have seen a tool call, a transcript and a loop.

The deck splits this folder across two lab blocks: **LAB 02** (slides 25–26)
is files 00–05, and **LAB 03** (slides 33–34) is the 06 files.

> The slide order puts planning (17–20) before memory (21) and tools (22–23).
> The files reorder it, because planning is much easier to follow once you have
> seen a tool call and a transcript. Run them in file order; the slide numbers
> are on each file so you can jump back.

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

**00 · Anatomy of an agent** — the four parts of slide 16, printed as they
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

## 06 · Prompting — four techniques, then a graded lab

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

**06_prompt_engineering.py** *(hands-on, 3 TODOs)* — 06a, 06b and 06c again,
with the training wheels off and a grader attached. Three challenges, each a
fixed test set with the answers already written down:

| **TODO 1** · zero-shot | 8 tickets, 8/8 to pass | the label set is not given — read the test data, work out the three labels, and name them in the instruction |
| **TODO 2** · few-shot | 6 tickets, 6/6 to pass | the instruction is **locked**; you may only choose 5 example rows out of the 7 in `TRAINING_DATA` |
| **TODO 3** · chain of thought | 4 word problems, 4/4 to pass | the grader reads one line, `TOTAL: <number>`, and ignores every other word |

Nothing here is marked out of ten by eye. You edit one string, re-run, and
watch a count move — which is the only honest way to tell a prompt that works
from a prompt that reads well.

TODO 2 is the one to slow down on, and it prints its own baseline: challenge 2
runs **twice**, once with no examples at all and once with yours.

```
--- 2a. FEW-SHOT, NO examples (the baseline to beat) ---
  FAIL  got=billing   want=security   I see a payment I never made on my sta
  [#####.] 5/6
--- 2b. FEW-SHOT, your examples ---
  [######] 6/6
```

The locked instruction already gets 5 of 6 on its own, so the entire challenge
rides on one ticket — *"I see a payment I never made on my statement"*, which
the instruction alone always calls `billing`. That is few-shot in one screen:
the examples reach a boundary the instruction never states, and you can watch
exactly which ticket moved.

One training row draws that boundary explicitly — *"There is a charge from a
device I don't recognise"* → `security`, the only row where money and security
collide. How much it matters depends on the model, and it is worth saying so in
the room. Testing all 21 legal picks against the full `TEST_2`: on **Groq /
gpt-oss, 20 of 21 pass**, including picks without that row, so most students
will clear it first try. On **Bedrock / Haiku only 13 of 21 pass, and every one
of them contains that row** — no pick without it ever passes. The lift from the
baseline is the part that holds on both.

TODO 3 is not the lesson its name suggests. The starter prompt gets all four
totals *right* on both providers and still scores 0/4. What the two models do
with "reply with the final number only" is opposite — Groq's gpt-oss obeys to
the letter and answers `62.70`; Bedrock's Haiku ignores it and prints a page of
markdown working — and neither produces a line the grader can read. Four
correct answers, zero points, twice, for opposite reasons. The challenge is an
output contract, not arithmetic, and the obedient model failing is the point:
an answer your code cannot parse is not an answer.

`play` runs the classifier against tickets you type:

```bash
uv run section_2_agentic_ai_basic/06_prompt_engineering.py play
```

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
