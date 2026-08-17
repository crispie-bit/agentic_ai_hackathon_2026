"""
§3 · 03 — Token accounting: what the call actually cost.

    uv run section_3_bedrock/03_token_costs.py

01 showed you the raw call. This one is about the `usage` block that comes back
with it, because that block is your invoice.

Four counters, not two. `input_tokens` does NOT include cached prompt tokens —
those are billed separately, at a premium on the write and at a discount on
every read. Price the call off all four or you will under-report every time you
turn caching on.

No arguments. Edit QUESTION below, or export BEDROCK_PROMPT.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import _bootstrap  # noqa: F401  — must come first; puts lab/ on sys.path

from _common import MODEL_ID, REGION, banner, bedrock_runtime, report_usage

ANTHROPIC_VERSION = "bedrock-2023-05-31"
MAX_TOKENS = 300
TEMPERATURE = 0

QUESTION = os.environ.get("BEDROCK_PROMPT", "In one sentence: what is prompt caching?")

# USD per MILLION tokens. Model- and region-specific, and they change. Override
# with BEDROCK_RATES_FILE (a JSON file with any of these four keys) rather than
# editing this dict, so the numbers below are never quietly wrong.
DEFAULT_RATES: dict[str, float] = {
    "input": 1.00,
    "output": 5.00,
    "cache_write": 1.25,   # first call: more expensive than plain input
    "cache_read": 0.10,    # every call after: ~10x cheaper
}

TOKEN_FIELDS = ("input", "output", "cache_write", "cache_read")

Message = dict[str, Any]


def load_rates(path: str | os.PathLike[str] | None = None) -> dict[str, float]:
    """DEFAULT_RATES, overlaid with $BEDROCK_RATES_FILE if it is set."""
    path = path or os.environ.get("BEDROCK_RATES_FILE")
    rates = dict(DEFAULT_RATES)
    if path:
        rates.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return rates


@dataclass(frozen=True)
class Usage:
    """The four counters, normalised off Bedrock's wire names."""

    input: int = 0
    output: int = 0
    cache_write: int = 0
    cache_read: int = 0

    @classmethod
    def from_payload(cls, usage: Mapping[str, Any] | None) -> "Usage":
        # `or 0` because Bedrock omits the cache keys entirely when caching is
        # off, and has been known to send them back as null.
        usage = usage or {}
        return cls(
            input=usage.get("input_tokens", 0) or 0,
            output=usage.get("output_tokens", 0) or 0,
            cache_write=usage.get("cache_creation_input_tokens", 0) or 0,
            cache_read=usage.get("cache_read_input_tokens", 0) or 0,
        )

    @property
    def billed_input(self) -> int:
        """Every prompt token you paid for, cached or not."""
        return self.input + self.cache_write + self.cache_read

    def cost(self, rates: Mapping[str, float] | None = None) -> float:
        rates = rates or DEFAULT_RATES
        return sum(
            getattr(self, f) * rates.get(f, 0.0) for f in TOKEN_FIELDS
        ) / 1_000_000

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(*(getattr(self, f) + getattr(other, f) for f in TOKEN_FIELDS))

    def as_dict(self) -> dict[str, int]:
        return {f: getattr(self, f) for f in TOKEN_FIELDS}


class BedrockClient:
    """invoke_model / invoke_model_with_response_stream, with a running tab."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        rates: Mapping[str, float] | None = None,
        client: Any = None,
        verbose: bool = True,
    ) -> None:
        self.model_id = model_id
        self.rates = dict(rates) if rates else load_rates()
        self._client = client or bedrock_runtime()
        self.verbose = verbose
        self.totals = Usage()
        self.last_usage = Usage()
        self.last_text = ""
        self.calls = 0

    # -- request -------------------------------------------------------------

    @staticmethod
    def _messages(prompt: str | Sequence[Message]) -> list[Message]:
        """Accept a bare string or a full messages list (for multi-turn)."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return [dict(m) for m in prompt]

    def _body(
        self,
        prompt: str | Sequence[Message],
        system: str | None = None,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        cache_system: bool = False,
    ) -> str:
        body: dict[str, Any] = {
            "anthropic_version": ANTHROPIC_VERSION,
            "messages": self._messages(prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            # `system` as a LIST of blocks, not a string — that is the only
            # shape that can carry cache_control.
            block: dict[str, Any] = {"type": "text", "text": system}
            if cache_system:
                block["cache_control"] = {"type": "ephemeral"}
            body["system"] = [block]
        return json.dumps(body)

    def _record(self, raw: Mapping[str, Any] | None, label: str) -> Usage:
        usage = Usage.from_payload(raw)
        self.totals += usage
        self.last_usage = usage
        self.calls += 1
        if self.verbose:
            report_usage(label, dict(raw or {}))
            print(f"  cost:        ${usage.cost(self.rates):.6f}")
        return usage

    # -- calls ---------------------------------------------------------------

    def invoke(
        self,
        prompt: str | Sequence[Message],
        system: str | None = None,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        cache_system: bool = False,
    ) -> tuple[str, Usage]:
        response = self._client.invoke_model(
            modelId=self.model_id,
            body=self._body(prompt, system, max_tokens, temperature, cache_system),
            contentType="application/json",
            accept="application/json",
        )
        envelope = json.loads(response["body"].read())
        text = "".join(
            block.get("text", "")
            for block in envelope.get("content", [])
            if block.get("type") == "text"
        )
        self.last_text = text
        return text, self._record(envelope.get("usage"), "invoke_model")

    def stream(
        self,
        prompt: str | Sequence[Message],
        system: str | None = None,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        cache_system: bool = False,
    ) -> Iterator[str]:
        """Yield text deltas; the tab is settled when the stream is exhausted.

        Usage arrives SPLIT across two events: prompt and cache counts ride on
        message_start, output tokens on message_delta. Read only the second and
        every streamed call looks like the prompt was free.
        """
        response = self._client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=self._body(prompt, system, max_tokens, temperature, cache_system),
            contentType="application/json",
            accept="application/json",
        )
        raw: dict[str, Any] = {}
        parts: list[str] = []
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            kind = chunk.get("type")
            if kind == "content_block_delta":
                delta = chunk.get("delta", {}).get("text", "")
                if delta:
                    parts.append(delta)
                    yield delta
            elif kind == "message_start":
                raw.update(chunk.get("message", {}).get("usage") or {})
            elif kind == "message_delta":
                raw.update(chunk.get("usage") or {})
        self.last_text = "".join(parts)
        self._record(raw, "invoke_model_with_response_stream")

    # -- reporting -----------------------------------------------------------

    def report(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "region": REGION,
            "calls": self.calls,
            "tokens": self.totals.as_dict(),
            "billed_input": self.totals.billed_input,
            "cost_usd": round(self.totals.cost(self.rates), 6),
        }


# A system prompt only gets cached if it clears the model's minimum — roughly
# 1k tokens for Sonnet, more for Haiku. Below that the API silently ignores
# cache_control and both cache counters stay 0. That is the expected result,
# not a broken script, which is why this one is padded to be long enough.
LONG_SYSTEM = (
    "You are a terse assistant for an AWS cost-engineering course. "
    "Answer in one sentence. Never speculate about pricing figures.\n\n"
    + "Reference note: token accounting on Bedrock is per-model and per-region, "
    "and cached prompt tokens are billed on a separate line from ordinary input "
    "tokens, so any cost model that reads only input_tokens and output_tokens "
    "will understate a cached workload.\n" * 40
)


def main() -> None:
    banner("token accounting — one call")
    print(f"  model:  {MODEL_ID}")
    print(f"  region: {REGION}")
    print(f"  prompt: {QUESTION}\n")

    client = BedrockClient()

    text, usage = client.invoke(QUESTION)
    print(f"  text:        {text.strip()}")
    print(f"  billed in:   {usage.billed_input} (input + cache_write + cache_read)")

    banner("the same system prompt, twice — watch the cache counters")
    # Call 1 pays cache_write. Call 2 pays cache_read, ~10x less, and its
    # input_tokens drops to just the new user turn.
    for label in ("write", "read "):
        print(f"\n  pass ({label.strip()}):")
        client.invoke(QUESTION, system=LONG_SYSTEM, cache_system=True)

    banner("streaming — usage arrives in pieces")
    for delta in client.stream("Count from 1 to 10, one number per line."):
        print(delta, end="", flush=True)
    print()

    banner("the tab")
    print(json.dumps(client.report(), indent=2))


if __name__ == "__main__":
    main()