"""Token usage and cost accounting for Amazon Bedrock (Anthropic Messages API).

Wraps ``invoke_model`` / ``invoke_model_with_response_stream`` and reports the
token usage and USD cost of every call.

Configuration (environment variables):
    BEDROCK_MODEL_ID    model id to invoke (required)
    AWS_REGION          region for the bedrock-runtime client (default us-east-1)
    BEDROCK_RATES_FILE  optional JSON file of per-million-token rates that
                        overrides DEFAULT_RATES, e.g.
                        {"input": 1.0, "output": 5.0,
                         "cache_write": 1.25, "cache_read": 0.1}

Usage:
    from bedrock_cost import BedrockClient

    client = BedrockClient()
    text, usage = client.invoke("Summarise this order.", system=SYSTEM_PROMPT)
    print(usage.cost(client.rates), client.totals)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import boto3

LOGGER = logging.getLogger(__name__)

ANTHROPIC_VERSION = "bedrock-2023-05-31"
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
TOKEN_FIELDS = ("input", "output", "cache_write", "cache_read")

# USD per million tokens. Rates are model- and region-specific and change over
# time; override with BEDROCK_RATES_FILE rather than editing this default.
DEFAULT_RATES: dict[str, float] = {
    "input": 1.00,
    "output": 5.00,
    "cache_write": 1.25,
    "cache_read": 0.10,
}


def load_rates(path: str | os.PathLike[str] | None = None) -> dict[str, float]:
    """Load per-million-token rates, falling back to DEFAULT_RATES."""
    path = path or os.environ.get("BEDROCK_RATES_FILE")
    rates = dict(DEFAULT_RATES)
    if path:
        rates.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return rates


@dataclass(frozen=True)
class Usage:
    """Token counts for one or more calls.

    Cached prompt tokens are not reported in ``input``; they appear under
    ``cache_write`` on the first call and ``cache_read`` thereafter. Cost must
    be computed from all four fields.
    """

    input: int = 0
    output: int = 0
    cache_write: int = 0
    cache_read: int = 0

    @classmethod
    def from_payload(cls, usage: Mapping[str, Any] | None) -> "Usage":
        usage = usage or {}
        return cls(
            input=usage.get("input_tokens", 0),
            output=usage.get("output_tokens", 0),
            cache_write=usage.get("cache_creation_input_tokens", 0),
            cache_read=usage.get("cache_read_input_tokens", 0),
        )

    @property
    def billed_input(self) -> int:
        return self.input + self.cache_write + self.cache_read

    def cost(self, rates: Mapping[str, float] | None = None) -> float:
        rates = rates or DEFAULT_RATES
        return sum(
            getattr(self, field) * rates.get(field, 0.0) for field in TOKEN_FIELDS
        ) / 1_000_000

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(*(getattr(self, f) + getattr(other, f) for f in TOKEN_FIELDS))

    def as_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in TOKEN_FIELDS}


class BedrockClient:
    """Bedrock runtime client that accumulates token usage across calls."""

    def __init__(
        self,
        model_id: str | None = None,
        region: str = DEFAULT_REGION,
        rates: Mapping[str, float] | None = None,
        client: Any = None,
    ) -> None:
        self.model_id = model_id or os.environ.get("BEDROCK_MODEL_ID")
        if not self.model_id:
            raise ValueError("model_id is required (set BEDROCK_MODEL_ID)")
        self.rates = dict(rates) if rates else load_rates()
        self._client = client or boto3.client("bedrock-runtime", region_name=region)
        self.totals = Usage()
        self.last_usage = Usage()
        self.calls = 0

    # -- request construction ------------------------------------------------

    def _body(
        self,
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
        cache_system: bool,
    ) -> str:
        body: dict[str, Any] = {
            "anthropic_version": ANTHROPIC_VERSION,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            block: dict[str, Any] = {"type": "text", "text": system}
            if cache_system:
                block["cache_control"] = {"type": "ephemeral"}
            body["system"] = [block]
        return json.dumps(body)

    def _record(self, usage: Usage) -> Usage:
        self.totals += usage
        self.calls += 1
        LOGGER.info(
            "bedrock call model=%s tokens=%s cost_usd=%.6f",
            self.model_id,
            usage.as_dict(),
            usage.cost(self.rates),
        )
        return usage

    # -- calls ---------------------------------------------------------------

    def invoke(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> tuple[str, Usage]:
        """Send a single prompt and return (text, usage)."""
        response = self._client.invoke_model(
            modelId=self.model_id,
            body=self._body(prompt, system, max_tokens, temperature, cache_system),
        )
        payload = json.loads(response["body"].read())
        text = "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        )
        return text, self._record(Usage.from_payload(payload.get("usage")))

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> Iterator[str]:
        """Yield text deltas. Usage is recorded once the stream is exhausted
        and is available as ``client.last_usage``."""
        response = self._client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=self._body(prompt, system, max_tokens, temperature, cache_system),
        )
        usage = Usage()
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                yield chunk["delta"].get("text", "")
            elif chunk["type"] == "message_delta":
                usage = Usage.from_payload(chunk.get("usage"))
        self.last_usage = self._record(usage)

    # -- reporting -----------------------------------------------------------

    def report(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "calls": self.calls,
            "tokens": self.totals.as_dict(),
            "cost_usd": round(self.totals.cost(self.rates), 6),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Invoke a Bedrock model and report cost.")
    parser.add_argument("prompt")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--system", default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = BedrockClient(model_id=args.model_id, region=args.region)

    if args.stream:
        for delta in client.stream(args.prompt, args.system, args.max_tokens):
            print(delta, end="", flush=True)
        print()
    else:
        text, _ = client.invoke(args.prompt, args.system, args.max_tokens)
        print(text)

    print(json.dumps(client.report(), indent=2))


if __name__ == "__main__":
    main()