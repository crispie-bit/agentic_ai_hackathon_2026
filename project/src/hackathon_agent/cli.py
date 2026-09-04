"""Command line entry point for the project agent."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .agent import ProjectAgent
from .models import CustomerSignal, Ticket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Ruiheng project starter agent.")
    parser.add_argument("ticket", help="Customer ticket or user request text.")
    parser.add_argument("--tier", default="standard", help="Customer tier, e.g. standard, vip, enterprise.")
    parser.add_argument("--region", default=None, help="Optional customer region.")
    parser.add_argument("--channel", default="chat", help="Source channel for the request.")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    customer = CustomerSignal(tier=args.tier, region=args.region)
    decision = ProjectAgent().handle(Ticket(args.ticket, customer=customer, channel=args.channel))
    indent = None if args.compact else 2
    print(json.dumps(decision.to_dict(), indent=indent, sort_keys=True))
    return 0

