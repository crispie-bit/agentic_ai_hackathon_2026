"""Command line entry point for the project agent."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .agent import ProjectAgent
from .classifiers import classifier_from_environment
from .evaluation import DEFAULT_EVALUATION_PATH, evaluate, load_cases
from .knowledge import KnowledgeBase
from .models import CustomerSignal, Ticket
from .server import run_server
from .storage import JSONLDecisionStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Ruiheng project starter agent.")
    parser.add_argument("ticket", nargs="?", help="Customer ticket or user request text.")
    parser.add_argument("--tier", default="standard", help="Customer tier, e.g. standard, vip, enterprise.")
    parser.add_argument("--region", default=None, help="Optional customer region.")
    parser.add_argument("--channel", default="chat", help="Source channel for the request.")
    parser.add_argument("--knowledge", type=Path, help="JSON or JSONL knowledge article file.")
    parser.add_argument("--history", type=Path, help="Append decisions to this JSONL audit log.")
    parser.add_argument("--llm-classifier", action="store_true", help="Use env-configured LLM classifier.")
    parser.add_argument("--evaluate", nargs="?", const=DEFAULT_EVALUATION_PATH, type=Path, help="Run evaluation cases.")
    parser.add_argument("--serve", action="store_true", help="Start the HTTP API server.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP API host.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP API port.")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    agent = build_agent(args)
    indent = None if args.compact else 2

    if args.serve:
        run_server(args.host, args.port, agent)
        return 0

    if args.evaluate:
        report = evaluate(agent, load_cases(args.evaluate))
        print(json.dumps(report.to_dict(), indent=indent, sort_keys=True))
        return 0

    if not args.ticket:
        build_parser().error("ticket is required unless --evaluate or --serve is used")

    customer = CustomerSignal(tier=args.tier, region=args.region)
    decision = agent.handle(Ticket(args.ticket, customer=customer, channel=args.channel))
    print(json.dumps(decision.to_dict(), indent=indent, sort_keys=True))
    return 0


def build_agent(args: argparse.Namespace) -> ProjectAgent:
    knowledge = KnowledgeBase.from_path(args.knowledge) if args.knowledge else KnowledgeBase.default()
    classifier = classifier_from_environment() if args.llm_classifier else None
    store = JSONLDecisionStore(args.history) if args.history else None
    return ProjectAgent(knowledge_base=knowledge, classifier=classifier, decision_store=store)
