from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_system.services.preparation import WorkspacePreparation


def run_course_specialist(question: str, preparation: WorkspacePreparation, model: Any) -> str:
    """Ask the course specialist to inspect only course-source records."""
    records = preparation.store.search(question, limit=8)
    course_records = [record for record in records if "course" in record["source_type"] or "ntulearn" in record["source_type"]]
    if not course_records:
        course_records = [record for record in preparation.store.search("deadline assignment lecture", limit=8) if "course" in record["source_type"] or "ntulearn" in record["source_type"]]
    return _summarise(
        "course specialist",
        question,
        course_records,
        model,
        "Extract course deadlines, assignments, lectures, and required preparation. Ignore unrelated inbox or calendar records.",
    )


def run_inbox_specialist(question: str, preparation: WorkspacePreparation, model: Any) -> str:
    """Ask the inbox specialist to inspect only inbox-source records."""
    records = preparation.store.search(question, limit=8)
    inbox_records = [record for record in records if "inbox" in record["source_type"] or "outlook" in record["source_type"]]
    return _summarise(
        "inbox specialist",
        question,
        inbox_records,
        model,
        "Extract actionable messages, meetings, reminders, and requested follow-ups. Ignore course records.",
    )


def _summarise(name: str, question: str, records: list[dict[str, Any]], model: Any, instruction: str) -> str:
    context = json.dumps(records)
    if not records:
        return f"{name}: no relevant records found."
    response = model.invoke([
        SystemMessage(
            f"You are the {name}. {instruction} Use only the supplied records. "
            "Return concise factual findings with source titles; do not invent details."
        ),
        HumanMessage(f"Question: {question}\nRecords: {context}"),
    ])
    return f"{name}: {str(response.content).strip()}"
