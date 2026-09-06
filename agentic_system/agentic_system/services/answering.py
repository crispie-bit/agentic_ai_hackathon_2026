from __future__ import annotations

import os

from agentic_system.config import GROQ_MODEL
from agentic_system.services.lead_loop import run_lead_loop
from agentic_system.services.preparation import WorkspacePreparation
from agentic_system.services.workday_planner import rank_tasks


def answer_mode() -> str:
    """Report whether questions use the model or the offline planner."""
    return "Groq lead + specialist agents" if os.getenv("GROQ_API_KEY") else "Offline planner"


def _offline_answer(question: str, preparation: WorkspacePreparation) -> str:
    tasks = rank_tasks(preparation.get_tasks())
    lowered = question.lower()
    if tasks and any(word in lowered for word in ("next", "priorit", "should i", "plan")):
        task = tasks[0]
        return (
            f"**Recommendation:** Start with **{task.title}**.\n\n"
            f"**Why:** It has the highest priority and needs about {task.estimated_minutes} minutes. "
            f"It is due {task.due_label}.\n\n"
            f"**Action:** {task.action}"
        )
    matches = preparation.store.search(question)
    if matches:
        lines = [
            f"**{item['title']}** ({item['source_type']}): {' '.join(item['content'].split())[:280]}"
            for item in matches[:5]
        ]
        return "**Relevant workspace information:**\n\n" + "\n\n".join(lines)
    return "I could not find a relevant task or source record. Try asking about a deadline, Monday, or what to do next."


def _model_answer(question: str, preparation: WorkspacePreparation) -> str:
    from langchain_groq import ChatGroq

    model = ChatGroq(model=GROQ_MODEL, temperature=0, max_tokens=300)
    return run_lead_loop(question, preparation, model)


def answer_question(question: str, preparation: WorkspacePreparation) -> str:
    """Return a model-backed grounded response with an offline fallback."""
    if os.getenv("GROQ_API_KEY"):
        try:
            return _model_answer(question, preparation)
        except Exception:
            pass
    return _offline_answer(question, preparation)
