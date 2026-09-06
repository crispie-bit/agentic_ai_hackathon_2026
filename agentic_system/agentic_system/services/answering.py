from __future__ import annotations

import os

from pydantic import BaseModel, Field

from agentic_system.config import GROQ_MODEL
from agentic_system.services.preparation import WorkspacePreparation
from agentic_system.services.workday_planner import rank_tasks


class LeadDecision(BaseModel):
    task_title: str = Field(description="One exact task title from the supplied workspace tasks")
    reason: str = Field(description="A concise reason grounded in deadline, effort, or priority")


def answer_mode() -> str:
    """Report whether questions use the model or the offline planner."""
    return "Groq reasoning" if os.getenv("GROQ_API_KEY") else "Offline planner"


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
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq

    tasks = rank_tasks(preparation.get_tasks())
    context = "\n".join(
        f"- {task.title}; due {task.due_label}; effort {task.estimated_minutes} minutes; priority {task.priority}; action: {task.action}"
        for task in tasks
    ) or "No structured tasks are currently available."
    model = ChatGroq(model=GROQ_MODEL, temperature=0, max_tokens=300).with_structured_output(LeadDecision)
    decision = model.invoke([
        SystemMessage(
            "You are the lead student workday agent. Use only the supplied workspace facts. "
            "Choose exactly one task title from the list. Do not invent tasks or dates. "
            "Return a concise reason based on deadline, effort, or priority."
        ),
        HumanMessage(f"Workspace tasks:\n{context}\n\nStudent question: {question}"),
    ])
    known_tasks = {task.title: task for task in tasks}
    task = known_tasks.get(decision.task_title)
    if task is None:
        raise ValueError("Model selected a task that was not in the workspace")
    return (
        f"**Recommendation:** Start with **{task.title}** — due {task.due_label}.\n\n"
        f"**Why:** {decision.reason}\n\n"
        f"**Action:** {task.action}"
    )


def answer_question(question: str, preparation: WorkspacePreparation) -> str:
    """Return a model-backed grounded response with an offline fallback."""
    if os.getenv("GROQ_API_KEY"):
        try:
            return _model_answer(question, preparation)
        except Exception:
            pass
    return _offline_answer(question, preparation)
