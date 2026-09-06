from __future__ import annotations

import os

from agentic_system.config import AGENT_MODEL_IDS, GROQ_MODEL
from agentic_system.services.lead_loop import run_lead_loop
from agentic_system.services.preparation import WorkspacePreparation
from agentic_system.services.workday_planner import rank_tasks
from agentic_system.workday_graph import run_workday_graph

# ---------------------------------------------------------------------------
# Backend detection helpers
# ---------------------------------------------------------------------------

def _has_bedrock() -> bool:
    return bool(
        os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")
    )


def _has_groq() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def answer_mode() -> str:
    """Report which inference backend is active."""
    if _has_bedrock():
        return "LangGraph + AWS Bedrock lead/specialists"
    if _has_groq():
        return "LangGraph + Groq lead/specialists"
    return "LangGraph offline planner"


# ---------------------------------------------------------------------------
# Answer backends
# ---------------------------------------------------------------------------

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
    return (
        "I could not find a relevant task or source record. "
        "Try asking about a deadline, Monday, or what to do next."
    )


def _groq_answer(question: str, preparation: WorkspacePreparation) -> str:
    from langchain_groq import ChatGroq

    model = ChatGroq(model=GROQ_MODEL, temperature=0, max_tokens=300)
    return run_lead_loop(question, preparation, model)


def _bedrock_answer(question: str, preparation: WorkspacePreparation) -> str:
    """Use AWS Bedrock Converse API via a LangChain-compatible adapter."""
    from agentic_system.services.bedrock_client import bedrock_client

    LEAD_SYSTEM_PROMPT = (
        "You are the lead student workday agent. Work in a bounded loop. "
        "Use get_workday_tasks for planning questions, get_available_time when "
        "the student mentions a time constraint, and search_workspace for "
        "specific facts. Delegate course questions to ask_course_specialist and "
        "email or meeting questions to ask_inbox_specialist. After observing "
        "tool results, answer concisely with a recommendation and reason. "
        "Never invent tasks, dates, or sources."
    )

    # Build context snapshot from workspace for grounded reasoning.
    tasks = preparation.get_tasks()
    task_lines = "\n".join(
        f"- {t.title} [due: {t.due_label}, ~{t.estimated_minutes}min, priority: {t.priority}]"
        for t in rank_tasks(tasks)[:8]
    )
    records = preparation.store.search(question, limit=5)
    record_lines = "\n".join(
        f"- [{r['source_type']}] {r['title']}: {' '.join(r['content'].split())[:200]}"
        for r in records
    )

    context_prompt = (
        f"Student workspace context:\n"
        f"Tasks (ranked by priority):\n{task_lines or 'No tasks recorded.'}\n\n"
        f"Relevant workspace records:\n{record_lines or 'No records found.'}\n\n"
        f"Student question: {question}"
    )

    return bedrock_client.converse(
        messages=[{"role": "user", "content": context_prompt}],
        system_prompt=LEAD_SYSTEM_PROMPT,
        model_id=AGENT_MODEL_IDS["lead_agent"],
        max_tokens=600,
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer_question(question: str, preparation: WorkspacePreparation) -> str:
    """Return a model-backed grounded response with layered fallback.

    Priority order: AWS Bedrock → Groq → offline planner.
    LangGraph state machine is used when Groq is available.
    Bedrock uses a direct converse call (no LangChain tooling needed).
    """
    # --- Bedrock path (primary for hackathon) ---
    if _has_bedrock():
        try:
            return _bedrock_answer(question, preparation)
        except Exception:
            pass

    # --- Groq + LangGraph path ---
    if _has_groq():
        if os.getenv("USE_LANGGRAPH", "true").lower() in {"1", "true", "yes"}:
            try:
                return str(
                    run_workday_graph(question, preparation, _groq_answer).get("response", "")
                )
            except Exception:
                pass
        try:
            return _groq_answer(question, preparation)
        except Exception:
            pass

    # --- LangGraph offline path ---
    if os.getenv("USE_LANGGRAPH", "true").lower() in {"1", "true", "yes"}:
        try:
            return str(
                run_workday_graph(question, preparation, _offline_answer).get("response", "")
            )
        except Exception:
            pass

    return _offline_answer(question, preparation)
