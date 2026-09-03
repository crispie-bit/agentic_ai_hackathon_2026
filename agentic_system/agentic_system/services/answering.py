from __future__ import annotations

from agentic_system.services.preparation import WorkspacePreparation


def answer_question(question: str, preparation: WorkspacePreparation) -> str:
    """Return a grounded response from the local synced workspace."""
    return preparation.answer(question)
