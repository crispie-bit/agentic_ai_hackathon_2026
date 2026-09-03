from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agentic_system.services.workspace_store import WorkspaceStore
from agentic_system.services.ntulearn_sync import NTULearnCourseSyncService
from agentic_system.tools.outlook_tool import fetch_outlook_messages


@dataclass
class PreparationState:
    status: str = "not_started"
    steps: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None


class WorkspacePreparation:
    """Coordinates authenticated source sync before enabling assistant queries."""

    def __init__(self, store: WorkspaceStore | None = None):
        self.store = store or WorkspaceStore()

    def run_outlook_sync(self, fetcher: Callable[[], list[dict[str, Any]]] = fetch_outlook_messages) -> int:
        messages = fetcher()
        return self.store.add_outlook_messages(messages)

    def run_ntulearn_sync(self, syncer: NTULearnCourseSyncService | None = None) -> int:
        service = syncer or NTULearnCourseSyncService()
        records = service.sync_current_courses()
        for record in records:
            self.store.add_source(
                source_type="ntulearn",
                title=str(record.get("title") or "Course material"),
                content=str(record.get("extracted_text") or record.get("summary") or ""),
                metadata=str(record.get("source_url") or ""),
            )
        return len(records)

    def answer(self, question: str) -> str:
        matches = self.store.search(question)
        if not matches:
            return "I could not find that in the synced NTULearn or Outlook data. Try a course code, document title, sender, or deadline keyword."
        lines = []
        for item in matches[:5]:
            excerpt = " ".join(item["content"].split())[:280]
            lines.append(f"**{item['title']}** ({item['source_type']}): {excerpt}")
        return "Here is what I found in your synced workspace:\n\n" + "\n\n".join(lines)
