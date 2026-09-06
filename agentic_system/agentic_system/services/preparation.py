from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agentic_system.services.workspace_store import WorkspaceStore
from agentic_system.services.ntulearn_sync import NTULearnCourseSyncService
from agentic_system.services.workday_planner import (
    WorkdayTask,
    db_tasks_as_workday_tasks,
    demo_tasks,
    extract_tasks_from_announcements,
)
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
        # Auto-mine tasks from any newly synced announcements.
        extract_tasks_from_announcements(self.store)
        return len(records)

    def load_demo_data(self) -> int:
        """Load fictional records so the product can be demonstrated offline."""
        if self.store.search("DEMO-COURSE-101"):
            return 0

        records = [
            {
                "source_type": "demo_course",
                "title": "DEMO-COURSE-101 project checkpoint",
                "content": "Project checkpoint due Monday at 17:00. Estimated effort: 90 minutes. Submit the design outline and test notes.",
                "metadata": "synthetic",
            },
            {
                "source_type": "demo_calendar",
                "title": "DEMO timetable: database lecture",
                "content": "Database lecture on Monday from 10:00 to 12:00 in LT-2.",
                "metadata": "synthetic",
            },
            {
                "source_type": "demo_inbox",
                "title": "DEMO group meeting reminder",
                "content": "Group meeting Monday at 19:00. Bring the draft evaluation plan.",
                "metadata": "synthetic",
            },
        ]
        for record in records:
            self.store.add_source(**record)
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

    def follow_up_prompts(self) -> list[str]:
        """Return safe example questions for the currently indexed workspace."""
        return [
            "What is due on Monday?",
            "What should I prepare for the group meeting?",
            "How much time should I reserve for the project checkpoint?",
        ]

    def get_tasks(self) -> list[WorkdayTask]:
        """Return structured tasks for the workday dashboard.

        If real NTULearn courses have been synced, return DB-driven tasks scored
        by ``calculate_priority()``. Otherwise fall back to 3 fictional demo tasks.
        """
        # Real data path: at least one course has been synced.
        if self.store.get_all_courses():
            tasks = db_tasks_as_workday_tasks(self.store)
            if tasks:
                return tasks
            # Courses present but no tasks yet — mine from announcements.
            extract_tasks_from_announcements(self.store)
            return db_tasks_as_workday_tasks(self.store)

        # Demo / offline path.
        if self.store.search("DEMO-COURSE-101"):
            return demo_tasks(self.store.completed_tasks(), self.store.rescheduled_tasks())
        return []

    def mark_task_complete(self, title: str) -> bool:
        """Persist completion only for a known current task."""
        if title not in {task.title for task in self.get_tasks()}:
            return False
        self.store.mark_task_complete(title)
        return True

    def reschedule_task(self, title: str, new_due_label: str) -> bool:
        if title not in {task.title for task in self.get_tasks()}:
            return False
        self.store.reschedule_task(title, new_due_label)
        return True
