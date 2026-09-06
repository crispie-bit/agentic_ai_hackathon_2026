from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkdayTask:
    title: str
    due_label: str
    estimated_minutes: int
    priority: int
    source: str
    action: str


def demo_tasks(completed_titles: set[str] | None = None) -> list[WorkdayTask]:
    """Return fictional tasks for the submission-safe demo flow."""
    tasks = [
        WorkdayTask(
            title="DEMO-COURSE-101 project checkpoint",
            due_label="Monday 17:00",
            estimated_minutes=90,
            priority=5,
            source="Course data",
            action="Submit the design outline and test notes.",
        ),
        WorkdayTask(
            title="Prepare group evaluation plan",
            due_label="Monday 19:00",
            estimated_minutes=30,
            priority=4,
            source="Calendar data",
            action="Bring the draft evaluation plan to the group meeting.",
        ),
        WorkdayTask(
            title="Review database lecture notes",
            due_label="Monday 12:00",
            estimated_minutes=45,
            priority=3,
            source="Calendar data",
            action="Attend the database lecture in LT-2 and capture follow-up work.",
        ),
    ]
    completed = completed_titles or set()
    return [task for task in tasks if task.title not in completed]


def rank_tasks(tasks: list[WorkdayTask]) -> list[WorkdayTask]:
    """Rank by urgency first, then prefer shorter tasks when urgency ties."""
    return sorted(tasks, key=lambda task: (-task.priority, task.estimated_minutes, task.title))
