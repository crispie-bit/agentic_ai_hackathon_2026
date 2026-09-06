"""Workday planner — task prioritisation and schedule generation.

Includes:
  - ``WorkdayTask``        — immutable task dataclass used by the lead loop
  - ``demo_tasks()``       — 3 fictional tasks for the offline demo flow
  - ``rank_tasks()``       — sort by urgency then estimated effort
  - ``calculate_priority()``              — ISO-timestamp urgency scorer (Harvey)
  - ``extract_tasks_from_announcements()``— regex miner for NTULearn announcements (Harvey)
  - ``generate_day_schedule()``           — time-blocked schedule builder (Harvey)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentic_system.services.workspace_store import WorkspaceStore


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkdayTask:
    title: str
    due_label: str
    estimated_minutes: int
    priority: int
    source: str
    action: str


# ---------------------------------------------------------------------------
# Demo data (unchanged — used when no live NTULearn data is present)
# ---------------------------------------------------------------------------

def demo_tasks(
    completed_titles: set[str] | None = None,
    rescheduled: dict[str, str] | None = None,
) -> list[WorkdayTask]:
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
    updated_due = rescheduled or {}
    return [
        WorkdayTask(
            task.title,
            updated_due.get(task.title, task.due_label),
            task.estimated_minutes,
            task.priority,
            task.source,
            task.action,
        )
        for task in tasks
        if task.title not in completed
    ]


def rank_tasks(tasks: list[WorkdayTask]) -> list[WorkdayTask]:
    """Rank by urgency first, then prefer shorter tasks when urgency ties."""
    return sorted(tasks, key=lambda task: (-task.priority, task.estimated_minutes, task.title))


# ---------------------------------------------------------------------------
# Real-data prioritisation (ported from harvey-workday-os-v2/services/planner.py)
# ---------------------------------------------------------------------------

_DEADLINE_PATTERNS: list[tuple[str, float, int]] = [
    (r"\b(?:homework|hw)\b", 8.5, 60),
    (r"\b(?:ca1|ca2|continuous assessment)\b", 9.0, 90),
    (r"\b(?:quiz|test|exam)\b", 9.2, 90),
    (r"\b(?:lab|laboratory|grouping)\b", 7.5, 60),
    (r"\b(?:tutorial|sheet)\b", 7.0, 45),
    (r"\b(?:due|submit|deadline|submission)\b", 8.8, 60),
    (r"\b(?:project|milestone|presentation)\b", 8.0, 120),
]


def calculate_priority(due_date_str: str, estimated_mins: int = 60, is_graded: bool = True) -> float:
    """Compute urgency score from an ISO 8601 due-date string.

    Returns a float in [0, 10] where 10 = overdue.
    Ported from harvey-workday-os-v2/services/planner.py.
    """
    base_score = 5.0
    if not due_date_str:
        return base_score
    try:
        due = datetime.fromisoformat(due_date_str.replace("Z", "+00:00"))
        now = datetime.now(due.tzinfo)
        diff_hours = (due - now).total_seconds() / 3600.0

        if diff_hours < 0:
            urgency = 10.0
        elif diff_hours < 24:
            urgency = 9.5
        elif diff_hours < 48:
            urgency = 8.5
        elif diff_hours < 120:
            urgency = 7.0
        else:
            urgency = 5.0

        if is_graded:
            urgency += 0.5
        return min(10.0, round(urgency, 1))
    except Exception:
        return base_score


def extract_tasks_from_announcements(store: "WorkspaceStore") -> None:
    """Scan NTULearn announcements in the DB and upsert detected action tasks.

    Ported from harvey-workday-os-v2/services/planner.py.
    """
    announcements = store.get_announcements(limit=50)
    for ann in announcements:
        title = str(ann.get("title", ""))
        body = str(ann.get("body", ""))
        text = f"{title} {body}".lower()

        matched = False
        priority = 6.0
        est_mins = 60

        for pattern, p_score, mins in _DEADLINE_PATTERNS:
            if re.search(pattern, text):
                matched = True
                priority = max(priority, p_score)
                est_mins = mins

        if not matched:
            continue

        task_id = f"ann_{ann.get('id', '')}"
        clean_title = title.replace("IMP/", "").replace("IMP:", "").strip()
        course_code = ann.get("course_code", "")

        # Try to extract a date from the announcement text
        date_match = re.search(r"([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)", text)
        due_date = (
            date_match.group(1)
            if date_match
            else (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        )

        store.upsert_task(
            task_id=task_id,
            title=f"Action: {clean_title[:80]}",
            source="ntulearn",
            course_code=course_code,
            due_date=due_date,
            estimated_minutes=est_mins,
            priority_score=priority,
            notes=body[:200],
        )


def db_tasks_as_workday_tasks(store: "WorkspaceStore") -> list[WorkdayTask]:
    """Convert DB task rows into ``WorkdayTask`` objects for the lead loop.

    Uses ``calculate_priority()`` to refresh the urgency score at query time
    so task ordering stays accurate throughout the day.
    """
    rows = store.get_tasks_from_db(status="pending")
    result: list[WorkdayTask] = []
    for row in rows:
        refreshed_priority = calculate_priority(
            row.get("due_date", ""),
            estimated_mins=row.get("estimated_minutes", 60),
        )
        result.append(
            WorkdayTask(
                title=row["title"],
                due_label=row.get("due_date", "Unknown"),
                estimated_minutes=row.get("estimated_minutes", 60),
                priority=int(round(refreshed_priority)),
                source=row.get("source", "ntulearn"),
                action=row.get("notes", "Check NTULearn for details.")[:200],
            )
        )
    return result


def generate_day_schedule(store: "WorkspaceStore", available_hours: float = 6.0) -> list[dict[str, Any]]:
    """Build a time-blocked schedule from pending DB tasks.

    Ported from harvey-workday-os-v2/services/planner.py.
    """
    # Auto-mine tasks from announcements if nothing is pending yet.
    pending = store.get_tasks_from_db(status="pending")
    if not pending:
        extract_tasks_from_announcements(store)
        pending = store.get_tasks_from_db(status="pending")

    schedule: list[dict[str, Any]] = []
    current_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    total_minutes_left = available_hours * 60

    for task in pending:
        duration = min(task.get("estimated_minutes", 60), 120)
        if total_minutes_left < 30:
            break

        start_str = current_time.strftime("%I:%M %p")
        end_time = current_time + timedelta(minutes=duration)
        end_str = end_time.strftime("%I:%M %p")

        schedule.append(
            {
                "task_id": task["id"],
                "title": task["title"],
                "course_code": task.get("course_code", ""),
                "start": start_str,
                "end": end_str,
                "duration_mins": duration,
                "priority": task.get("priority_score", 5.0),
            }
        )

        current_time = end_time + timedelta(minutes=15)
        total_minutes_left -= duration + 15

    return schedule

