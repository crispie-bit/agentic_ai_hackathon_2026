from __future__ import annotations

from typing import Any

from agentic_system.services.course_memory import CourseMemoryStore


class CourseRetriever:
    """Ranks the most relevant course materials for a user asking about the next task or current content."""

    def __init__(self, db_path: str | None = None):
        self.store = CourseMemoryStore(db_path)

    def get_next_material(self, *, course_code: str, semester: str | None = None, limit: int = 3) -> list[dict[str, Any]]:
        return self.store.search_latest(course_code=course_code, semester=semester, limit=limit)

    def find_latest_for_user(self, *, course_code: str | None = None, query: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        if course_code:
            return self.store.search_latest(course_code=course_code, limit=limit)
        if query:
            return self.store.search_by_keyword(query=query, limit=limit)
        return []
