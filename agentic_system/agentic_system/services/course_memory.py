from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class CourseMemoryStore:
    """Lightweight local memory store for course documents and extracted content."""

    def __init__(self, db_path: str | None = None):
        resolved = Path(db_path) if db_path else Path(__file__).resolve().parents[1] / "course_memory.sqlite"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(resolved)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS course_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    course_code TEXT NOT NULL,
                    semester TEXT NOT NULL,
                    title TEXT NOT NULL,
                    file_type TEXT,
                    week_number INTEGER,
                    due_date TEXT,
                    source_url TEXT,
                    extracted_text TEXT,
                    summary TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_course_documents_latest ON course_documents(course_code, semester, week_number DESC, due_date DESC)"
            )

    def add_document(
        self,
        *,
        course_code: str,
        semester: str,
        title: str,
        file_type: str,
        week_number: int | None,
        due_date: str | None,
        source_url: str,
        extracted_text: str,
        summary: str,
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO course_documents (
                    course_code, semester, title, file_type, week_number, due_date, source_url, extracted_text, summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    course_code,
                    semester,
                    title,
                    file_type,
                    week_number,
                    due_date,
                    source_url,
                    extracted_text,
                    summary,
                ),
            )
            return int(cursor.lastrowid)

    def search_latest(self, *, course_code: str, semester: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        query = """
            SELECT course_code, semester, title, file_type, week_number, due_date, source_url, extracted_text, summary
            FROM course_documents
            WHERE course_code = ?
        """
        params: list[Any] = [course_code]
        if semester is not None:
            query += " AND semester = ?"
            params.append(semester)
        query += " ORDER BY week_number DESC, due_date DESC, created_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def search_by_keyword(self, *, query: str, limit: int = 5) -> list[dict[str, Any]]:
        like_query = f"%{query.lower()}%"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT course_code, semester, title, file_type, week_number, due_date, source_url, extracted_text, summary
                FROM course_documents
                WHERE LOWER(title) LIKE ? OR LOWER(extracted_text) LIKE ? OR LOWER(summary) LIKE ?
                ORDER BY week_number DESC, due_date DESC, created_at DESC
                LIMIT ?
                """,
                (like_query, like_query, like_query, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_courses(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT DISTINCT course_code FROM course_documents ORDER BY course_code").fetchall()
        return [row[0] for row in rows]
