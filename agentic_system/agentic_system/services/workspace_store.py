from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class WorkspaceStore:
    """Local searchable store for synced course material and Outlook messages."""

    def __init__(self, db_path: str | None = None):
        path = Path(db_path) if db_path else Path(__file__).resolve().parents[1] / "workspace.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")

            # ── existing tables (unchanged) ─────────────────────────────────
            connection.execute(
                """CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type)")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS task_actions (
                    title TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )"""
            )

            # ── new tables (ported from harvey-workday-os-v2/core/database.py) ──

            connection.execute(
                """CREATE TABLE IF NOT EXISTS courses (
                    id TEXT PRIMARY KEY,
                    course_code TEXT,
                    title TEXT,
                    term TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            )

            connection.execute(
                """CREATE TABLE IF NOT EXISTS announcements (
                    id TEXT PRIMARY KEY,
                    course_id TEXT,
                    course_code TEXT,
                    title TEXT,
                    body TEXT,
                    posted_at TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_ann_course ON announcements(course_code)"
            )

            connection.execute(
                """CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT DEFAULT 'manual',
                    course_code TEXT,
                    due_date TEXT,
                    estimated_minutes INTEGER DEFAULT 60,
                    priority_score REAL DEFAULT 5.0,
                    status TEXT DEFAULT 'pending',
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, priority_score DESC)"
            )

            connection.execute(
                """CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT DEFAULT 'default',
                    role TEXT NOT NULL,
                    agent_name TEXT DEFAULT 'Lead Orchestrator',
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            )


    def add_source(self, *, source_type: str, title: str, content: str, metadata: str = "") -> int:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "INSERT INTO sources (source_type, title, content, metadata) VALUES (?, ?, ?, ?)",
                (source_type, title, content, metadata),
            )
            return int(cursor.lastrowid)

    def add_outlook_messages(self, messages: list[dict[str, Any]]) -> int:
        added = 0
        for message in messages:
            title = str(message.get("subject") or "No subject")
            content = " ".join(
                part for part in [
                    f"From: {message.get('sender', 'Unknown sender')}",
                    f"Received: {message.get('received_at', '')}",
                    str(message.get("body", "")),
                ] if part
            )
            self.add_source(source_type="outlook", title=title, content=content)
            added += 1
        return added

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        terms = [term.strip().lower() for term in query.split() if term.strip()]
        if not terms:
            return []
        clauses = " OR ".join("(LOWER(title) LIKE ? OR LOWER(content) LIKE ?)" for _ in terms)
        params: list[Any] = []
        for term in terms:
            value = f"%{term}%"
            params.extend([value, value])
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"SELECT source_type, title, content, metadata, created_at FROM sources WHERE {clauses} ORDER BY created_at DESC",
                params,
            ).fetchall()
        ranked = sorted(
            rows,
            key=lambda row: sum(
                term in f"{row['title']} {row['content']}".lower() for term in terms
            ),
            reverse=True,
        )
        return [dict(row) for row in ranked[:limit]]

    def counts(self) -> dict[str, int]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute("SELECT source_type, COUNT(*) FROM sources GROUP BY source_type").fetchall()
        return {str(source_type): int(count) for source_type, count in rows}

    def mark_task_complete(self, title: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO task_actions (title, status) VALUES (?, 'completed')",
                (title,),
            )

    def completed_tasks(self) -> set[str]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT title FROM task_actions WHERE status = 'completed'"
            ).fetchall()
        return {str(row[0]) for row in rows}

    def reschedule_task(self, title: str, new_due_label: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO task_actions (title, status) VALUES (?, ?)",
                (title, f"rescheduled:{new_due_label}"),
            )

    def rescheduled_tasks(self) -> dict[str, str]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT title, status FROM task_actions WHERE status LIKE 'rescheduled:%'"
            ).fetchall()
        return {str(title): str(status).removeprefix("rescheduled:") for title, status in rows}

    # ------------------------------------------------------------------
    # NTULearn / course data (ported from harvey-workday-os-v2)
    # ------------------------------------------------------------------

    def upsert_course(self, course_id: str, course_code: str, title: str, term: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """INSERT INTO courses (id, course_code, title, term)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    course_code=excluded.course_code,
                    title=excluded.title,
                    term=excluded.term,
                    updated_at=CURRENT_TIMESTAMP""",
                (course_id, course_code, title, term),
            )

    def upsert_announcement(
        self,
        ann_id: str,
        course_id: str,
        course_code: str,
        title: str,
        body: str,
        posted_at: str,
    ) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """INSERT INTO announcements (id, course_id, course_code, title, body, posted_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    body=excluded.body,
                    posted_at=excluded.posted_at""",
                (ann_id, course_id, course_code, title, body, posted_at),
            )

    def upsert_task(
        self,
        *,
        task_id: str,
        title: str,
        source: str = "ntulearn",
        course_code: str = "",
        due_date: str = "",
        estimated_minutes: int = 60,
        priority_score: float = 5.0,
        notes: str = "",
    ) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """INSERT INTO tasks
                    (id, title, source, course_code, due_date, estimated_minutes, priority_score, notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    due_date=excluded.due_date,
                    estimated_minutes=excluded.estimated_minutes,
                    priority_score=excluded.priority_score,
                    notes=excluded.notes,
                    updated_at=CURRENT_TIMESTAMP""",
                (task_id, title, source, course_code, due_date, estimated_minutes, priority_score, notes),
            )

    def get_tasks_from_db(self, status: str | None = None) -> list[dict[str, Any]]:
        """Return tasks ordered by priority (descending) then due_date (ascending)."""
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            if status:
                rows = connection.execute(
                    "SELECT * FROM tasks WHERE status = ? ORDER BY priority_score DESC, due_date ASC",
                    (status,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM tasks ORDER BY status ASC, priority_score DESC, due_date ASC"
                ).fetchall()
        return [dict(row) for row in rows]

    def set_task_status(self, task_id: str, new_status: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (new_status, task_id),
            )

    def get_all_courses(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM courses ORDER BY term DESC, course_code ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_announcements(self, limit: int = 20) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM announcements ORDER BY posted_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_chat_message(
        self,
        role: str,
        content: str,
        agent_name: str = "Lead Orchestrator",
        session_id: str = "default",
    ) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO chat_messages (session_id, role, agent_name, content) VALUES (?, ?, ?, ?)",
                (session_id, role, agent_name, content),
            )

    def get_chat_history(self, session_id: str = "default", limit: int = 50) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]
