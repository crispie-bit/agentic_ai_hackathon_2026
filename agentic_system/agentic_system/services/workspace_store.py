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
        clauses = " AND ".join("(LOWER(title) LIKE ? OR LOWER(content) LIKE ?)" for _ in terms)
        params: list[Any] = []
        for term in terms:
            value = f"%{term}%"
            params.extend([value, value])
        params.append(limit)
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"SELECT source_type, title, content, metadata, created_at FROM sources WHERE {clauses} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def counts(self) -> dict[str, int]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute("SELECT source_type, COUNT(*) FROM sources GROUP BY source_type").fetchall()
        return {str(source_type): int(count) for source_type, count in rows}
