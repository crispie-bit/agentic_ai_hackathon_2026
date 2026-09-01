from __future__ import annotations

import re
from typing import Any, Callable

from agentic_system.services.course_memory import CourseMemoryStore
from agentic_system.services.ntulearn_browser import NTULearnBrowser


class NTULearnCourseSyncService:
    """Bridge a live NTULearn session to the local course knowledge store."""

    def __init__(self, db_path: str | None = None, browser_factory: Callable[[], Any] | None = None):
        self.memory = CourseMemoryStore(db_path)
        self.browser_factory = browser_factory or (lambda: NTULearnBrowser())

    def sync_current_courses(self, *, semester: str = "current") -> list[dict[str, Any]]:
        browser = self.browser_factory()
        page = browser.new_page() if hasattr(browser, "new_page") else browser

        page.goto("https://learning.ntu.edu.sg/")
        if hasattr(page, "wait_for_load_state"):
            page.wait_for_load_state("networkidle")

        discovered = self._discover_materials(page)
        if not discovered:
            return []

        records: list[dict[str, Any]] = []
        for item in discovered:
            item_title = item.get("title") or item.get("label") or item.get("name") or ""
            course_code = self._extract_course_code(item.get("href") or item_title or "")
            if not course_code:
                continue

            week_number = self._extract_week_number(item_title or item.get("href") or "")
            title = item_title or item.get("name") or item.get("href", "").split("/")[-1]
            if week_number is not None:
                week_tag = f"week{week_number}"
                if week_tag not in title.lower():
                    title = f"{title} ({week_tag})"
            summary = item.get("summary") or f"Course material for {course_code}."
            record = {
                "course_code": course_code,
                "semester": semester,
                "title": title,
                "file_type": self._extract_file_type(item.get("href") or ""),
                "week_number": week_number,
                "due_date": None,
                "source_url": item.get("href") or "",
                "extracted_text": item.get("text") or summary,
                "summary": summary,
            }
            row_id = self.memory.add_document(**record)
            record["id"] = row_id
            records.append(record)

        return records

    def _discover_materials(self, page: Any) -> list[dict[str, Any]]:
        links: list[dict[str, Any]] = []

        locator = getattr(page, "locator", None)
        if callable(locator):
            try:
                found = locator("a[href]")
                if hasattr(found, "all"):
                    elements = found.all()
                    for elem in elements:
                        href = self._safe_get_attribute(elem, "href") or ""
                        text = self._safe_text_content(elem)
                        if href:
                            links.append({"href": href, "title": text or href})
            except Exception:
                pass

        downloads = getattr(page, "downloads", None)
        if downloads:
            for item in downloads:
                href = getattr(item, "href", None) or getattr(item, "url", None) or ""
                label = getattr(item, "label", None) or getattr(item, "title", None) or href
                if href:
                    links.append({"href": href, "title": label, "label": label, "text": label})

        # Remove duplicates by URL
        unique: dict[str, dict[str, Any]] = {}
        for item in links:
            href = item.get("href") or ""
            unique[href] = item
        return list(unique.values())

    def _safe_get_attribute(self, element: Any, name: str) -> str:
        try:
            value = element.get_attribute(name)
        except Exception:
            value = None
        return value or ""

    def _safe_text_content(self, element: Any) -> str:
        try:
            return element.text_content() or ""
        except Exception:
            return ""

    def _extract_course_code(self, text: str) -> str:
        match = re.search(r"(?i)\b[A-Z]{2,5}\d{4,5}\b", text)
        if match:
            return match.group(0).upper()
        return "UNKNOWN"

    def _extract_week_number(self, text: str) -> int | None:
        match = re.search(r"(?i)week\s*[-_ ]?(\d+)", text)
        if match:
            return int(match.group(1))
        return None

    def _extract_file_type(self, href: str) -> str:
        lower = href.lower()
        for ext in (".pdf", ".ppt", ".pptx", ".doc", ".docx", ".txt", ".md"):
            if ext in lower:
                return ext.lstrip(".")
        return "link"
