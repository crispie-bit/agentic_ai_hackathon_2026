"""NTULearn sync — Blackboard REST API primary, Playwright DOM fallback.

Strategy:
1. ``BlackboardRESTClient.sync_courses_and_announcements()`` — cookie-based
   REST calls to the Blackboard Learn v1 JSON API. Writes directly into
   ``WorkspaceStore`` via ``upsert_course / upsert_announcement``.
   Includes Harvey's dict-body fix for announcement body payloads.
2. ``NTULearnCourseSyncService.sync_current_courses()`` — Playwright DOM
   scraper (original logic). Used as fallback when REST returns nothing.

Call ``sync_ntulearn_to_store(store)`` from ``app.py`` instead of instantiating
these classes directly.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from agentic_system.config import NTULEARN_BASE_URL, NTULEARN_SESSION_PATH
from agentic_system.services.course_memory import CourseMemoryStore
from agentic_system.services.ntulearn_browser import NTULearnBrowser

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Blackboard REST API client (ported from harvey-workday-os-v2)
# ---------------------------------------------------------------------------

class BlackboardRESTClient:
    """Cookie-based REST client for the Blackboard Learn v1 JSON API."""

    def __init__(
        self,
        auth_file: Path | None = None,
        store: Any | None = None,  # WorkspaceStore, typed loosely to avoid circular import
    ) -> None:
        self.auth_file = auth_file or NTULEARN_SESSION_PATH
        self.store = store

    # ── auth helpers ──────────────────────────────────────────────────────

    def is_authenticated(self) -> bool:
        if not self.auth_file.exists():
            return False
        try:
            with open(self.auth_file, encoding="utf-8") as f:
                data = json.load(f)
            return len(data.get("cookies", [])) > 0
        except Exception:
            return False

    def _get_cookie_header(self) -> str:
        try:
            if not self.auth_file.exists():
                return ""
            with open(self.auth_file, encoding="utf-8") as f:
                data = json.load(f)
            cookies = data.get("cookies", [])
            return "; ".join(
                f"{c['name']}={c['value']}"
                for c in cookies
                if "ntu" in c.get("domain", "") or "blackboard" in c.get("domain", "")
            )
        except Exception:
            return ""

    def _api_get(self, url: str) -> dict[str, Any] | None:
        cookie_header = self._get_cookie_header()
        if not cookie_header:
            return None
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": _DEFAULT_USER_AGENT,
                    "Cookie": cookie_header,
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            return None
        return None

    # ── sync entry points ─────────────────────────────────────────────────

    def get_user_id(self) -> str | None:
        base = NTULEARN_BASE_URL.rstrip("/")
        data = self._api_get(f"{base}/learn/api/v1/users/me")
        if data:
            return data.get("id")
        return None

    def sync_courses_and_announcements(self) -> dict[str, Any]:
        """Fetch enrolled courses and recent announcements via the REST API.

        Writes results into self.store if provided.
        Includes Harvey's dict-body fix for announcement body payloads.
        """
        if not self.is_authenticated():
            return {"success": False, "error": "Not authenticated — complete NTULearn login first."}

        user_id = self.get_user_id()
        if not user_id:
            return {"success": False, "error": "Session expired or REST API unavailable."}

        base = NTULEARN_BASE_URL.rstrip("/")
        memberships_url = (
            f"{base}/learn/api/v1/users/{user_id}/memberships"
            "?expand=course.effectiveAvailability,course.permissions,courseRole"
            "&includeCount=true&limit=100"
        )
        data = self._api_get(memberships_url)
        if not data or "results" not in data:
            return {"success": False, "error": "Failed to fetch course memberships."}

        courses_saved = 0
        announcements_saved = 0

        for item in data.get("results", []):
            course_obj = item.get("course", {})
            course_id = item.get("courseId") or course_obj.get("id")
            if not course_id:
                continue

            course_code = course_obj.get("courseId", "")
            title = course_obj.get("name") or course_obj.get("title") or course_code
            term = "General"
            m = re.search(r"(\d{2}[sS]\d)", course_code)
            if m:
                term = m.group(1).upper()

            if self.store:
                self.store.upsert_course(course_id, course_code, title, term)
            courses_saved += 1

            # Fetch announcements for this course
            ann_url = f"{base}/learn/api/v1/courses/{course_id}/announcements?limit=5"
            ann_data = self._api_get(ann_url)
            if ann_data and "results" in ann_data:
                for ann in ann_data["results"]:
                    ann_id = ann.get("id", "")
                    ann_title = ann.get("title", "Course Announcement")

                    # Harvey's dict-body fix — Blackboard may return body as dict
                    raw_body = ann.get("body", "")
                    if isinstance(raw_body, dict):
                        raw_body = raw_body.get("raw") or raw_body.get("text") or ""
                    elif not isinstance(raw_body, str):
                        raw_body = str(raw_body or "")
                    ann_body = re.sub(r"<[^>]+>", "", raw_body).strip()
                    posted_at = ann.get("created", "")

                    if self.store:
                        self.store.upsert_announcement(
                            ann_id, course_id, course_code, ann_title, ann_body, posted_at
                        )
                    announcements_saved += 1

        return {
            "success": True,
            "courses_synced": courses_saved,
            "announcements_synced": announcements_saved,
        }

    def fetch_course_contents(self, course_id: str) -> list[dict[str, Any]]:
        """Fetch content items (folders, documents, files) for a course."""
        base = NTULEARN_BASE_URL.rstrip("/")
        url = f"{base}/learn/api/v1/courses/{course_id}/contents?limit=50"
        data = self._api_get(url)
        if not data or "results" not in data:
            return []

        results = []
        for c in data.get("results", []):
            cid = c.get("id")
            title = c.get("title") or "Untitled Material"
            handlers = c.get("contentHandler", {})
            ctype = handlers.get("id", "folder" if c.get("hasChildren") else "document")
            url_link = c.get("links", {}).get("self", "")
            download_url = ""
            if "attachment" in str(handlers).lower() or "file" in str(handlers).lower():
                download_url = (
                    f"{base}/webapps/blackboard/execute/content/file"
                    f"?cmd=view&content_id={cid}&course_id={course_id}"
                )
            results.append(
                {"id": cid, "title": title, "type": ctype, "download_url": download_url}
            )
        return results


# ---------------------------------------------------------------------------
# Original Playwright DOM-scraper (kept as fallback)
# ---------------------------------------------------------------------------

class NTULearnCourseSyncService:
    """Bridge a live NTULearn session to the local course knowledge store.

    Used as fallback when the Blackboard REST API returns no data.
    """

    def __init__(self, db_path: str | None = None, browser_factory: Callable[[], Any] | None = None):
        self.memory = CourseMemoryStore(db_path)
        self.browser_factory = browser_factory or (lambda: NTULearnBrowser())

    def sync_current_courses(self, *, semester: str = "current") -> list[dict[str, Any]]:
        browser = self.browser_factory()
        opened_browser = None
        if isinstance(browser, NTULearnBrowser):
            opened_browser, page = browser.open_authenticated_page(NTULEARN_BASE_URL)
        else:
            page = browser.new_page() if hasattr(browser, "new_page") else browser
            try:
                page.goto(NTULEARN_BASE_URL, wait_until="domcontentloaded", timeout=30_000)
            except TypeError:
                page.goto(NTULEARN_BASE_URL)

        try:
            discovered = self._discover_materials(page)
        finally:
            if opened_browser is not None:
                opened_browser.close()
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

    # ── internal helpers ──────────────────────────────────────────────────

    def _discover_materials(self, page: Any) -> list[dict[str, Any]]:
        links: list[dict[str, Any]] = []
        locator = getattr(page, "locator", None)
        if callable(locator):
            try:
                found = locator("a[href]")
                if hasattr(found, "all"):
                    for elem in found.all():
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
        unique: dict[str, dict[str, Any]] = {}
        for item in links:
            unique[item.get("href", "")] = item
        return list(unique.values())

    def _safe_get_attribute(self, element: Any, name: str) -> str:
        try:
            return element.get_attribute(name) or ""
        except Exception:
            return ""

    def _safe_text_content(self, element: Any) -> str:
        try:
            return element.text_content() or ""
        except Exception:
            return ""

    def _extract_course_code(self, text: str) -> str:
        match = re.search(r"(?i)\b[A-Z]{2,5}\d{4,5}\b", text)
        return match.group(0).upper() if match else "UNKNOWN"

    def _extract_week_number(self, text: str) -> int | None:
        match = re.search(r"(?i)week\s*[-_ ]?(\d+)", text)
        return int(match.group(1)) if match else None

    def _extract_file_type(self, href: str) -> str:
        lower = href.lower()
        for ext in (".pdf", ".ppt", ".pptx", ".doc", ".docx", ".txt", ".md"):
            if ext in lower:
                return ext.lstrip(".")
        return "link"


# ---------------------------------------------------------------------------
# Convenience helper used by app.py
# ---------------------------------------------------------------------------

def sync_ntulearn_to_store(store: Any) -> dict[str, Any]:
    """Run the REST API sync first; fall back to Playwright DOM scraper.

    Returns a result dict with ``success``, ``courses_synced``, and
    ``announcements_synced`` keys.
    """
    rest_client = BlackboardRESTClient(store=store)
    if rest_client.is_authenticated():
        result = rest_client.sync_courses_and_announcements()
        if result.get("success") and (result.get("courses_synced", 0) > 0):
            return result

    # Playwright fallback — still writes to the course_memory SQLite.
    try:
        playwright_service = NTULearnCourseSyncService()
        records = playwright_service.sync_current_courses()
        return {
            "success": bool(records),
            "courses_synced": len(records),
            "announcements_synced": 0,
            "note": "Playwright fallback used.",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


    def sync_current_courses(self, *, semester: str = "current") -> list[dict[str, Any]]:
        browser = self.browser_factory()
        opened_browser = None
        if isinstance(browser, NTULearnBrowser):
            opened_browser, page = browser.open_authenticated_page(NTULEARN_BASE_URL)
        else:
            page = browser.new_page() if hasattr(browser, "new_page") else browser
            try:
                page.goto(NTULEARN_BASE_URL, wait_until="domcontentloaded", timeout=30_000)
            except TypeError:
                page.goto(NTULEARN_BASE_URL)

        try:
            discovered = self._discover_materials(page)
        finally:
            if opened_browser is not None:
                opened_browser.close()
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
