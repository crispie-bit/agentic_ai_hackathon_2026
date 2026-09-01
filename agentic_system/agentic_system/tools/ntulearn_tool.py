"""Utility functions for reading NTULearn page content and extracting update summaries.

This module supports two paths:
- a local snapshot parser for testing and offline use
- a browser-based flow for the first startup login through NTU SSO
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


def login_to_ntulearn(storage_state_path: str | None = None, headless: bool = False):
    """Open a visible browser on first startup so the user can complete NTU SSO login.

    After login, the browser state is stored to disk so future runs can load it and keep the workflow lightweight.
    """
    if sync_playwright is None:
        raise RuntimeError(
            "Playwright is not installed. Install it with: pip install playwright && python -m playwright install chromium"
        )

    target = Path(storage_state_path) if storage_state_path else Path(__file__).resolve().parents[1] / "ntulearn_session.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=target.read_text(encoding="utf-8") if target.exists() else None)
        page = context.new_page()
        page.goto("https://learning.ntu.edu.sg/")
        page.wait_for_load_state("networkidle")
        context.storage_state(path=str(target))
        return browser


class _NTULearnHTMLParser(HTMLParser):
    """Minimal HTML parser that keeps useful text content and ignores boilerplate."""

    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []
        self._skip_tags = {"script", "style", "noscript"}
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        if tag in self._skip_tags:
            self._ignore_depth += 1

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in self._skip_tags and self._ignore_depth > 0:
            self._ignore_depth -= 1

    def handle_data(self, data: str):
        if self._ignore_depth > 0:
            return
        text = " ".join(data.split())
        if text:
            self.chunks.append(text)


def _extract_text_from_html(raw_html: str) -> list[str]:
    parser = _NTULearnHTMLParser()
    parser.feed(raw_html)
    parser.close()
    return parser.chunks


def _extract_updates_from_text(text_chunks: list[str]) -> list[str]:
    relevant: list[str] = []
    current = ""

    for chunk in text_chunks:
        lower = chunk.lower()
        is_relevant = any(
            keyword in lower
            for keyword in ["assignment", "announcement", "deadline", "quiz", "lecture", "due", "course"]
        )

        if not is_relevant:
            if current and len(current.split()) < 25:
                current = f"{current} {chunk}".strip()
            continue

        if not current:
            current = chunk
            continue

        if len(current.split()) < 25:
            current = f"{current} {chunk}".strip()
        else:
            relevant.append(current)
            current = chunk

    if current:
        relevant.append(current)

    if not relevant:
        relevant = text_chunks[:10]

    return relevant


def fetch_ntulearn_updates(source: str | None = None) -> list[str]:
    """Return relevant NTULearn updates from an HTML page or text file.

    Args:
        source: path to a local HTML/text file. If omitted, reads the default project data file if present.
    """
    source_path = Path(source) if source else Path(__file__).resolve().parents[1] / "ntulearn_updates.txt"

    if source_path.exists():
        raw_text = source_path.read_text(encoding="utf-8", errors="ignore")
    else:
        raw_text = ""

    if not raw_text:
        return [
            "No NTULearn updates found. Add a local page snapshot or provide a source file path.",
        ]

    if "<" in raw_text and ">" in raw_text:
        text_chunks = _extract_text_from_html(raw_text)
    else:
        text_chunks = [line.strip() for line in raw_text.splitlines() if line.strip()]

    updates = _extract_updates_from_text(text_chunks)
    return updates[:10]
