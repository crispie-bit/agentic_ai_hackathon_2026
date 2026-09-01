from __future__ import annotations

from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


class NTULearnBrowser:
    """Thin Playwright wrapper for authenticated NTULearn access and page traversal."""

    def __init__(self, storage_state_path: str | None = None):
        self.storage_state_path = Path(storage_state_path) if storage_state_path else Path(__file__).resolve().parents[1] / "ntulearn_session.json"
        self.storage_state_path.parent.mkdir(parents=True, exist_ok=True)

    def open_headed_session(self, headless: bool = False):
        if sync_playwright is None:
            raise RuntimeError(
                "Playwright is not installed. Install it with: pip install playwright && python -m playwright install chromium"
            )

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=headless)
            context = browser.new_context(storage_state=self.storage_state_path.read_text(encoding="utf-8") if self.storage_state_path.exists() else None)
            page = context.new_page()
            page.goto("https://learning.ntu.edu.sg/")
            page.wait_for_load_state("networkidle")
            context.storage_state(path=str(self.storage_state_path))
            return browser

    def open_authenticated_page(self, url: str, headless: bool = False):
        browser = self.open_headed_session(headless=headless)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state("networkidle")
        return browser, page
