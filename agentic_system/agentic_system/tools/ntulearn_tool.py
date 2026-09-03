"""NTULearn browser automation via Playwright.

This module intentionally avoids a separate HTML scraper. NTULearn login and page inspection are handled in-browser so the
user can complete SSO and the app can read the live page content after authentication.
"""

from __future__ import annotations

from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None


def login_to_ntulearn(storage_state_path: str | None = None, headless: bool = False):
    """Open a visible browser on first startup so the user can complete NTU SSO login.

    The browser session is stored to disk for later use so we can reopen the authenticated session without re-logging in.
    """
    if sync_playwright is None:
        raise RuntimeError(
            "Playwright is not installed. Install it with: pip install playwright && python -m playwright install chromium"
        )

    target = Path(storage_state_path) if storage_state_path else Path(__file__).resolve().parents[1] / "ntulearn_session.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    manager = sync_playwright()
    if hasattr(manager, "start"):
        playwright = manager.start()
        browser = playwright.chromium.launch(headless=headless)
        browser._agentic_playwright = playwright
        context = browser.new_context(storage_state=target.read_text(encoding="utf-8") if target.exists() else None)
        page = context.new_page()
        page.goto("https://learning.ntu.edu.sg/")
        page.wait_for_load_state("networkidle")
        context.storage_state(path=str(target))
        return browser

    # Keeps the lightweight fake used by tests compatible with the real flow.
    with manager as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=target.read_text(encoding="utf-8") if target.exists() else None)
        page = context.new_page()
        page.goto("https://learning.ntu.edu.sg/")
        page.wait_for_load_state("networkidle")
        context.storage_state(path=str(target))
        return browser


def save_ntulearn_session(browser, storage_state_path: str | None = None) -> None:
    """Persist cookies after the user completes the visible NTULearn SSO flow."""
    target = Path(storage_state_path) if storage_state_path else Path(__file__).resolve().parents[1] / "ntulearn_session.json"
    contexts = getattr(browser, "contexts", [])
    if not contexts:
        raise RuntimeError("The NTULearn browser session is no longer available.")
    contexts[0].storage_state(path=str(target))


def fetch_ntulearn_updates(*args, **kwargs):
    """Playwright-only NTULearn update retrieval.

    The live site is read through an authenticated browser session rather than a static HTML parser.
    """
    raise NotImplementedError(
        "NTULearn updates must be fetched through Playwright in a live browser session after SSO login."
    )
