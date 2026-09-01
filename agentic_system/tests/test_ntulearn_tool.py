from pathlib import Path

from agentic_system.tools.ntulearn_tool import login_to_ntulearn


def test_login_to_ntulearn_uses_headed_browser(monkeypatch, tmp_path):
    class FakePage:
        def __init__(self):
            self.goto_calls = []
            self.load_state_calls = []

        def goto(self, url):
            self.goto_calls.append(url)

        def wait_for_load_state(self, state):
            self.load_state_calls.append(state)

    class FakeContext:
        def __init__(self, page):
            self.page = page
            self.cookies = [{"name": "session", "value": "abc"}]

        def new_page(self):
            return self.page

        def storage_state(self, path=None):
            if path is not None:
                Path(path).write_text('{"cookies": [{"name": "session", "value": "abc"}]}', encoding="utf-8")
            return {"cookies": self.cookies}

    class FakeBrowser:
        def __init__(self):
            self.page = FakePage()
            self.context = FakeContext(self.page)

        def new_context(self, storage_state=None):
            return FakeContext(self.page)

        def close(self):
            return None

    class FakeChromium:
        def launch(self, headless=False):
            assert headless is False
            return FakeBrowser()

    class FakePlaywright:
        def __init__(self):
            self.chromium = FakeChromium()

    class FakeSyncPlaywright:
        def __enter__(self):
            return FakePlaywright()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "agentic_system.tools.ntulearn_tool.sync_playwright",
        lambda: FakeSyncPlaywright(),
    )

    storage_path = tmp_path / "ntulearn_session.json"
    browser = login_to_ntulearn(storage_state_path=str(storage_path), headless=False)

    assert browser is not None
    assert storage_path.exists()
