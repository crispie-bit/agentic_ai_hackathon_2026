from __future__ import annotations

from agentic_system.services.ntulearn_sync import NTULearnCourseSyncService


def test_ntulearn_sync_ingests_course_materials(tmp_path):
    class FakeDownload:
        def __init__(self, name: str):
            self.suggested_filename = name

    class FakeDownloadable:
        def __init__(self, *, href: str, label: str):
            self.href = href
            self.label = label

    class FakePage:
        def __init__(self):
            self.visited: list[str] = []
            self.downloads: list[FakeDownloadable] = [
                FakeDownloadable(
                    href="https://learning.ntu.edu.sg/courses/CZ3005/files/week5_tutorial.pdf",
                    label="Week 5 Tutorial Slides",
                ),
                FakeDownloadable(
                    href="https://learning.ntu.edu.sg/courses/CZ3005/files/week5_notes.pdf",
                    label="Week 5 Notes",
                ),
            ]

        def goto(self, url: str):
            self.visited.append(url)
            return None

        def locator(self, selector: str):
            if selector.endswith("a[href]"):
                return FakeLocator(self.downloads)
            return FakeLocator([])

    class FakeLocator:
        def __init__(self, entries):
            self.entries = entries

        def all(self):
            return [
                type("Entry", (), {"get_attribute": lambda self, attr: self._values[attr], "text_content": lambda self: self._values["text"]})()
                for self in []
            ]

    class FakeBrowser:
        def __init__(self):
            self.page = FakePage()

        def new_page(self):
            return self.page

        def close(self):
            return None

    class FakePlaywright:
        def __init__(self):
            self.chromium = type("Chromium", (), {"launch": lambda self, **kwargs: FakeBrowser()})()

    service = NTULearnCourseSyncService(
        db_path=str(tmp_path / "course_memory.sqlite"),
        browser_factory=lambda: FakeBrowser(),
    )

    rows = service.sync_current_courses()

    assert len(rows) >= 2
    assert any(row["course_code"] == "CZ3005" for row in rows)
    assert any("week5" in (row["title"] or "").lower() for row in rows)
