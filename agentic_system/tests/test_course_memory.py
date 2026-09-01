from agentic_system.services.course_memory import CourseMemoryStore


def test_course_memory_tracks_latest_course_materials(tmp_path):
    db_path = tmp_path / "course_memory.sqlite"
    store = CourseMemoryStore(str(db_path))

    store.add_document(
        course_code="CZ1003",
        semester="2026S1",
        title="Week 5 Tutorial Slides",
        file_type="pdf",
        week_number=5,
        due_date="2026-02-10",
        source_url="https://example.com/cz1003/week5.pdf",
        extracted_text="This tutorial covers loops and functions. Assignment 2 is due on 2026-02-10.",
        summary="Loops and functions tutorial.",
    )
    store.add_document(
        course_code="CZ1003",
        semester="2026S1",
        title="Week 3 Reading Notes",
        file_type="pdf",
        week_number=3,
        due_date="2026-01-27",
        source_url="https://example.com/cz1003/week3.pdf",
        extracted_text="Reading notes on data structures and arrays.",
        summary="Reading notes.",
    )

    latest = store.search_latest(course_code="CZ1003", limit=2)

    assert latest[0]["title"] == "Week 5 Tutorial Slides"
    assert latest[0]["week_number"] == 5
    assert latest[1]["title"] == "Week 3 Reading Notes"
