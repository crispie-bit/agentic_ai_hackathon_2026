from agentic_system.services.workday_planner import WorkdayTask, rank_tasks
from agentic_system.services.preparation import WorkspacePreparation
from agentic_system.services.workspace_store import WorkspaceStore
from agentic_system.services.answering import answer_mode, answer_question


def test_rank_tasks_prioritises_urgent_work_and_shorter_ties():
    tasks = [
        WorkdayTask("long urgent", "today", 90, 5, "demo", "do it"),
        WorkdayTask("short normal", "tomorrow", 20, 3, "demo", "do it"),
        WorkdayTask("short urgent", "today", 30, 5, "demo", "do it"),
    ]

    ranked = rank_tasks(tasks)

    assert [task.title for task in ranked] == [
        "short urgent",
        "long urgent",
        "short normal",
    ]


def test_demo_answers_natural_language_questions(tmp_path):
    preparation = WorkspacePreparation(WorkspaceStore(str(tmp_path / "demo.sqlite")))
    preparation.load_demo_data()

    answer = preparation.answer("What is happening on Monday?")

    assert "Monday" in answer


def test_offline_answer_recommends_highest_priority_task(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    preparation = WorkspacePreparation(WorkspaceStore(str(tmp_path / "demo.sqlite")))
    preparation.load_demo_data()

    answer = answer_question("What should I do next?", preparation)

    assert answer_mode() == "Offline planner"
    assert "Recommendation" in answer
    assert "DEMO-COURSE-101" in answer


def test_marking_task_complete_updates_available_tasks(tmp_path):
    preparation = WorkspacePreparation(WorkspaceStore(str(tmp_path / "demo.sqlite")))
    preparation.load_demo_data()

    title = preparation.get_tasks()[0].title
    assert preparation.mark_task_complete(title) is True
    assert title not in {task.title for task in preparation.get_tasks()}


def test_rescheduling_task_updates_planner_deadline(tmp_path):
    preparation = WorkspacePreparation(WorkspaceStore(str(tmp_path / "demo.sqlite")))
    preparation.load_demo_data()
    title = preparation.get_tasks()[0].title

    assert preparation.reschedule_task(title, "Tuesday 17:00") is True
    updated = {task.title: task for task in preparation.get_tasks()}

    assert updated[title].due_label == "Tuesday 17:00"
