from agentic_system.services.workday_planner import WorkdayTask, rank_tasks


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
