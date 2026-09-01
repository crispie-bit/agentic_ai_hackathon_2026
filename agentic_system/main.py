from agentic_system.orchestrator import AgentOrchestrator


def main() -> None:
    orchestrator = AgentOrchestrator()
    orchestrator.register_default_agents()

    print("Agentic Workday OS ready.")
    print("Lead agent is online and waiting for delegated tasks.")

    for request in [
        "Monitor my inbox for urgent emails.",
        "Check my NTULearn announcements and assignments.",
        "Summarize my day and decide what requires my attention first.",
    ]:
        result = orchestrator.run(request)
        print(f"\nRequest: {request}\nResult: {result}\n")


if __name__ == "__main__":
    main()
