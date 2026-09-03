from agentic_system.agents.outlook_agent import OutlookAgent


def test_outlook_agent_prioritizes_urgent_messages():
    agent = OutlookAgent()

    response = agent.run(
        "Monitor my inbox for urgent emails.",
        {
            "messages": [
                {
                    "sender": "news@daily-update.com",
                    "subject": "Weekly digest",
                    "body": "Here are all the stories from this week.",
                },
                {
                    "sender": "manager@company.com",
                    "subject": "Action required today",
                    "body": "Please review the proposal and respond by 3pm today.",
                },
            ]
        },
    )

    assert "Action required today" in response
    assert "urgent" in response.lower()
    assert "Weekly digest" not in response
