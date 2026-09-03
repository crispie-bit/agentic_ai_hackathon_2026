from agentic_system.tools.outlook_tool import fetch_outlook_messages


def test_fetch_outlook_messages_uses_graph_api(monkeypatch):
    class FakeApp:
        def __init__(self, client_id, authority):
            self.client_id = client_id
            self.authority = authority

        def acquire_token_interactive(self, scopes):
            return {"access_token": "fake-token"}

    class FakeResponse:
        def __init__(self):
            self.status_code = 200
            self._payload = {
                "value": [
                    {
                        "from": {"emailAddress": {"address": "manager@company.com"}},
                        "subject": "Action required today",
                        "bodyPreview": "Please review by 3pm today.",
                        "receivedDateTime": "2026-09-03T15:00:00Z",
                    }
                ]
            }

        def json(self):
            return self._payload

    class FakeRequestsModule:
        @staticmethod
        def get(url, headers=None, timeout=30):
            return FakeResponse()

    monkeypatch.setenv("MICROSOFT_TENANT_ID", "tenant-123")
    monkeypatch.setenv("MICROSOFT_CLIENT_ID", "client-123")
    monkeypatch.setattr("agentic_system.tools.outlook_tool.msal", type("FakeMSAL", (), {"PublicClientApplication": FakeApp}))
    monkeypatch.setattr("agentic_system.tools.outlook_tool.requests", FakeRequestsModule)

    messages = fetch_outlook_messages(limit=5)

    assert len(messages) == 1
    assert messages[0]["subject"] == "Action required today"
    assert messages[0]["sender"] == "manager@company.com"
