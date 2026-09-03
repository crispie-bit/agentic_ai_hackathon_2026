from agentic_system.services.azure_setup import azure_status, build_graph_auth_url


def test_azure_status_requires_graph_app_registration(monkeypatch):
    monkeypatch.setenv("MICROSOFT_TENANT_ID", "tenant-123")
    monkeypatch.setenv("MICROSOFT_CLIENT_ID", "client-123")
    monkeypatch.setenv("MICROSOFT_REDIRECT_URI", "http://localhost")

    status = azure_status()

    assert status["status"] == "ready"
    assert status["ready"] is True
    assert status["tenant_id"] == "tenant-123"
    assert status["client_id"] == "client-123"
    assert status["redirect_uri"] == "http://localhost"


def test_azure_status_requires_redirect_uri(monkeypatch):
    monkeypatch.setenv("MICROSOFT_TENANT_ID", "tenant-123")
    monkeypatch.setenv("MICROSOFT_CLIENT_ID", "client-123")
    monkeypatch.delenv("MICROSOFT_REDIRECT_URI", raising=False)

    status = azure_status()

    assert status["status"] == "missing_app_registration"
    assert status["ready"] is False


def test_build_graph_auth_url_includes_required_parameters(monkeypatch):
    monkeypatch.setenv("MICROSOFT_TENANT_ID", "tenant-123")
    monkeypatch.setenv("MICROSOFT_CLIENT_ID", "client-123")
    monkeypatch.setenv("MICROSOFT_REDIRECT_URI", "http://localhost")

    url = build_graph_auth_url()

    assert "tenant-123" in url
    assert "client-123" in url
    assert "Mail.Read" in url
    assert "redirect_uri=http%3A%2F%2Flocalhost" in url
