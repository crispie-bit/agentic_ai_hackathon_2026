from agentic_system.services.aws_setup import aws_status


def test_aws_status_validates_real_credentials(monkeypatch):
    class FakeSTSClient:
        def get_caller_identity(self):
            return {"Arn": "arn:aws:iam::123456789012:user/demo-user"}

    class FakeSession:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def client(self, service_name, region_name=None):
            assert service_name == "sts"
            assert region_name == "ap-southeast-1"
            return FakeSTSClient()

    monkeypatch.setattr("agentic_system.services.aws_setup.boto3", type("FakeBoto3", (), {"Session": FakeSession}))

    status = aws_status(
        access_key_id="AKIA_TEST",
        secret_key="secret",
        session_token="token",
        region="ap-southeast-1",
    )

    assert status["status"] == "ready"
    assert status["ready"] is True
    assert "demo-user" in status["account_arn"]
