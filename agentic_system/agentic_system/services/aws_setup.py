"""AWS integration stub.

This file intentionally does not call AWS on startup. It is a safe placeholder for
future Bedrock or service integration work.
"""


def aws_status() -> dict[str, str]:
    return {
        "status": "not_started",
        "mode": "setup_only",
        "note": "Enable AWS integration only after credentials and permissions are configured.",
    }
