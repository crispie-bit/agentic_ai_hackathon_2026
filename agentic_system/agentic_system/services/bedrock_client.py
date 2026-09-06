"""AWS Bedrock runtime client — ported from harvey-workday-os-v2/services/aws_bedrock.py.

Provides a thin BedrockClient wrapper around the Bedrock Converse API.
The module-level ``bedrock_client`` singleton is imported by ``answering.py``
and ``app.py``.
"""

from __future__ import annotations

import os
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from agentic_system.config import (
    AWS_ACCESS_KEY_ID,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    AGENT_MODEL_IDS,
)

# Resolve model IDs once at import time.
LEAD_MODEL_ID = AGENT_MODEL_IDS["lead_agent"]
SPECIALIST_MODEL_ID = AGENT_MODEL_IDS["ntulearn_agent"]


class BedrockClient:
    """Thin wrapper around boto3 bedrock-runtime for the Converse API."""

    def __init__(self) -> None:
        self._session: boto3.Session | None = None
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        access_key = (AWS_ACCESS_KEY_ID or os.getenv("AWS_ACCESS_KEY_ID", "")).strip()
        secret_key = (AWS_SECRET_ACCESS_KEY or os.getenv("AWS_SECRET_ACCESS_KEY", "")).strip()
        session_token = (AWS_SESSION_TOKEN or os.getenv("AWS_SESSION_TOKEN", "")).strip()
        region = (AWS_REGION or os.getenv("AWS_REGION", "us-east-1")).strip()

        if not access_key or not secret_key:
            return  # No credentials; remain in offline mode

        try:
            # Remove AWS_PROFILE when explicit keys are present to avoid
            # botocore ProfileNotFound errors — same fix Harvey applied.
            if os.environ.get("AWS_PROFILE") == "default":
                os.environ.pop("AWS_PROFILE", None)

            self._session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                aws_session_token=session_token or None,
                region_name=region,
            )
            self._client = self._session.client("bedrock-runtime", region_name=region)
        except Exception:
            self._client = None

    def is_ready(self) -> bool:
        return self._client is not None

    def get_account_identity(self) -> dict[str, Any]:
        """Return STS caller identity for status display."""
        if not self._session:
            return {"ready": False, "error": "No AWS credentials configured."}
        try:
            region = (AWS_REGION or "us-east-1").strip()
            sts = self._session.client("sts", region_name=region)
            identity = sts.get_caller_identity()
            return {
                "ready": True,
                "arn": identity.get("Arn", ""),
                "region": region,
            }
        except Exception as exc:
            return {"ready": False, "error": str(exc)}

    def converse(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        model_id: str | None = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        """Call Bedrock Converse API and return the assistant text reply.

        Falls back to an offline warning string when credentials are missing.
        """
        if not self._client:
            self._init_client()
        if not self._client:
            return (
                "⚠️ AWS Bedrock is not configured. "
                "Add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to .env."
            )

        selected_model = model_id or LEAD_MODEL_ID

        # Normalise messages to Bedrock's content-block format.
        formatted: list[dict[str, Any]] = []
        for m in messages:
            content = m.get("content", "")
            role = m.get("role", "user")
            if isinstance(content, str):
                formatted.append({"role": role, "content": [{"text": content}]})
            elif isinstance(content, list):
                formatted.append({"role": role, "content": content})

        kwargs: dict[str, Any] = {
            "modelId": selected_model,
            "messages": formatted,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        try:
            response = self._client.converse(**kwargs)
            output = response.get("output", {}).get("message", {})
            for block in output.get("content", []):
                if block.get("text"):
                    return block["text"]
            return ""
        except (NoCredentialsError, ClientError, BotoCoreError) as exc:
            return f"⚠️ Bedrock call failed: {exc}"
        except Exception as exc:
            return f"⚠️ Unexpected Bedrock error: {exc}"


def aws_bedrock_status() -> dict[str, Any]:
    """Return a status dict for the Streamlit sidebar."""
    client = BedrockClient()
    if not client.is_ready():
        return {
            "ready": False,
            "status": "missing_credentials",
            "note": "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env.",
        }
    identity = client.get_account_identity()
    if identity.get("ready"):
        return {
            "ready": True,
            "status": "ready",
            "arn": identity.get("arn", ""),
            "region": identity.get("region", ""),
            "note": "AWS Bedrock credentials validated successfully.",
        }
    return {
        "ready": False,
        "status": "invalid_credentials",
        "note": identity.get("error", "Unknown error."),
    }


# Module-level singleton used by answering.py and app.py
bedrock_client = BedrockClient()
