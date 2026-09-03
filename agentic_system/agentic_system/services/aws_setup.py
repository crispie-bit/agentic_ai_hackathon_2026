"""AWS credential validation and runtime setup helpers."""

from __future__ import annotations

import os
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


def configure_aws_credentials(
    *,
    access_key_id: str | None = None,
    secret_key: str | None = None,
    session_token: str | None = None,
    region: str | None = None,
) -> dict[str, Any]:
    """Apply AWS credentials from arguments or the environment and validate them."""
    resolved_access_key = (access_key_id or os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
    resolved_secret_key = (secret_key or os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
    resolved_session_token = (session_token or os.getenv("AWS_SESSION_TOKEN") or "").strip()
    resolved_region = (region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-southeast-1").strip()

    if resolved_access_key:
        os.environ["AWS_ACCESS_KEY_ID"] = resolved_access_key
    if resolved_secret_key:
        os.environ["AWS_SECRET_ACCESS_KEY"] = resolved_secret_key
    if resolved_session_token:
        os.environ["AWS_SESSION_TOKEN"] = resolved_session_token
    os.environ["AWS_REGION"] = resolved_region
    os.environ["AWS_DEFAULT_REGION"] = resolved_region

    return aws_status(
        access_key_id=resolved_access_key,
        secret_key=resolved_secret_key,
        session_token=resolved_session_token,
        region=resolved_region,
    )


def aws_status(
    *,
    access_key_id: str | None = None,
    secret_key: str | None = None,
    session_token: str | None = None,
    region: str | None = None,
) -> dict[str, Any]:
    """Validate AWS credentials and report runtime readiness."""
    resolved_access_key = (access_key_id or os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
    resolved_secret_key = (secret_key or os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()
    resolved_session_token = (session_token or os.getenv("AWS_SESSION_TOKEN") or "").strip()
    resolved_region = (region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-southeast-1").strip()

    if not resolved_access_key or not resolved_secret_key:
        return {
            "status": "missing_credentials",
            "ready": False,
            "mode": "setup_only",
            "region": resolved_region,
            "note": "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY before enabling AWS features.",
        }

    try:
        session = boto3.Session(
            aws_access_key_id=resolved_access_key,
            aws_secret_access_key=resolved_secret_key,
            aws_session_token=resolved_session_token or None,
            region_name=resolved_region,
        )
        identity = session.client("sts", region_name=resolved_region).get_caller_identity()
        arn = identity.get("Arn", "")
        return {
            "status": "ready",
            "ready": True,
            "mode": "live",
            "region": resolved_region,
            "account_arn": arn,
            "note": "AWS credentials were validated successfully.",
        }
    except (NoCredentialsError, ClientError, BotoCoreError, Exception) as exc:
        return {
            "status": "invalid_credentials",
            "ready": False,
            "mode": "setup_only",
            "region": resolved_region,
            "note": f"AWS credentials are invalid or inaccessible: {exc}",
        }
