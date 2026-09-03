"""Microsoft Entra / Azure App Registration helpers for Graph-backed Outlook access."""

from __future__ import annotations

import os
from urllib.parse import urlencode

DEFAULT_OUTLOOK_SCOPE = "https://graph.microsoft.com/Mail.Read offline_access"


def validate_azure_config() -> dict[str, str | bool]:
    """Validate the Microsoft Graph App Registration settings required for Outlook access."""
    tenant_id = (os.getenv("MICROSOFT_TENANT_ID") or "").strip()
    client_id = (os.getenv("MICROSOFT_CLIENT_ID") or "").strip()
    redirect_uri = (os.getenv("MICROSOFT_REDIRECT_URI") or "").strip()
    missing = []

    if not tenant_id:
        missing.append("MICROSOFT_TENANT_ID")
    if not client_id:
        missing.append("MICROSOFT_CLIENT_ID")
    if not redirect_uri:
        missing.append("MICROSOFT_REDIRECT_URI")

    return {
        "ready": not missing,
        "missing": missing,
        "tenant_id": tenant_id,
        "client_id": client_id,
        "redirect_uri": redirect_uri,
    }


def azure_status() -> dict[str, str | bool | list[str]]:
    """Return the current Microsoft Graph app registration status."""
    validated = validate_azure_config()

    if not validated["ready"]:
        return {
            "status": "missing_app_registration",
            "ready": False,
            "tenant_id": validated["tenant_id"],
            "client_id": validated["client_id"],
            "redirect_uri": validated["redirect_uri"],
            "missing": validated["missing"],
            "note": "Create an Azure App Registration and set MICROSOFT_TENANT_ID, MICROSOFT_CLIENT_ID, and MICROSOFT_REDIRECT_URI.",
        }

    return {
        "status": "ready",
        "ready": True,
        "tenant_id": validated["tenant_id"],
        "client_id": validated["client_id"],
        "redirect_uri": validated["redirect_uri"],
        "missing": [],
        "note": "App registration is configured for Microsoft Graph Outlook access.",
    }


def build_graph_auth_url() -> str:
    """Build the interactive Microsoft Entra auth URL for first-run Outlook login."""
    tenant_id = (os.getenv("MICROSOFT_TENANT_ID") or "").strip()
    client_id = (os.getenv("MICROSOFT_CLIENT_ID") or "").strip()
    redirect_uri = (os.getenv("MICROSOFT_REDIRECT_URI") or "http://localhost").strip()

    if not tenant_id or not client_id:
        raise ValueError("MICROSOFT_TENANT_ID and MICROSOFT_CLIENT_ID must be set before starting Graph auth.")

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "response_mode": "query",
        "scope": DEFAULT_OUTLOOK_SCOPE,
        "state": "outlook-agent-auth",
    }
    return "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize?{query}".format(
        tenant=tenant_id,
        query=urlencode(params),
    )
