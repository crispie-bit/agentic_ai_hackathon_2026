"""Microsoft Graph-based Outlook mailbox connector."""

from __future__ import annotations

import os
from typing import Any

import requests

try:
    import msal
except ImportError:  # pragma: no cover
    msal = None

GRAPH_ME_MESSAGES_URL = "https://graph.microsoft.com/v1.0/me/messages?$select=from,subject,bodyPreview,receivedDateTime&$top={limit}"


def _get_graph_token() -> str:
    tenant_id = os.getenv("MICROSOFT_TENANT_ID")
    client_id = os.getenv("MICROSOFT_CLIENT_ID")

    if not tenant_id or not client_id:
        raise ValueError("MICROSOFT_TENANT_ID and MICROSOFT_CLIENT_ID must be configured before first-run Outlook auth.")

    if msal is None:
        raise ImportError("The msal package is required for Microsoft Graph authentication. Install it with pip install msal.")

    app = msal.PublicClientApplication(client_id=client_id, authority=f"https://login.microsoftonline.com/{tenant_id}")
    result = app.acquire_token_interactive(scopes=["Mail.Read"])

    access_token = result.get("access_token") if isinstance(result, dict) else None
    if not access_token:
        raise RuntimeError("Microsoft Graph authentication did not return an access token.")

    return access_token


def fetch_outlook_messages(limit: int = 10) -> list[dict[str, Any]]:
    """Fetch recent mailbox messages from Microsoft Graph for triage."""
    token = _get_graph_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.get(GRAPH_ME_MESSAGES_URL.format(limit=limit), headers=headers, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"Graph request failed with status {response.status_code}: {response.text}")

    payload = response.json()
    messages: list[dict[str, Any]] = []
    for item in payload.get("value", []):
        sender = item.get("from", {}).get("emailAddress", {}).get("address", "unknown")
        messages.append(
            {
                "sender": sender,
                "subject": item.get("subject", "No subject"),
                "body": item.get("bodyPreview", ""),
                "received_at": item.get("receivedDateTime"),
            }
        )

    return messages
