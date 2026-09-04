"""Small HTTP API for the project agent."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .agent import ProjectAgent
from .models import CustomerSignal, Ticket


def make_handler(agent: ProjectAgent) -> type[BaseHTTPRequestHandler]:
    class AgentHandler(BaseHTTPRequestHandler):
        server_version = "HackathonAgent/0.1"

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json({"ok": True})

        def do_POST(self) -> None:
            if self.path != "/handle":
                self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self._read_json()
                decision = agent.handle(_ticket_from_payload(payload))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json({"error": "bad_request", "detail": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(decision.to_dict())

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("request body is required")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise TypeError("request body must be a JSON object")
            return body

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return AgentHandler


def run_server(host: str = "127.0.0.1", port: int = 8765, agent: ProjectAgent | None = None) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(agent or ProjectAgent()))
    print(f"serving hackathon_agent on http://{host}:{port}")
    httpd.serve_forever()


def _ticket_from_payload(payload: dict[str, Any]) -> Ticket:
    text = payload.get("ticket") or payload.get("text")
    if not isinstance(text, str):
        raise ValueError("payload needs string field 'ticket' or 'text'")
    customer_payload = payload.get("customer") or {}
    if not isinstance(customer_payload, dict):
        raise TypeError("'customer' must be a JSON object when provided")
    customer = CustomerSignal(
        name=str(customer_payload.get("name", "customer")),
        tier=str(customer_payload.get("tier", "standard")),
        region=customer_payload.get("region"),
        account_age_days=customer_payload.get("account_age_days"),
    )
    return Ticket(text, customer=customer, channel=str(payload.get("channel", "api")))

