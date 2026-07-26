"""
FastMCP server exposing the immediate-alert tool for the SafeSignal Decision Agent.

Forwards escalation requests to the n8n workflow, which is responsible for placing
the Amazon Polly voice call / notification (n8n.json, "immediate-alert" webhook).
Run standalone: `python alert_mcp.py` (serves Streamable HTTP on :8011/mcp by default).

Moved off :8001 (2026-07-22) -- that port is now services/rag_service's HTTP
API in the 4-service microservices split; running both on :8001 would collide.
"""
import os

import requests
from mcp.server.fastmcp import FastMCP

ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "http://localhost:5678/webhook/immediate-alert")
MCP_PORT = int(os.environ.get("ALERT_MCP_PORT", "8011"))

mcp = FastMCP("safesignal-alert", host="0.0.0.0", port=MCP_PORT)


@mcp.tool()
def trigger_immediate_alert(incident_id: str, urgency_reason: str) -> str:
    """
    Trigger an immediate human/Amazon Polly voice alert for a high-risk incident.

    Call this only when the situation requires urgent human intervention (e.g.
    High urgency with an imminent safety risk) -- it pages a human
    responder, so it should not be used for routine or low/medium urgency cases.

    Args:
        incident_id: Unique identifier of the incident being escalated.
        urgency_reason: Short factual explanation of why immediate escalation is required.
    """
    payload = {"incident_id": incident_id, "urgency_reason": urgency_reason}

    try:
        response = requests.post(ALERT_WEBHOOK_URL, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        return f"ALERT_FAILED: could not reach alert webhook for incident '{incident_id}': {e}"

    if response.status_code == 200:
        return f"ALERT_SENT: immediate alert successfully triggered for incident '{incident_id}'."

    return (
        f"ALERT_FAILED: alert webhook returned status {response.status_code} "
        f"for incident '{incident_id}': {response.text}"
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
