"""
Manual validation script for the DB + real-time WebSocket layer added to
safesignal.py (database.py / models.py / realtime.py). Run with:
    python test_realtime_pipeline.py

Deliberately forces decision_graph = None right after startup -- the
existing stub/fallback branch in /api/v1/decision-agent, unchanged by this
migration -- so this script is independent of GOOGLE_API_KEY / MCP-server /
Bedrock availability. Exercising the real LLM-driven graph is test_graph.py's
job; this script's job is only to verify: a POST to /api/v1/decision-agent
(1) commits an Incident + ExtractedEntities row to the DB and (2) broadcasts
a matching INCIDENT_NEW frame over /api/v1/realtime, then that a PATCH to
/api/v1/incidents/{id}/status updates the row and broadcasts INCIDENT_UPDATE.

Note: startup still runs the full safesignal.py lifespan (HeBERT/Comprehend/
Bedrock/RAG-index warm-up), so this takes as long as any other safesignal.py
boot -- see project memory on model-load time.
"""
import asyncio
import os
import uuid

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./data/test_realtime.db")

from fastapi.testclient import TestClient

import safesignal
from database import AsyncSessionLocal
from models import ExtractedEntities, Incident


def _receive_until(ws, msg_type: str, max_frames: int = 5) -> dict:
    """Skips past any SYSTEM_STATUS_CHANGE heartbeat frames that happen to
    land between the frame we're waiting for and when we started listening."""
    for _ in range(max_frames):
        frame = ws.receive_json()
        if frame.get("type") == msg_type:
            return frame
    raise AssertionError(f"did not receive a {msg_type} frame within {max_frames} messages")


def run_validation():
    incident_id = str(uuid.uuid4())
    payload = {
        "incident_id": incident_id,
        "user_id": "test-user-1",
        "platform": "telegram",
        "text_content": "בדיקה: אני מרגיש מאוד לבד ומיואש בזמן האחרון",
        "names": ["אין שם"],
        "addresses": ["באר שבע"],
        "ages": [17],
        "phone_numbers": [],
        "risk_level": "medium",
    }

    with TestClient(safesignal.app) as client:
        safesignal.decision_graph = None  # deterministic stub fallback -- see module docstring

        with client.websocket_connect("/api/v1/realtime") as ws:
            response = client.post("/api/v1/decision-agent", json=payload)
            assert response.status_code == 200, response.text
            print("POST /api/v1/decision-agent ->", response.status_code)

            new_frame = _receive_until(ws, "INCIDENT_NEW")
            assert new_frame["payload"]["incidentId"] == incident_id, new_frame
            print(
                "Received INCIDENT_NEW:",
                new_frame["payload"]["incidentId"],
                new_frame["payload"]["riskLevel"],
            )

            patch_response = client.patch(
                f"/api/v1/incidents/{incident_id}/status",
                json={"status": "investigating", "isUnread": False},
            )
            assert patch_response.status_code == 200, patch_response.text
            print("PATCH status ->", patch_response.status_code, patch_response.json())

            update_frame = _receive_until(ws, "INCIDENT_UPDATE")
            assert update_frame["payload"]["incidentId"] == incident_id
            assert update_frame["payload"]["changes"]["status"] == "investigating"
            print("Received INCIDENT_UPDATE:", update_frame["payload"]["changes"])

    async def _verify_db():
        async with AsyncSessionLocal() as session:
            incident = await session.get(Incident, incident_id)
            entities = await session.get(ExtractedEntities, incident_id)
            assert incident is not None, "Incident row missing from DB"
            assert incident.status == "investigating"
            assert incident.is_unread is False
            assert entities is not None, "ExtractedEntities row missing from DB"
            assert entities.addresses == ["באר שבע"]
            print(
                f"DB verified: incident_id={incident.incident_id} status={incident.status} "
                f"is_unread={incident.is_unread} entities.ages={entities.ages}"
            )

    asyncio.run(_verify_db())
    print("\nAll checks passed.")


if __name__ == "__main__":
    run_validation()
