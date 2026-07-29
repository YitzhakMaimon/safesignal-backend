"""
WebSocket connection manager + broadcast payload builders for the SafeSignal
real-time dashboard contract (/api/v1/realtime).

Kept separate from safesignal.py so the n8n-facing HTTP routes there stay
exactly as they are today -- this module is imported and called *from inside*
those existing handlers, not a replacement for them.
"""
import asyncio
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import WebSocket

from distress_classification import category_code_for
from models import ExtractedEntities, Incident


def utc_iso(dt: datetime) -> str:
    """
    SQLite silently drops tzinfo on read through SQLAlchemy even for a
    DateTime(timezone=True) column -- every value in this column is written
    as UTC (see models.py's _utcnow / safesignal.py's content_fields), so a
    naive value read back is always really UTC, just missing the marker.
    Without this, .isoformat() on the naive datetime produces a string with
    no 'Z'/offset, which browsers parse as *local* time -- silently skewing
    "time ago" displays by the client's UTC offset (caught via a live demo
    incident showing "3h ago" seconds after creation, Israel being UTC+3).
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


class ConnectionManager:
    """Tracks concurrently connected dashboard operators and broadcasts frames
    to all of them, dropping any socket whose send fails (safe disconnect
    handling without requiring the caller to track sockets itself)."""

    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        for websocket in list(self.active):
            try:
                await websocket.send_json(message)
            except Exception:
                self.active.discard(websocket)


manager = ConnectionManager()


def serialize_incident(incident: Incident, entities: ExtractedEntities) -> dict:
    """
    Translates an Incident + ExtractedEntities row pair into the frontend's
    Incident shape (camelCase, see safesignal-dashboard's src/types/incident.ts)
    -- the one place this mapping is defined, shared by the INCIDENT_NEW
    WebSocket frame and the GET /api/v1/incidents history endpoint so the two
    can never drift apart.
    """
    return {
        "incidentId": incident.incident_id,
        "userId": incident.user_id,
        "platform": incident.platform,
        "rawText": incident.raw_text,
        "createdAt": utc_iso(incident.created_at),
        "updatedAt": utc_iso(incident.updated_at),
        "riskLevel": incident.risk_level,
        "distressCategory": category_code_for(incident.distress_classification),
        "isUnread": incident.is_unread,
        "status": incident.status,
        "entities": {
            "names": entities.names,
            "ages": entities.ages,
            "phoneNumbers": entities.phone_numbers,
            "addresses": entities.addresses,
        },
        "summary": incident.summary,
        "thoughtProcess": incident.thought_process,
    }


ISRAEL_TZ = ZoneInfo("Asia/Jerusalem")


def format_history_timestamp(dt: datetime) -> str:
    """YYYY-MM-DD HH:mm:ss in Israel local time (IST/IDT, DST-aware via
    zoneinfo), for the Alert History table's Timestamp column -- distinct
    from utc_iso (ISO 8601 UTC), which is what the live dashboard's
    Incident.createdAt uses instead and which the browser itself localizes."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ISRAEL_TZ).strftime("%Y-%m-%d %H:%M:%S")


def serialize_history_row(incident: Incident, log_id: int) -> dict:
    """
    Row shape for GET /api/v1/incidents/history -- the Alert History screen's
    server-side paginated/sorted/filtered table. Deliberately not
    serialize_incident's shape: that one feeds the live dashboard's full
    Incident domain object, this returns the eight columns the History
    table renders plus riskLevel (used by the Severity Level chart above the
    table, not rendered as its own column), pre-formatted for direct display
    (no client-side date math or truncation needed).

    log_id comes from SQLite's implicit rowid (see the /history query in
    safesignal.py) rather than a new schema column -- incident_id is a
    string UUID, not a sequential int, and rowid already gives a stable,
    monotonically-increasing id for free on every ordinary SQLite table.
    """
    return {
        "logId": log_id,
        "timestamp": format_history_timestamp(incident.created_at),
        "userOrIp": incident.user_id,
        "platform": incident.platform,
        "category": category_code_for(incident.distress_classification),
        "detectedText": incident.raw_text,
        "resolution": incident.status,
        "riskLevel": incident.risk_level,
        "tokens": incident.tokens_used,
        "confidenceScore": incident.confidence_score,
    }


def incident_new_payload(incident: Incident, entities: ExtractedEntities) -> dict:
    """Builds the exact INCIDENT_NEW frame from ORM rows."""
    return {"type": "INCIDENT_NEW", "payload": serialize_incident(incident, entities)}


def incident_update_payload(incident_id: str, changes: dict) -> dict:
    return {
        "type": "INCIDENT_UPDATE",
        "payload": {"incidentId": incident_id, "changes": changes},
    }


# --- Lightweight in-process telemetry for the SYSTEM_STATUS_CHANGE heartbeat ---
# No real load/token/confidence metrics are instrumented anywhere upstream yet
# (same "not_implemented, always X" honesty convention as safesignal.py's
# _ml_output_classifier) -- aiLoadPercent/tokensPerMinute are a real proxy
# derived from actual decision-agent invocation counts, not fabricated;
# modelConfidencePercent is a fixed honest baseline until a real confidence
# signal exists upstream.
_recent_invocations: list[float] = []
_MODEL_CONFIDENCE_BASELINE = 90


def record_invocation() -> None:
    _recent_invocations.append(time.time())


def _prune_and_count(window_seconds: float) -> int:
    cutoff = time.time() - window_seconds
    while _recent_invocations and _recent_invocations[0] < cutoff:
        _recent_invocations.pop(0)
    return sum(1 for t in _recent_invocations if t >= cutoff)


async def system_status_loop() -> None:
    """Background heartbeat: broadcasts SYSTEM_STATUS_CHANGE every 3s until cancelled."""
    try:
        while True:
            await asyncio.sleep(3)
            calls_last_3s = _prune_and_count(3)
            calls_last_60s = _prune_and_count(60)
            await manager.broadcast({
                "type": "SYSTEM_STATUS_CHANGE",
                "payload": {
                    "connectivity": "operational",
                    "aiLoadPercent": min(100, calls_last_3s * 20),
                    "tokensPerMinute": calls_last_60s * 250,
                    "modelConfidencePercent": _MODEL_CONFIDENCE_BASELINE,
                },
            })
    except asyncio.CancelledError:
        pass
