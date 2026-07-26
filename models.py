"""SQLAlchemy ORM models for the SafeSignal incidents store."""
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Incident(Base):
    __tablename__ = "incidents"

    incident_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, default="anonymous")
    platform: Mapped[str] = mapped_column(String, default="telegram")
    raw_text: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    risk_level: Mapped[str] = mapped_column(String, default="unknown")
    is_unread: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String, default="new")

    distress_classification: Mapped[str] = mapped_column(String, default="")
    summary: Mapped[str] = mapped_column(String, default="")
    thought_process: Mapped[str] = mapped_column(String, default="")
    recommended_action: Mapped[str] = mapped_column(String, default="")
    final_urgency_assessment: Mapped[str] = mapped_column(String, default="")

    screening_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    screening_reason: Mapped[str] = mapped_column(String, default="")
    screening_tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    tools_triggered: Mapped[list[str]] = mapped_column(JSON, default=list)
    screening_logs: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)

    entities: Mapped["ExtractedEntities"] = relationship(
        back_populates="incident", uselist=False, cascade="all, delete-orphan"
    )


class ExtractedEntities(Base):
    __tablename__ = "extracted_entities"

    incident_id: Mapped[str] = mapped_column(
        String, ForeignKey("incidents.incident_id"), primary_key=True
    )
    names: Mapped[list[str]] = mapped_column(JSON, default=list)
    ages: Mapped[list[int]] = mapped_column(JSON, default=list)
    phone_numbers: Mapped[list[str]] = mapped_column(JSON, default=list)
    addresses: Mapped[list[str]] = mapped_column(JSON, default=list)

    incident: Mapped["Incident"] = relationship(back_populates="entities")


class IngestionSettings(Base):
    """
    Singleton row (always id=1) holding the Telegram scan's configurable
    message_limit/interval_minutes, plus last_scan_at -- the state
    GET /api/v1/settings/scan-check reads/updates to enforce that interval
    from outside n8n's own (static-only) Schedule Trigger. See safesignal.py's
    scan-check endpoint for the gate logic this backs.
    """

    __tablename__ = "ingestion_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    message_limit: Mapped[int] = mapped_column(Integer, default=10)
    interval_minutes: Mapped[int] = mapped_column(Integer, default=20)
    last_scan_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
