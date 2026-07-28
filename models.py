"""SQLAlchemy ORM models for the SafeSignal incidents store."""
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
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

    # Total tokens (input+output, summed across every model call site, from
    # Image Analysis/Voice Transcription at n8n's entry through the Decision
    # Agent's hallucination check at its exit) spent classifying this incident
    # end-to-end -- read back from
    # local_storage.get_total_tokens_for_sentence(raw_text, incident_id) once
    # the decision graph finishes, since that's the same data
    # data/tokens_log.xlsx already tracks per-incident for the token-usage
    # audit trail. Nullable and defaults to None (not 0) -- an incident with
    # no token record (e.g. logged before token tracking existed) has no
    # known usage, which the History screen renders as a blank cell rather
    # than a misleading "0 tokens used".
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)

    # Decision Agent LLM's own confidence in final_urgency_assessment, 0.0-1.0
    # (see DecisionOutput.confidence_score in schemas.py). Nullable/defaults to
    # None, not 0.0 -- an incident whose graph run never reached the structured
    # decision step (e.g. the decision_graph-unavailable stub path in
    # safesignal.py) has no real score, which the History screen renders as a
    # blank cell rather than a misleading "0%".
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)

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
