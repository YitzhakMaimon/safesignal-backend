"""Pydantic schemas for the SafeSignal Decision Agent LangGraph workflow."""
from typing import Annotated, Any, Dict, List, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


class AgentState(BaseModel):
    """Global graph state threaded through the Decision Agent LangGraph workflow."""

    # Fields carried in from earlier pipeline stages (screening, classification, RAG).
    raw_input: str
    user_id: str
    incident_id: str
    distress_classification: str
    initial_rag_context: str

    # Identifying details pulled out by Relevant Information Extraction
    # (info_extraction.py, local Ollama NER) earlier in the pipeline. Threaded
    # through so safesignal.py's _persist_and_broadcast_incident can store
    # them in the extracted_entities DB table, and so immediate_alert_node
    # can include them in the high-risk alert webhook payload (the human
    # responder needs identifying details for an emergency page). Never
    # written to S3 -- log_raw_message's S3 archive only carries
    # incident_id/user_id/text_content/timestamp.
    names: List[str] = Field(default_factory=list)
    addresses: List[str] = Field(default_factory=list)
    ages: List[int] = Field(default_factory=list)
    phone_numbers: List[str] = Field(default_factory=list)

    # Conversation/tool-call history. Named "messages" because the prebuilt ToolNode
    # reads and writes that key by default; add_messages appends rather than overwrites.
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)

    # Mutable evaluation fields populated by decision_agent_node.
    thought_process: str = ""
    tools_triggered: List[str] = Field(default_factory=list)
    final_urgency_assessment: str = ""
    recommended_action: str = ""
    summary_for_human_reviewer: str = ""
    # Model's own confidence in final_urgency_assessment, 0.0-1.0. None until
    # decision_agent_node's structured-output pass runs (e.g. still mid tool-call
    # loop) -- the Alert History table renders that as a blank cell, not "0%".
    confidence_score: float | None = None

    # Populated by output_screening_node: Bedrock Guardrails + Claude Haiku
    # hallucination check on the Decision Agent's own generated text, run
    # before the "Routing by Risk Level" step reads the final graph result.
    screening_passed: bool = True
    screening_tags: List[str] = Field(default_factory=list)
    screening_reason: str = ""
    # One entry per output-screening attempt (guardrail + hallucination-check
    # results, pass/fail) -- kept across retries so a human reviewer can see
    # exactly why earlier drafts of the agent's answer were rejected.
    screening_logs: List[Dict[str, Any]] = Field(default_factory=list)
    # How many times output_screening has sent the agent back to redo its
    # answer. Reset to 0 on a pass; capped at MAX_OUTPUT_RETRIES before
    # escalating straight to human_review instead of looping forever.
    output_retry_count: int = 0
    # Internal routing signal set by output_screening_node for its own
    # conditional edge ("decision_agent" | "human_review" | "routing_by_risk_level").
    output_screening_route: str = ""

    # Populated by routing_by_risk_level_node. One of "high" / "medium" / "low".
    risk_level: str = ""


class IncidentStatusUpdate(BaseModel):
    """Validates the PATCH /api/v1/incidents/{id}/status request body sent by
    the dashboard's action bar (Acknowledge/Escalate/False Positive/Close)."""

    status: Literal["investigating", "false_positive", "escalated", "closed"]
    is_unread: bool | None = Field(default=None, alias="isUnread")

    model_config = ConfigDict(populate_by_name=True)



class DecisionOutput(BaseModel):
    """Structured output schema forced on the LLM via `.with_structured_output()`."""

    thought_process: str = Field(description="Step-by-step reasoning behind the assessment.")
    tools_triggered: List[str] = Field(
        default_factory=list,
        description=(
            "Names of tools called during this node's execution, "
            "e.g. ['trigger_immediate_alert']. Empty list if none were needed."
        ),
    )
    # Lowercase, 3-level -- matches routing_by_risk_level_node's risk_level
    # vocabulary exactly, so the graph never carries two differently-cased,
    # differently-sized urgency scales side by side (see decision_agent_graph.py
    # URGENCY_TO_RISK_LEVEL, which used to need to collapse a 4th "Critical"
    # tier down to "high" -- that tier no longer exists at the source).
    final_urgency_assessment: Literal["low", "medium", "high"] = Field(
        description="Final urgency level determined for this incident."
    )
    recommended_action: str = Field(description="Short description of the next routing step.")
    summary_for_human_reviewer: str = Field(
        description="Factual summary of the incident for the manual control dashboard."
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your confidence in final_urgency_assessment, from 0.0 (a guess) to "
            "1.0 (certain), based on how clearly the raw input and context support it."
        ),
    )
