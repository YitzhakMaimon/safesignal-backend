"""
Minimal stub backend for the SafeSignal n8n pipeline (n8n.json).

Purpose: unblock end-to-end smoke testing of the n8n workflow before the
real Comprehend / Bedrock / LangChain / LangGraph integrations exist.
Every endpoint echoes back whatever it received and merges in a generous
set of placeholder fields, so downstream nodes that read fields set by
*earlier* stages (not just the immediately preceding one) always find
something. This is NOT how the real backend should behave -- the real
implementation should have each n8n node explicitly reference the node
that actually owns a field (e.g. $node["Distress Classification"].json.summary)
instead of relying on every hop blindly forwarding everything.
"""
import asyncio
import base64
import math
import os
import uuid
from calendar import monthrange
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timedelta, timezone
from xml.sax.saxutils import escape as xml_escape

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import asc, case, desc, func, literal_column, or_, select
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client as TwilioClient
import uvicorn

from ml_comprehend import DistressScreeningPipeline
from info_extraction import RelevantInfoExtractor
from distress_classification import BedrockDistressClassifier, CLASS_TO_CATEGORY_CODE
from rag_retrieval import RAGContextRetriever
from decision_agent_graph import create_decision_agent, log_raw_message
from local_storage import (
    save_error_record,
    get_error_records,
    save_false_positive_record,
    save_token_usage_record,
    get_total_tokens_for_sentence,
)
from database import get_session, init_db, upsert_stmt
from models import ExtractedEntities, Incident
from realtime import (
    incident_new_payload,
    incident_update_payload,
    manager,
    record_invocation,
    serialize_history_row,
    serialize_incident,
    system_status_loop,
    utc_iso,
)
from schemas import IncidentStatusUpdate

# Same AWS region every other Bedrock/Comprehend call in this project is
# pinned to (see decision_agent_graph.py's BEDROCK_AWS_REGION) -- reusing the
# env var name rather than inventing a separate SES-only one, since this
# deployment only ever has credentials configured for one region at a time.
SES_AWS_REGION = os.environ.get("BEDROCK_AWS_REGION", "us-east-1")
# Both sender and recipient: this is the operator's own inbox, not a
# distribution list, so one verified SES identity covers both directions.
EMERGENCY_ALERT_EMAIL = os.environ.get("EMERGENCY_ALERT_EMAIL", "yitzhak.maimon1@gmail.com")

# Twilio voice-call counterpart to the SES email above, same destination
# (the operator's own phone, not a real emergency dispatcher). All four
# unset (the default) means "not configured" -- calling is skipped, not
# attempted with empty credentials.
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")
EMERGENCY_ALERT_PHONE_NUMBER = os.environ.get("EMERGENCY_ALERT_PHONE_NUMBER")

distress_pipeline: DistressScreeningPipeline | None = None
info_extractor: RelevantInfoExtractor | None = None
distress_classifier: BedrockDistressClassifier | None = None
rag_retriever: RAGContextRetriever | None = None
decision_graph = None  # compiled LangGraph Decision Agent; None if MCP servers are unreachable
_mcp_exit_stack: AsyncExitStack | None = None
_status_loop_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global distress_pipeline, info_extractor, distress_classifier, rag_retriever
    global decision_graph, _mcp_exit_stack, _status_loop_task

    await init_db()
    _status_loop_task = asyncio.create_task(system_status_loop())

    distress_pipeline = DistressScreeningPipeline()
    info_extractor = RelevantInfoExtractor()
    info_extractor.warm_up()
    distress_classifier = BedrockDistressClassifier()
    rag_retriever = RAGContextRetriever()
    rag_retriever.build_index()

    # Decision Agent prefers the real MCP tools (alert_mcp.py :8001, rag_mcp.py :8002).
    # If they're not reachable, fall back to local mock tools so the model still runs
    # the real reasoning/structured-output pass (tool execution is simulated).
    # If even that fails (e.g. no GOOGLE_API_KEY), fall back fully to the stub
    # decision below -- keeps this file usable for smoke-testing the rest of the
    # n8n pipeline without needing any of that running.
    # A refused TCP connection to a not-yet-running MCP server can surface through
    # the anyio/mcp streamable-http client as asyncio.CancelledError rather than a
    # plain connection error -- and CancelledError is a BaseException (since Python
    # 3.8), not an Exception, so a bare `except Exception` silently misses it and
    # the whole app crashes instead of falling back. Catch both explicitly.
    _mcp_exit_stack = AsyncExitStack()
    await _mcp_exit_stack.__aenter__()
    try:
        decision_graph = await create_decision_agent(_mcp_exit_stack)
    except (Exception, asyncio.CancelledError) as e:
        print(f"[decision-agent] MCP servers unreachable ({e}); falling back to mock tools.")
        try:
            decision_graph = await create_decision_agent(_mcp_exit_stack, use_mock_tools=True)
        except (Exception, asyncio.CancelledError) as e2:
            print(f"[decision-agent] Mock-tools fallback also failed ({e2}); using stub decision instead.")
            decision_graph = None

    yield

    _status_loop_task.cancel()
    await _mcp_exit_stack.aclose()


app = FastAPI(lifespan=lifespan)

# n8n calls this backend server-to-server (not subject to CORS at all), but
# the dashboard's GET /api/v1/incidents fetch() is a real cross-origin
# browser request (dashboard on its own Next.js dev port, backend on :8000)
# -- without this, the browser silently blocks the response before
# JavaScript ever sees it, regardless of anything the endpoint itself
# returns. Wildcard is fine here: local/demo project, no cookies or auth
# headers involved, nothing credentialed to leak cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PLACEHOLDER_FIELDS = {
    "text_content": "בדיקה: אני מרגיש מאוד לבד ומיואש בזמן האחרון",
    "summary": "סיכום מדומה: המשתמש מביע תחושות בדידות ויאוש",
    "class": "medium",
    "risk_level": "medium",
    "context": "הקשר מדומה מתוך בסיס הידע: פנה לקו סיוע 105 או ער\"ן 1201",
    "decision": "route_to_human_review",
    "agent_output": "מומלץ לבצע בדיקה אנושית בהקדם (stub)",
    "passed": True,
    "DetectedText": "טקסט מדומה שזוהה מתוך תמונה (stub)",
    "TranscriptionText": "תמלול מדומה מהודעה קולית (stub)",
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_TRANSCRIBE_MODEL = "whisper-large-v3"


def _transcribe_audio(audio_url: str) -> str:
    """
    Real speech-to-text via Groq's hosted Whisper Large v3. Synchronous and
    fast enough to run inline in this request -- unlike AWS Transcribe's
    upload-to-S3-then-poll-a-job model, which would need real async plumbing
    here. Fails *open* to the stub placeholder text on any error (no
    GROQ_API_KEY configured, no audio_url provided, the URL isn't fetchable,
    or the Groq API call itself fails) so a transcription hiccup degrades to
    the old stub behavior instead of crashing the whole ingestion pipeline.

    Deliberately NOT logged via save_token_usage_record: Whisper is billed by
    audio duration, not tokens, and groq.types.audio.transcription.Transcription
    carries only `text` -- no usage/token field exists to report here, unlike
    _analyze_image's Gemini call just below.
    """
    if not GROQ_API_KEY or not audio_url:
        return PLACEHOLDER_FIELDS["TranscriptionText"]

    try:
        audio_response = requests.get(audio_url, timeout=30)
        audio_response.raise_for_status()
        filename = audio_url.rsplit("/", 1)[-1] or "audio"
        client = Groq(api_key=GROQ_API_KEY)
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_response.content),
            model=GROQ_TRANSCRIBE_MODEL,
        )
        return transcription.text
    except Exception as e:
        print(f"[Groq Transcription Error] {e}")
        return PLACEHOLDER_FIELDS["TranscriptionText"]


GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")

VISION_PROMPT = """את/ה כלי סינון במערכת טריאז' מצוקה (SafeSignal). קיבלת תמונה שהועלתה \
או צולמה על ידי משתמש. שתי מטרות, בסדר הזה:
1. אם יש בתמונה טקסט כתוב (צילום מסך של הודעה, פתק, שיחה וכו') - תעתיק אותו כלשונו, \
במדויק ובשלמותו.
2. בנוסף (או אם אין טקסט), תאר/י בקצרה כל רמז חזותי במצוקה עצמה - סימני פגיעה עצמית, \
מצב מסוכן, סביבה מדאיגה וכו'. אם אין כזה, אין צורך לציין זאת במפורש.

החזר/י תשובה אחת, טקסטואלית וקצרה ככל האפשר, המשלבת את שני הדברים (אם רלוונטיים). \
אל תמציא/י תוכן שאינו נראה בתמונה בפועל."""


def _analyze_image(image_url: str, incident_id: str = "") -> str:
    """
    Real vision analysis via Gemini (multimodal): OCRs any text visible in the
    image and separately flags visual distress signals the text alone wouldn't
    capture (e.g. a photo of a dangerous situation with no caption). One model
    call covers both "Textract" and "Rekognition" roles from the node's name.
    Fails *open* to the stub placeholder text on any error, same convention as
    _transcribe_audio -- a vision hiccup degrades gracefully instead of crashing
    the ingestion pipeline.

    This is n8n's very first model call (Image Analysis, right after Route by
    Detected Content) -- logged under save_token_usage_record's
    "vision_gemini_analysis" stage, matched to the incident by incident_id
    rather than by sentence, since the sentence this stage produces
    (the OCR'd/described text) isn't necessarily the same string later stages
    log token usage against once it's merged with other sources.
    """
    if not GOOGLE_API_KEY or not image_url:
        return PLACEHOLDER_FIELDS["DetectedText"]

    try:
        image_response = requests.get(image_url, timeout=30)
        image_response.raise_for_status()
        mime_type = image_response.headers.get("Content-Type", "image/jpeg").split(";")[0]
        b64_image = base64.b64encode(image_response.content).decode("utf-8")

        llm = ChatGoogleGenerativeAI(model=GEMINI_VISION_MODEL, google_api_key=GOOGLE_API_KEY)
        message = HumanMessage(content=[
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
        ])
        response = llm.invoke([message])

        usage = getattr(response, "usage_metadata", None)
        if usage:
            save_token_usage_record(
                pipeline_stage="vision_gemini_analysis",
                model_id=GEMINI_VISION_MODEL,
                sentence=response.content,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                incident_id=incident_id,
            )
        return response.content
    except Exception as e:
        print(f"[Gemini Vision Error] {e}")
        return PLACEHOLDER_FIELDS["DetectedText"]


def merged(body: dict, **overrides) -> dict:
    return {**PLACEHOLDER_FIELDS, **body, **overrides}


def _detect_pii(text: str) -> tuple[list[dict], bool]:
    """
    Real Amazon Comprehend PII detection (DetectPiiEntities) on the Decision
    Agent's own generated text -- distinct purpose from the distress classifier:
    this checks whether the *agent's output* leaks sensitive personal data
    (names, addresses, phone numbers, etc.), not whether the text describes
    distress. Comprehend's PII API only supports English; for any other
    language this returns (empty list, checked=False) so callers can tell
    "confirmed clean" apart from "not checked".
    """
    assert distress_pipeline is not None
    language = distress_pipeline.detect_language(text)
    if language != "en":
        return [], False

    try:
        res = distress_pipeline.comprehend.detect_pii_entities(Text=text, LanguageCode="en")
        entities = [
            {"type": e["Type"], "score": round(e["Score"], 4)} for e in res.get("Entities", [])
        ]
        return entities, True
    except Exception as e:
        print(f"[PII Detection Error] {e}")
        return [], False


def _ml_output_classifier(text: str) -> tuple[bool, str]:
    """
    Placeholder for the trained ML classifier that should verify the Decision
    Agent's output isn't hallucinated and complies with policy/guidelines.
    No such model has been trained yet -- always passes, and says so
    explicitly rather than silently pretending to have checked. Train and
    swap this in the same way hebert_distress_model was built (see
    training/train_hebert_colab.ipynb) once a labeled dataset of
    acceptable-vs-problematic agent output exists.
    """
    return True, "not_implemented: no trained output-quality classifier yet -- always passes"


@app.post("/api/v1/vision/textract-rekognition")
async def vision(req: Request):
    body = await req.json()
    detected_text = _analyze_image(body.get("image_url", ""), incident_id=body.get("incident_id", ""))
    return merged(body, DetectedText=detected_text)


@app.post("/api/v1/audio/transcribe")
async def audio(req: Request):
    body = await req.json()
    transcription_text = _transcribe_audio(body.get("audio_url", ""))
    return merged(body, TranscriptionText=transcription_text)


@app.post("/api/v1/ingest/raw-log")
async def ingest_raw_log(req: Request):
    """
    n8n's "Upload a file" step calls this right after Unified Ingestion,
    before screening -- replaces the previous one-JSON-object-per-message S3
    upload with a single cumulative Excel workbook (raw/raw_messages_log.xlsx),
    same pattern as the incidents table.
    """
    body = await req.json()
    logged = log_raw_message(
        user_id=body.get("user_id", "anonymous"),
        incident_id=body.get("incident_id", ""),
        text_content=body.get("text_content", ""),
        message_timestamp=body.get("message_timestamp"),
    )
    return merged(body, logged=logged)


@app.post("/api/v1/screening/input")
async def screening_input(req: Request):
    body = await req.json()
    real_text = body.get("text") or body.get("text_content") or PLACEHOLDER_FIELDS["text_content"]
    assert distress_pipeline is not None
    result = distress_pipeline.analyze_post(real_text, incident_id=body.get("incident_id", ""))
    return merged(
        body,
        passed=result["passed_screening"],
        text_content=real_text,
        screening_reason=result["reason"],
        screening_metrics=result.get("raw_metrics", {}),
    )


@app.post("/api/v1/extract-info")
async def extract_info(req: Request):
    body = await req.json()
    text = body.get("text_content") or body.get("text") or PLACEHOLDER_FIELDS["text_content"]
    assert info_extractor is not None
    extracted = info_extractor.extract(text, incident_id=body.get("incident_id", ""))
    return merged(
        body,
        text_content=text,
        names=extracted["names"],
        addresses=extracted["addresses"],
        ages=extracted["ages"],
        phone_numbers=extracted["phone_numbers"],
    )


@app.post("/api/v1/classify-distress")
async def classify_distress(req: Request):
    body = await req.json()
    text = body.get("text_content", "")
    assert distress_classifier is not None
    result = distress_classifier.classify(text, incident_id=body.get("incident_id", ""))
    return merged(
        body,
        **{"class": result["class"]},
        risk_level=result["risk_level"],
        summary=result["summary"],
    )


@app.post("/api/v1/rag-context")
async def rag_context(req: Request):
    """
    צומת ה-Context Retrieval (RAG - LangChain + Vector DB): מקבל את הטקסט + את
    הסיווג שכבר נקבע ע"י Bedrock (class/risk_level/summary), מאתר הקשר תומך-החלטה
    מבסיס הידע (rag_retrieval.py) ומחזיר ל-Decision Agent JSON מפורש ונקי -
    לא רק "merged" גנרי - כך שהוא תמיד מכיל גם את מה ש-Bedrock קבע וגם את מה
    שה-RAG אחזר, בלי תלות בזה שכל hop קודם באמת העביר הלאה את כל השדות.
    """
    body = await req.json()
    text = body.get("text_content") or body.get("text") or PLACEHOLDER_FIELDS["text_content"]

    assert rag_retriever is not None
    context, retrieved_examples = rag_retriever.build_context(text)

    return {
        "text_content": text,
        # מזהים שחייבים לשרוד עד ל-Decision Agent (הוא מייצר incident_id אקראי
        # חדש אם זה לא מגיע אליו) - נשמטו כאן בעבר כי הפונקציה בונה dict מפורש
        # ולא merged() גנרי, בדיוק המקרה שההערה למעלה מזהירה מפניו.
        "incident_id": body.get("incident_id", ""),
        "user_id": body.get("user_id", "anonymous"),
        "platform": body.get("platform", "telegram"),
        # שדות בבעלות Bedrock (distress_classification.py) - מועברים הלאה במפורש.
        "class": body.get("class", PLACEHOLDER_FIELDS["class"]),
        "risk_level": body.get("risk_level", PLACEHOLDER_FIELDS["risk_level"]),
        "summary": body.get("summary", PLACEHOLDER_FIELDS["summary"]),
        # שדות בבעלות ה-RAG (rag_retrieval.py).
        "context": context,
        "retrieved_examples": retrieved_examples,
        # שדות תומכים מ-Relevant Information Extraction, ממשיכים איתם הלאה כדי שלא
        # יאבדו לפני Decision Agent / Human Review.
        "names": body.get("names", []),
        "addresses": body.get("addresses", []),
        "ages": body.get("ages", []),
        "phone_numbers": body.get("phone_numbers", []),
    }


async def _persist_and_broadcast_incident(
    *,
    incident_id: str,
    user_id: str,
    platform: str,
    raw_text: str,
    risk_level: str,
    summary: str,
    thought_process: str,
    recommended_action: str,
    final_urgency_assessment: str,
    distress_classification: str,
    screening_passed: bool,
    screening_reason: str,
    screening_tags: list,
    tools_triggered: list,
    screening_logs: list,
    names: list,
    addresses: list,
    ages: list,
    phone_numbers: list,
    tokens_used: int | None = None,
    confidence_score: float | None = None,
) -> None:
    """
    Upserts the now-complete incident (every decision_agent_graph.py terminal
    node -- immediate_alert/human_review/log_and_close -- already ran inside
    the ainvoke() call this is invoked after, so this is the one place a full
    incident record exists) into the DB and broadcasts INCIDENT_NEW to every
    connected dashboard operator. Upsert rather than plain insert because n8n
    can retry the same incident_id; is_unread/status are deliberately left out
    of the update set so a retry can't clobber an operator's in-progress
    review back to "new"/unread.
    """
    content_fields = dict(
        user_id=user_id,
        platform=platform,
        raw_text=raw_text,
        risk_level=risk_level,
        summary=summary,
        thought_process=thought_process,
        recommended_action=recommended_action,
        final_urgency_assessment=final_urgency_assessment,
        distress_classification=distress_classification,
        screening_passed=screening_passed,
        screening_reason=screening_reason,
        screening_tags=screening_tags,
        tools_triggered=tools_triggered,
        screening_logs=screening_logs,
        tokens_used=tokens_used,
        confidence_score=confidence_score,
        updated_at=datetime.now(timezone.utc),
    )
    entity_fields = dict(names=names, ages=ages, phone_numbers=phone_numbers, addresses=addresses)

    async with get_session() as session:
        await session.execute(
            upsert_stmt(
                Incident,
                index_elements=["incident_id"],
                values={"incident_id": incident_id, **content_fields},
            )
        )
        await session.execute(
            upsert_stmt(
                ExtractedEntities,
                index_elements=["incident_id"],
                values={"incident_id": incident_id, **entity_fields},
            )
        )
        await session.commit()

        incident_row = await session.get(Incident, incident_id)
        entities_row = await session.get(ExtractedEntities, incident_id)

    record_invocation()
    await manager.broadcast(incident_new_payload(incident_row, entities_row))


@app.post("/api/v1/decision-agent")
async def decision_agent(req: Request):
    body = await req.json()
    text = body.get("text_content") or body.get("text") or PLACEHOLDER_FIELDS["text_content"]
    incident_id = body.get("incident_id") or str(uuid.uuid4())
    user_id = body.get("user_id", "anonymous")
    platform = body.get("platform", "telegram")
    names = body.get("names", [])
    addresses = body.get("addresses", [])
    ages = body.get("ages", [])
    phone_numbers = body.get("phone_numbers", [])

    if decision_graph is None:
        classification = body.get("classification", body.get("class", "רגיל"))
        recommended_action = f"route_to_human_review ({classification})"
        await _persist_and_broadcast_incident(
            incident_id=incident_id,
            user_id=user_id,
            platform=platform,
            raw_text=text,
            risk_level=body.get("risk_level", "unknown"),
            summary=PLACEHOLDER_FIELDS["summary"],
            thought_process="",
            recommended_action=recommended_action,
            final_urgency_assessment="",
            distress_classification=classification,
            screening_passed=True,
            screening_reason="",
            screening_tags=[],
            tools_triggered=[],
            screening_logs=[],
            names=names,
            addresses=addresses,
            ages=ages,
            phone_numbers=phone_numbers,
            tokens_used=get_total_tokens_for_sentence(text, incident_id=incident_id),
            confidence_score=None,
        )
        return merged(body, decision=recommended_action)

    initial_state = {
        "raw_input": text,
        "user_id": user_id,
        "incident_id": incident_id,
        "distress_classification": body.get("class", body.get("risk_level", "unknown")),
        "initial_rag_context": body.get("context", ""),
        "names": names,
        "addresses": addresses,
        "ages": ages,
        "phone_numbers": phone_numbers,
    }
    result = await decision_graph.ainvoke(initial_state)

    await _persist_and_broadcast_incident(
        incident_id=incident_id,
        user_id=user_id,
        platform=platform,
        raw_text=text,
        risk_level=result.get("risk_level") or "unknown",
        summary=result.get("summary_for_human_reviewer", ""),
        thought_process=result.get("thought_process", ""),
        recommended_action=result.get("recommended_action", ""),
        final_urgency_assessment=result.get("final_urgency_assessment", ""),
        distress_classification=initial_state["distress_classification"],
        screening_passed=result.get("screening_passed", True),
        screening_reason=result.get("screening_reason", ""),
        screening_tags=result.get("screening_tags", []),
        tools_triggered=result.get("tools_triggered", []),
        screening_logs=result.get("screening_logs", []),
        names=names,
        addresses=addresses,
        ages=ages,
        phone_numbers=phone_numbers,
        tokens_used=get_total_tokens_for_sentence(text, incident_id=incident_id),
        confidence_score=result.get("confidence_score"),
    )

    return merged(
        body,
        thought_process=result.get("thought_process", ""),
        tools_triggered=result.get("tools_triggered", []),
        final_urgency_assessment=result.get("final_urgency_assessment", ""),
        recommended_action=result.get("recommended_action", ""),
        summary_for_human_reviewer=result.get("summary_for_human_reviewer", ""),
        confidence_score=result.get("confidence_score"),
        # Set by output_screening_node / routing_by_risk_level_node, which now run
        # as part of the same graph (Bedrock Guardrails + Claude Haiku hallucination
        # check, then risk-level routing to immediate_alert/human_review/log_and_close).
        passed=result.get("screening_passed", True),
        risk_level=result.get("risk_level", ""),
        screening_tags=result.get("screening_tags", []),
        screening_reason=result.get("screening_reason", ""),
        screening_logs=result.get("screening_logs", []),
        output_retry_count=result.get("output_retry_count", 0),
        # Kept for any n8n nodes still wired to the older stub field names.
        decision=result.get("recommended_action", ""),
        agent_output=result.get("summary_for_human_reviewer", ""),
    )


@app.post("/api/v1/screening/output")
async def screening_output(req: Request):
    """
    Legacy standalone output-screening endpoint (Comprehend PII + placeholder
    ML classifier). No longer called by decision_agent_graph.py -- that graph's
    output_screening_node now does its own checks in-process via boto3 Bedrock
    Guardrails + Claude Haiku (see decision_agent_graph.py's
    _run_bedrock_guardrails / _run_hallucination_check), so it can retry the
    Decision Agent directly on failure instead of round-tripping through HTTP.
    Left here as a standalone testable endpoint for the original Comprehend-
    based approach. Two checks, matching the n8n node's two tools:
      1. PII / sensitive content -- real Amazon Comprehend detect_pii_entities.
      2. Hallucination / policy compliance -- placeholder ML classifier; see
         _ml_output_classifier (no trained model exists yet).
    """
    body = await req.json()
    text = body.get("text_content") or body.get("text") or ""

    pii_entities, pii_checked = _detect_pii(text)
    classifier_passed, classifier_reason = _ml_output_classifier(text)

    passed = not pii_entities and classifier_passed
    tags = [f"pii:{e['type']}" for e in pii_entities]

    # Lowercase high/medium/low -- same vocabulary routing_by_risk_level_node
    # and DecisionOutput.final_urgency_assessment use, so a consumer reading
    # `risk_level` back doesn't see different casing depending on which
    # endpoint produced it. PII detection is treated as the most severe case.
    if pii_entities:
        risk_level = "high"
    elif not classifier_passed:
        risk_level = "medium"
    else:
        risk_level = "low"

    return merged(
        body,
        passed=passed,
        risk_level=risk_level,
        tags=tags,
        pii_entities=pii_entities,
        pii_checked=pii_checked,
        screening_reason=classifier_reason,
    )


@app.post("/api/v1/review-queue")
async def review_queue(req: Request):
    """
    יעד ה-Human Review בפועל: אין מופע Open WebUI אמיתי לדחוף אליו. Called by
    decision_agent_graph.py's human_review_node *during* the same ainvoke()
    that /api/v1/decision-agent's own persistence call runs after -- so this
    endpoint no longer writes its own record (that would race the not-yet-
    existing DB row); /api/v1/decision-agent's _persist_and_broadcast_incident
    is the single write site for every risk tier.
    """
    body = await req.json()
    return {"status": "queued", **body}


@app.post("/api/v1/immediate-alert")
async def immediate_alert(req: Request):
    body = await req.json()
    return {"status": "alert_sent", **body}


@app.post("/api/v1/log-error")
async def log_error(req: Request):
    """
    יעד גנרי לדיווח שגיאות - למשל מ-n8n Error Workflow שרץ כשכשל execution
    (כגון Invoke Decision Agent (Backend) כשה-backend לא זמין). נשמר ב-
    data/error_log.xlsx כדי שמסך השגיאות העתידי ב-UI (GET /api/v1/errors)
    יוכל להציג אותו למשתמש, במקום שהכשל ייבלע בשקט. לא נשלח מייל על כך --
    רק הרישום לקובץ.
    """
    body = await req.json()
    source = body.get("source", "unknown")
    error_message = body.get("error_message", "")
    incident_id = body.get("incident_id", "")
    user_id = body.get("user_id", "")
    save_error_record(
        source=source,
        error_message=error_message,
        incident_id=incident_id,
        user_id=user_id,
    )
    return {"status": "logged"}


@app.get("/api/v1/errors")
async def list_errors():
    """מסך השגיאות העתידי ב-UI קורא מכאן - רשימת כל השגיאות שנרשמו ב-data/error_log.xlsx."""
    return {"errors": get_error_records()}


@app.get("/api/v1/incidents")
async def list_incidents(limit: int = 100):
    """
    History-fetch endpoint for the dashboard's initial load / page refresh --
    the WebSocket (/api/v1/realtime) only ever pushes *new*/*updated*
    incidents, it has no memory of anything that happened before a given
    client connected. Without this, refreshing the page had nothing to
    hydrate the feed from and always started blank.

    incident_id + entities are always written together in the same
    transaction (_persist_and_broadcast_incident upserts both), so an inner
    join is safe -- there's no code path that creates one without the other.
    """
    limit = max(1, min(limit, 500))
    async with get_session() as session:
        result = await session.execute(
            select(Incident, ExtractedEntities)
            .join(ExtractedEntities, ExtractedEntities.incident_id == Incident.incident_id)
            .order_by(Incident.created_at.desc())
            .limit(limit)
        )
        rows = result.all()

    return [serialize_incident(incident, entities) for incident, entities in rows]


# Sorting "category" by the raw distress_classification column scatters rows
# instead of grouping them: every raw label the classifier can produce that
# ISN'T one of the three mapped ones (CLASS_TO_CATEGORY_CODE) -- "רגיל",
# "unknown", etc -- sorts by its own distinct string instead of collapsing
# into the single blank/no-badge group the CategoryBadge actually renders.
# This CASE expression sorts by the same normalized code the frontend
# displays, so every row with no badge shares one sort key and clusters
# together instead of being interleaved with real categories.
_CATEGORY_SORT_COLUMN = case(
    *[(Incident.distress_classification == raw, code) for raw, code in CLASS_TO_CATEGORY_CODE.items()],
    else_="",
)

# Column a `sortBy` value from the History screen's clickable headers maps to
# -- kept as an explicit whitelist (rather than getattr(Incident, sortBy))
# so an unrecognized/malicious sortBy value can never reach raw SQL, and
# falls back to Incident.created_at below instead of erroring.
_HISTORY_SORT_COLUMNS = {
    "logId": literal_column("incidents.rowid"),
    "timestamp": Incident.created_at,
    "user": Incident.user_id,
    "category": _CATEGORY_SORT_COLUMN,
    "text": Incident.raw_text,
    "status": Incident.status,
    "tokens": Incident.tokens_used,
    "score": Incident.confidence_score,
}

# English labels the History table's Category/Resolution badges actually
# display (see safesignal-dashboard's CategoryBadge/ResolutionBadge) --
# distress_classification is stored in Hebrew and status is an internal
# code ("new", "false_positive", ...), so a plain ILIKE against those raw
# columns would never match what a search for e.g. "suicide" or "escalated"
# is really looking for. Global Search below OR's these in alongside the
# raw-column ILIKE match.
_CATEGORY_CODE_LABELS = {
    "emotional_distress": "emotional distress",
    "cyberbullying": "cyberbullying",
    "suicide_emergency": "suicide emergency",
}
_RESOLUTION_LABELS = {
    "new": "open",
    "investigating": "investigating",
    "escalated": "escalated",
    "false_positive": "false positive",
    "closed": "closed",
}


def _build_history_filters(search: str, startDate: str, endDate: str) -> list:
    """
    Shared WHERE-clause builder for both the paginated History table
    (list_incidents_history) and its aggregate-stats counterpart
    (get_incidents_history_stats) -- factored out so the two endpoints can
    never drift apart on what "the same search/date filters" actually means.
    """
    filters = []
    search = search.strip()
    if search:
        like = f"%{search}%"
        search_lower = search.lower()
        conditions = [
            Incident.user_id.ilike(like),
            Incident.raw_text.ilike(like),
            Incident.distress_classification.ilike(like),
            Incident.status.ilike(like),
            Incident.incident_id.ilike(like),
        ]

        matching_hebrew_labels = [
            hebrew_label
            for hebrew_label, category_code in CLASS_TO_CATEGORY_CODE.items()
            if search_lower in _CATEGORY_CODE_LABELS.get(category_code, "")
        ]
        if matching_hebrew_labels:
            conditions.append(Incident.distress_classification.in_(matching_hebrew_labels))

        matching_statuses = [
            status for status, label in _RESOLUTION_LABELS.items() if search_lower in label
        ]
        if matching_statuses:
            conditions.append(Incident.status.in_(matching_statuses))

        filters.append(or_(*conditions))
    if startDate:
        start_dt = datetime.strptime(startDate, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        filters.append(Incident.created_at >= start_dt)
    if endDate:
        # Inclusive of the whole end day -- a date-only picker value with no
        # time component would otherwise exclude every row from that day.
        end_dt = datetime.strptime(endDate, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
        filters.append(Incident.created_at < end_dt)
    return filters


@app.get("/api/v1/incidents/history")
async def list_incidents_history(
    page: int = 1,
    limit: int = 50,
    search: str = "",
    sortBy: str = "timestamp",
    sortOrder: str = "desc",
    startDate: str = "",
    endDate: str = "",
):
    """
    Server-side paginated/sorted/filtered feed for the Alert History screen --
    unlike list_incidents() above (a fixed most-recent-N hydration fetch for
    the live dashboard), every page/sort/filter here is computed in SQL via
    LIMIT/OFFSET and WHERE, not fetched-then-sliced in Python: the frontend
    never receives more than one page's worth of rows for any given request,
    which is what keeps this correct as the incidents table grows past what
    a single page load could reasonably hold.

    This endpoint is deliberately NOT what feeds the charts above the History
    table -- see get_incidents_history_stats for that. Fetching every row
    through this one with a huge `limit` would work today but silently
    undercounts the moment the table grows past this endpoint's own 200 cap.

    log_id is SQLite's implicit rowid (see _HISTORY_SORT_COLUMNS) -- there is
    no dedicated autoincrement column on Incident (its primary key is a
    string incident_id), and rowid already gives every ordinary SQLite table
    a stable, monotonically-increasing integer for free.
    """
    page = max(1, page)
    limit = max(1, min(limit, 200))
    sort_column = _HISTORY_SORT_COLUMNS.get(sortBy, Incident.created_at)
    order = asc if sortOrder.lower() == "asc" else desc

    filters = _build_history_filters(search, startDate, endDate)

    async with get_session() as session:
        total = (
            await session.execute(select(func.count()).select_from(Incident).where(*filters))
        ).scalar_one()

        result = await session.execute(
            select(Incident, literal_column("incidents.rowid").label("log_id"))
            .where(*filters)
            .order_by(order(sort_column))
            .limit(limit)
            .offset((page - 1) * limit)
        )
        rows = result.all()

    return {
        "rows": [serialize_history_row(incident, log_id) for incident, log_id in rows],
        "total": total,
        "page": page,
        "limit": limit,
        "totalPages": max(1, math.ceil(total / limit)),
    }


@app.get("/api/v1/incidents/history/stats")
async def get_incidents_history_stats(
    search: str = "",
    startDate: str = "",
    endDate: str = "",
):
    """
    Aggregate counts for the charts above the Alert History table (category
    mix, severity mix, monthly trend, one month's call-density heatmap,
    handled/total) -- every number here is a COUNT/GROUP BY over every row
    matching the filters, computed in SQLite, not fetched-then-counted in
    the browser. That's the difference from list_incidents_history: this
    endpoint has no page/limit/sort of its own and stays correct regardless
    of how large the incidents table grows, whereas paging through that one
    with an oversized `limit` would silently stop counting past its 200 cap.

    Shares search/date filters with the History table (_build_history_filters)
    so narrowing those narrows the charts too -- but has nothing analogous to
    the table's `page`, since narrowing *that* is pagination, not scope.
    """
    filters = _build_history_filters(search, startDate, endDate)

    async with get_session() as session:
        total = (
            await session.execute(select(func.count()).select_from(Incident).where(*filters))
        ).scalar_one()

        # "Handled" = any status other than "new" -- clicking Acknowledge
        # (-> "investigating") is itself the operator's closing action on an
        # incident from the History screen's point of view, not a
        # still-in-progress state (product decision: Acknowledge marks the
        # case as handled). "Active Open Cases" on the main screen answers a
        # different question -- "how many cards are still in my personal
        # feed" -- and deliberately also excludes locally-hidden
        # acknowledged incidents, which this DB-level count has no way to
        # see; the two panels are allowed to disagree by design.
        handled_total = (
            await session.execute(
                select(func.count())
                .select_from(Incident)
                .where(*filters, Incident.status != "new")
            )
        ).scalar_one()

        category_rows = (
            await session.execute(
                select(Incident.distress_classification, func.count())
                .where(*filters)
                .group_by(Incident.distress_classification)
            )
        ).all()

        severity_rows = (
            await session.execute(
                select(Incident.risk_level, func.count())
                .where(*filters)
                .group_by(Incident.risk_level)
            )
        ).all()

        month_key = func.strftime("%Y-%m", Incident.created_at)
        monthly_category_rows = (
            await session.execute(
                select(month_key.label("month_key"), Incident.distress_classification, func.count())
                .where(*filters)
                .group_by("month_key", Incident.distress_classification)
            )
        ).all()

        monthly_total_rows = (
            await session.execute(
                select(month_key.label("month_key"), func.count())
                .where(*filters)
                .group_by("month_key")
            )
        ).all()

    # "Other" absorbs both the classifier's non-distress label ("רגיל") and
    # any raw label CLASS_TO_CATEGORY_CODE doesn't recognize -- same rule
    # serialize_history_row's category_code_for uses for the table.
    category_counts = {"suicide_emergency": 0, "cyberbullying": 0, "emotional_distress": 0, "other": 0}
    for raw_label, count in category_rows:
        category_counts[CLASS_TO_CATEGORY_CODE.get(raw_label, "other")] += count

    # Only high/medium/low ever reach the Severity Level chart -- a row
    # whose risk_level is still "unknown" (or the reserved "critical" tier,
    # see the frontend's RiskLevel comment) hasn't been triaged to a severity
    # yet and is excluded rather than guessed at.
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for level, count in severity_rows:
        if level in severity_counts:
            severity_counts[level] += count

    monthly_by_key: dict[str, dict[str, int]] = {}
    for month_key_value, raw_label, count in monthly_category_rows:
        code = CLASS_TO_CATEGORY_CODE.get(raw_label)
        if code is None:
            continue
        bucket = monthly_by_key.setdefault(
            month_key_value, {"emotional_distress": 0, "cyberbullying": 0, "suicide_emergency": 0}
        )
        bucket[code] += count
    monthly_trends = [
        {"month": datetime.strptime(key, "%Y-%m").strftime("%b"), **counts}
        for key, counts in sorted(monthly_by_key.items())
    ]

    # Calendar heatmap covers a continuous 6-month strip (GitHub-contributions
    # style) ending on whichever month had the most matching rows -- mirrors
    # the frontend's old client-side heuristic (usually the most recent
    # active month, since the table defaults to timestamp desc) while still
    # surfacing the five months before it instead of just one.
    month_totals = {key: count for key, count in monthly_total_rows}
    top_month_key = max(month_totals, key=month_totals.get) if month_totals else None

    if top_month_key:
        end_year, end_month = (int(part) for part in top_month_key.split("-"))
    else:
        now = datetime.now(timezone.utc)
        end_year, end_month = now.year, now.month

    # 5 months before end_month, wrapping year boundaries.
    start_ordinal = end_year * 12 + (end_month - 1) - 5
    start_year, start_month = divmod(start_ordinal, 12)
    start_month += 1

    range_start = datetime(start_year, start_month, 1)
    range_end = datetime(end_year, end_month, monthrange(end_year, end_month)[1])

    async with get_session() as session:
        day_key = func.strftime("%Y-%m-%d", Incident.created_at)
        day_rows = (
            await session.execute(
                select(day_key.label("day"), func.count())
                .where(
                    *filters,
                    day_key >= range_start.strftime("%Y-%m-%d"),
                    day_key <= range_end.strftime("%Y-%m-%d"),
                )
                .group_by("day")
            )
        ).all()
    day_counts = {day: count for day, count in day_rows}

    heatmap_cells = []
    cursor = range_start
    while cursor <= range_end:
        iso_date = cursor.strftime("%Y-%m-%d")
        heatmap_cells.append(
            {
                "date": iso_date,
                "day": cursor.day,
                "month": cursor.month,
                "year": cursor.year,
                # Python's Monday=0 vs JS Date.getDay()'s Sunday=0 -- the
                # frontend calendar grid groups by the JS convention, so
                # convert here once rather than re-deriving it client-side.
                "weekday": (cursor.weekday() + 1) % 7,
                "count": day_counts.get(iso_date, 0),
            }
        )
        cursor += timedelta(days=1)

    return {
        "total": total,
        "handledTotal": handled_total,
        "categoryCounts": category_counts,
        "severityCounts": severity_counts,
        "monthlyTrends": monthly_trends,
        "heatmapRangeLabel": f"{range_start.strftime('%b %Y')} – {range_end.strftime('%b %Y')}",
        "heatmapCells": heatmap_cells,
    }


@app.websocket("/api/v1/realtime")
async def realtime(websocket: WebSocket):
    """
    Persistent WebSocket the dashboard connects to for live incident/system
    frames (INCIDENT_NEW, INCIDENT_UPDATE, SYSTEM_STATUS_CHANGE). Supports any
    number of concurrently connected operators -- every broadcast fans out to
    all of them via realtime.py's ConnectionManager. The dashboard doesn't
    send anything on this socket; we just block on receive_text() so we
    notice a disconnect (raises WebSocketDisconnect) instead of busy-polling.
    """
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


def _send_escalation_email(incident: Incident, entities: ExtractedEntities | None) -> None:
    """
    Fires when an operator presses "Escalate to Authorities" in the dashboard
    (PATCH .../status with status="escalated"). This is the manual-escalation
    counterpart to decision_agent_graph.py's automatic immediate_alert_node --
    that path currently posts to /api/v1/immediate-alert, which is still a
    stub (see that endpoint), so today NO email goes out for either the
    automatic high-risk path or a manual click; this wires up the manual one.

    n8n.json already has a fully-configured "Send Emergency Email (AWS SES)"
    node (id awsses-emergency-email, from/to yitzhak.maimon1@gmail.com) after
    its "Immediate Alert" node, but that workflow is no longer in the live
    execution path -- the dashboard talks to this backend directly, not to
    n8n -- so that node never fires. Sending directly from here with the same
    SES identity reuses the one thing from that dead path that was already
    correct, instead of resurrecting a webhook hop through n8n.

    Best-effort: a failed send must not block the status update/broadcast,
    so failures are only logged loudly (same fail-open convention as
    decision_agent_graph.py's immediate_alert_node).
    """
    names = ", ".join(entities.names) if entities and entities.names else "(אין שם)"
    addresses = ", ".join(entities.addresses) if entities and entities.addresses else "(אין כתובת)"
    ages = ", ".join(str(a) for a in entities.ages) if entities and entities.ages else "(אין גיל)"
    phones = ", ".join(entities.phone_numbers) if entities and entities.phone_numbers else "(אין טלפון)"

    rows = "".join(
        f'<tr><td style="padding:6px 10px;border:1px solid #ccc;font-weight:bold;'
        f'background:#f5f5f5;white-space:nowrap;">{label}</td>'
        f'<td style="padding:6px 10px;border:1px solid #ccc;">{value}</td></tr>'
        for label, value in [
            ("Incident ID", incident.incident_id),
            ("User ID", incident.user_id),
            ("פלטפורמה", incident.platform),
            ("רמת סיכון", incident.risk_level),
            ("שם/שמות שזוהו", names),
            ("כתובות שזוהו", addresses),
            ("גיל", ages),
            ("טלפון", phones),
        ]
    )
    html_body = (
        '<div dir="rtl" style="font-family:Arial,sans-serif;max-width:640px;margin:0 auto;">'
        '<div style="background:#c0392b;color:#fff;padding:16px;text-align:center;'
        'font-size:22px;font-weight:bold;border-radius:6px 6px 0 0;">'
        "🚨 אירוע הועבר ידנית לטיפול הרשויות 🚨</div>"
        '<div style="border:3px solid #c0392b;border-top:none;padding:16px;'
        'border-radius:0 0 6px 6px;">'
        '<p style="color:#c0392b;font-weight:bold;font-size:16px;">מפעיל המערכת סימן '
        'אירוע זה כ"Escalate to Authorities":</p>'
        f'<table style="width:100%;border-collapse:collapse;margin-bottom:16px;">{rows}</table>'
        '<p style="font-weight:bold;">ציטוט ההודעה המקורית:</p>'
        f'<blockquote style="background:#fdecea;border-right:4px solid #c0392b;'
        f'margin:0;padding:10px 14px;font-size:15px;">{incident.raw_text}</blockquote>'
        '<p style="font-weight:bold;margin-top:16px;">סיכום:</p>'
        f"<p>{incident.summary}</p></div></div>"
    )

    try:
        client = boto3.client("ses", region_name=SES_AWS_REGION)
        client.send_email(
            Source=EMERGENCY_ALERT_EMAIL,
            Destination={"ToAddresses": [EMERGENCY_ALERT_EMAIL]},
            Message={
                "Subject": {
                    "Data": f"🚨 SafeSignal - Incident {incident.incident_id} escalated to authorities",
                    "Charset": "UTF-8",
                },
                "Body": {"Html": {"Data": html_body, "Charset": "UTF-8"}},
            },
        )
    except (BotoCoreError, ClientError) as e:
        print(f"[Escalate Email Error] could not send SES email for incident '{incident.incident_id}': {e}")


def _place_escalation_call(incident: Incident, entities: ExtractedEntities | None) -> None:
    """
    Twilio counterpart to _send_escalation_email -- same trigger (operator
    clicks "Escalate to Authorities") and same destination (the operator's
    own phone, EMERGENCY_ALERT_PHONE_NUMBER -- not a real emergency
    dispatcher; the operator is who decides whether to actually contact
    authorities, exactly like the email).

    Uses Twilio's inline `twiml=` param (a literal TwiML document handed
    straight to the API) instead of pointing Twilio at a webhook URL that
    returns TwiML -- this backend runs locally with no public inbound
    address for Twilio to fetch from.

    Best-effort / fail-open, same convention as _send_escalation_email: skips
    entirely (logged, not silent) if Twilio isn't configured, and a failed
    call must not block the status update/broadcast.
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER and EMERGENCY_ALERT_PHONE_NUMBER):
        print(
            "[Escalate Call] Twilio not configured (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / "
            "TWILIO_FROM_NUMBER / EMERGENCY_ALERT_PHONE_NUMBER) -- skipping call."
        )
        return

    names = ", ".join(entities.names) if entities and entities.names else "לא זוהה שם"
    addresses = ", ".join(entities.addresses) if entities and entities.addresses else "לא זוהתה כתובת"

    # Read twice -- a phone call has no screen to re-read from, so the one
    # chance to catch the incident ID/address is a second pass.
    spoken_message = (
        f"התראת חירום ממערכת סייף סיגנל. מספר תיק {incident.incident_id}. "
        f"רמת סיכון: {incident.risk_level}. שם שזוהה: {names}. "
        f"כתובת שזוהתה: {addresses}. סיכום: {incident.summary}. "
        f"חוזר על הפרטים. מספר תיק {incident.incident_id}. "
        f"רמת סיכון: {incident.risk_level}. שם שזוהה: {names}. "
        f"כתובת שזוהתה: {addresses}."
    )
    twiml = f'<Response><Say language="he-IL">{xml_escape(spoken_message)}</Say></Response>'

    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.calls.create(to=EMERGENCY_ALERT_PHONE_NUMBER, from_=TWILIO_FROM_NUMBER, twiml=twiml)
    except TwilioRestException as e:
        print(f"[Escalate Call Error] could not place Twilio call for incident '{incident.incident_id}': {e}")


@app.patch("/api/v1/incidents/{incident_id}/status")
async def update_incident_status(incident_id: str, update: IncidentStatusUpdate):
    """
    Dashboard action-bar endpoint (Acknowledge/Escalate/False Positive/Close).
    Updates the DB row and broadcasts INCIDENT_UPDATE so every connected
    operator's view stays in sync. status="false_positive" also appends a
    row to data/false_positive_log.xlsx (see save_false_positive_record) --
    a local record of the incident_id/raw_text/classification an operator
    overrode, for later model review.
    """
    changes: dict = {"status": update.status}
    if update.is_unread is not None:
        changes["isUnread"] = update.is_unread

    async with get_session() as session:
        incident = await session.get(Incident, incident_id)
        if incident is None:
            raise HTTPException(status_code=404, detail=f"incident '{incident_id}' not found")

        incident.status = update.status
        if update.is_unread is not None:
            incident.is_unread = update.is_unread
        incident.updated_at = datetime.now(timezone.utc)
        changes["updatedAt"] = utc_iso(incident.updated_at)

        if update.status == "escalated":
            entities = await session.get(ExtractedEntities, incident_id)
            _send_escalation_email(incident, entities)
            _place_escalation_call(incident, entities)
        elif update.status == "false_positive":
            save_false_positive_record(
                incident_id=incident.incident_id,
                user_id=incident.user_id,
                platform=incident.platform,
                raw_text=incident.raw_text,
                risk_level=incident.risk_level,
                distress_classification=incident.distress_classification,
                summary=incident.summary,
            )

        await session.commit()

    await manager.broadcast(incident_update_payload(incident_id, changes))
    return {"incidentId": incident_id, "changes": changes}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
