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
import os
import uuid
from contextlib import AsyncExitStack, asynccontextmanager

import requests
from fastapi import FastAPI, Request
from groq import Groq
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn

from ml_comprehend import DistressScreeningPipeline
from info_extraction import RelevantInfoExtractor
from distress_classification import BedrockDistressClassifier
from rag_retrieval import RAGContextRetriever
from decision_agent_graph import create_decision_agent, log_raw_message
from local_storage import (
    save_screened_record,
    save_review_queue_record,
    save_error_record,
    get_error_records,
)

distress_pipeline: DistressScreeningPipeline | None = None
info_extractor: RelevantInfoExtractor | None = None
distress_classifier: BedrockDistressClassifier | None = None
rag_retriever: RAGContextRetriever | None = None
decision_graph = None  # compiled LangGraph Decision Agent; None if MCP servers are unreachable
_mcp_exit_stack: AsyncExitStack | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global distress_pipeline, info_extractor, distress_classifier, rag_retriever
    global decision_graph, _mcp_exit_stack
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

    await _mcp_exit_stack.aclose()


app = FastAPI(lifespan=lifespan)

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


def _analyze_image(image_url: str) -> str:
    """
    Real vision analysis via Gemini (multimodal): OCRs any text visible in the
    image and separately flags visual distress signals the text alone wouldn't
    capture (e.g. a photo of a dangerous situation with no caption). One model
    call covers both "Textract" and "Rekognition" roles from the node's name.
    Fails *open* to the stub placeholder text on any error, same convention as
    _transcribe_audio -- a vision hiccup degrades gracefully instead of crashing
    the ingestion pipeline.
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
    detected_text = _analyze_image(body.get("image_url", ""))
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
    result = distress_pipeline.analyze_post(real_text)
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
    extracted = info_extractor.extract(text)
    save_screened_record(
        text_content=text,
        screening_reason=body.get("screening_reason", ""),
        names=extracted["names"],
        addresses=extracted["addresses"],
        ages=extracted["ages"],
    )
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
    result = distress_classifier.classify(text)
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


@app.post("/api/v1/decision-agent")
async def decision_agent(req: Request):
    body = await req.json()

    if decision_graph is None:
        classification = body.get("classification", body.get("class", "רגיל"))
        return merged(body, decision=f"route_to_human_review ({classification})")

    text = body.get("text_content") or body.get("text") or PLACEHOLDER_FIELDS["text_content"]
    initial_state = {
        "raw_input": text,
        "user_id": body.get("user_id", "anonymous"),
        "incident_id": body.get("incident_id") or str(uuid.uuid4()),
        "distress_classification": body.get("class", body.get("risk_level", "unknown")),
        "initial_rag_context": body.get("context", ""),
        "names": body.get("names", []),
        "addresses": body.get("addresses", []),
        "ages": body.get("ages", []),
        "phone_numbers": body.get("phone_numbers", []),
    }
    result = await decision_graph.ainvoke(initial_state)

    return merged(
        body,
        thought_process=result.get("thought_process", ""),
        tools_triggered=result.get("tools_triggered", []),
        final_urgency_assessment=result.get("final_urgency_assessment", ""),
        recommended_action=result.get("recommended_action", ""),
        summary_for_human_reviewer=result.get("summary_for_human_reviewer", ""),
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

    if pii_entities:
        risk_level = "Critical"
    elif not classifier_passed:
        risk_level = "Medium"
    else:
        risk_level = "Low"

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
    יעד ה-Human Review בפועל: אין מופע Open WebUI אמיתי לדחוף אליו, אז זה נשמר
    לתור מקומי (data/review_queue.xlsx) שדשבורד עתידי יוכל לקרוא ממנו.
    """
    body = await req.json()
    screening_logs = body.get("screening_logs") or []
    last_screening_reason = screening_logs[-1].get("hallucination_check", {}).get("reason", "") if screening_logs else ""
    save_review_queue_record(
        incident_id=body.get("incident_id", ""),
        user_id=body.get("user_id", "anonymous"),
        risk_level=body.get("risk_level", ""),
        summary_for_human_reviewer=body.get("summary_for_human_reviewer", ""),
        screening_reason=last_screening_reason,
    )
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
    יוכל להציג אותו למשתמש, במקום שהכשל ייבלע בשקט.
    """
    body = await req.json()
    save_error_record(
        source=body.get("source", "unknown"),
        error_message=body.get("error_message", ""),
        incident_id=body.get("incident_id", ""),
        user_id=body.get("user_id", ""),
    )
    return {"status": "logged"}


@app.get("/api/v1/errors")
async def list_errors():
    """מסך השגיאות העתידי ב-UI קורא מכאן - רשימת כל השגיאות שנרשמו ב-data/error_log.xlsx."""
    return {"errors": get_error_records()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
