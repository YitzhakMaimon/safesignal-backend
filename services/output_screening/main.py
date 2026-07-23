"""
Output Screening microservice (async, T3.micro-optimized) -- AWS Bedrock
Guardrails + Claude Haiku tone/hallucination check, exposed as its own
standalone FastAPI service with a POST /v1/screen contract.

Relationship to services/guardrails_service: that service's output_screening.py
already implements the same two checks (run_bedrock_guardrails /
run_hallucination_check), used synchronously and called in-process by
services/langgraph_agent/agent_graph.py's output_screening_node. This is a
separate, from-scratch async rewrite with a different request/response shape
(POST /v1/screen, not /check/output) -- it is NOT wired into the existing
pipeline yet; see the deployment notes for what that would require.

Two deliberate deviations from a literal "aioboto3 for everything" reading,
both because this exact codebase already has a proven-working answer:
- The Claude Haiku call uses AsyncAnthropicBedrock (from the `anthropic`
  package's Bedrock integration), not a raw aioboto3 bedrock-runtime
  invoke_model() call. services/guardrails_service/output_screening.py
  already uses the sync AnthropicBedrock client successfully, with structured
  JSON-schema output; the async client is the same client, non-blocking
  (httpx-based under the hood), just async. aioboto3 IS used below for
  apply_guardrail, which is a plain boto3-style API with no Anthropic SDK
  involved.
- HAIKU_MODEL_ID defaults to the Haiku 4.5 inference-profile ID already
  verified working in this AWS account/region (see project notes: older/
  legacy Claude model IDs 404 here), not "claude-3-5-haiku".
"""
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import aioboto3
from anthropic import AsyncAnthropicBedrock
from fastapi import FastAPI, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# --- Structured JSON logging ------------------------------------------------


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


logger = logging.getLogger("output_screening")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logger.addHandler(_handler)
logger.propagate = False


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            json.dumps(
                {
                    "event": "request",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
                ensure_ascii=False,
            )
        )
        return response


# --- Config ------------------------------------------------------------

BEDROCK_AWS_REGION = os.environ.get("BEDROCK_AWS_REGION", "us-east-1")
# Already-provisioned Guardrail used elsewhere in this project (PHONE/NAME PII
# types excluded so it doesn't block hotline numbers like 105 / Eran 1201).
BEDROCK_GUARDRAIL_ID = os.environ.get("BEDROCK_GUARDRAIL_ID", "ac9am0yn2k06")
BEDROCK_GUARDRAIL_VERSION = os.environ.get("BEDROCK_GUARDRAIL_VERSION", "DRAFT")
HAIKU_MODEL_ID = os.environ.get(
    "BEDROCK_HALLUCINATION_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)

CALL_TIMEOUT_SECONDS = 3.0
MAX_RETRIES = 2
RETRY_BASE_DELAY_SECONDS = 0.5

# --- App state: one session/client for the process lifetime, not per-request,
# to keep connection/TLS-handshake overhead low on a t3.micro. -------------

_aioboto3_session: Optional[aioboto3.Session] = None
_anthropic_client: Optional[AsyncAnthropicBedrock] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _aioboto3_session, _anthropic_client
    _aioboto3_session = aioboto3.Session(region_name=BEDROCK_AWS_REGION)
    _anthropic_client = AsyncAnthropicBedrock(aws_region=BEDROCK_AWS_REGION)
    logger.info(json.dumps({"event": "startup", "region": BEDROCK_AWS_REGION}))
    yield
    await _anthropic_client.close()
    logger.info(json.dumps({"event": "shutdown"}))


app = FastAPI(title="SafeSignal Output Screening (Async)", lifespan=lifespan)
app.add_middleware(RequestLogMiddleware)


# --- Schemas ------------------------------------------------------------


class ScreenRequest(BaseModel):
    event_id: str
    raw_llm_output: str
    context_severity: str = "unknown"


class ScreenResponse(BaseModel):
    event_id: str
    is_safe: bool
    action_approved: bool
    screened_output: str
    guardrail_triggered: bool
    # "none" | "guardrail_block" | "network_timeout" | "proxy_error"
    failure_reason: str = "none"


HAIKU_SYSTEM_PROMPT = """את/ה בודק/ת בקרת-איכות אחרונה על תשובה שנוצרה ע"י סוכן AI במערכת \
טריאז' מצוקה (SafeSignal), רגע לפני שהיא נשלחת החוצה. קיבלת את "רמת החומרה של ההקשר" \
(context_severity, למשל: low / medium / high / critical) ואת "פלט הסוכן" עצמו.

שתי בדיקות, ששתיהן חייבות לעבור כדי שהפלט ייחשב תקין:
1. התאמת טון (tone_appropriate): האם הטון של הפלט מתאים לחומרת ההקשר? למשל, טון מזלזל, \
קליל מדי או משועשע לגבי אירוע עם context_severity גבוה/קריטי (כמו אירוע התאבדות) הוא \
בעייתי ולא מתאים.
2. הזיה (hallucination_detected): האם הפלט מכיל טעויות עובדתיות שנשמעות כמו שגיאות/הזיה \
(סתירה פנימית, מספרים או פרטים שלא הגיוניים בהקשר הנתון)?

החזר/י JSON בלבד: tone_appropriate (bool), hallucination_detected (bool), reason (משפט קצר \
בעברית שמסביר את ההחלטה)."""

HAIKU_SCHEMA = {
    "type": "object",
    "properties": {
        "tone_appropriate": {"type": "boolean"},
        "hallucination_detected": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["tone_appropriate", "hallucination_detected", "reason"],
    "additionalProperties": False,
}


# --- Retry/timeout helper --------------------------------------------------


async def _with_retry(coro_factory):
    """Runs coro_factory() up to 1 + MAX_RETRIES times, each attempt capped
    at CALL_TIMEOUT_SECONDS, with exponential backoff between attempts.
    Re-raises the last exception if every attempt fails -- callers decide
    what "every attempt failed" means for the response (see screen())."""
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await asyncio.wait_for(coro_factory(), timeout=CALL_TIMEOUT_SECONDS)
        except Exception as e:  # noqa: BLE001 -- intentionally broad: network/proxy/SDK errors all degrade the same way
            last_exc = e
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_BASE_DELAY_SECONDS * (2**attempt))
    raise last_exc


def _failure_reason_for(e: Exception) -> str:
    return "network_timeout" if isinstance(e, asyncio.TimeoutError) else "proxy_error"


# --- Step 1: Bedrock Guardrails (aioboto3) ----------------------------------


async def _apply_guardrail(text: str) -> dict:
    async def _call():
        async with _aioboto3_session.client("bedrock-runtime") as client:
            return await client.apply_guardrail(
                guardrailIdentifier=BEDROCK_GUARDRAIL_ID,
                guardrailVersion=BEDROCK_GUARDRAIL_VERSION,
                source="OUTPUT",
                content=[{"text": {"text": text}}],
            )

    response = await _with_retry(_call)
    return {"blocked": response.get("action", "NONE") == "GUARDRAIL_INTERVENED"}


# --- Step 2: Claude Haiku tone + hallucination check ------------------------


async def _haiku_evaluate(text: str, context_severity: str) -> dict:
    async def _call():
        return await _anthropic_client.messages.create(
            model=HAIKU_MODEL_ID,
            max_tokens=256,
            system=HAIKU_SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": HAIKU_SCHEMA}},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"רמת חומרת ההקשר (context_severity): {context_severity}\n\n"
                        f"פלט הסוכן לבדיקה:\n{text}"
                    ),
                }
            ],
        )

    response = await _with_retry(_call)
    block = next(b for b in response.content if b.type == "text")
    return json.loads(block.text)


# --- Routes ------------------------------------------------------------


@app.post("/v1/screen", response_model=ScreenResponse)
async def screen(req: ScreenRequest) -> ScreenResponse:
    text = req.raw_llm_output or ""

    if not text.strip():
        return ScreenResponse(
            event_id=req.event_id,
            is_safe=True,
            action_approved=True,
            screened_output="",
            guardrail_triggered=False,
            failure_reason="none",
        )

    # Step 1: Bedrock Guardrails
    try:
        guardrail_result = await _apply_guardrail(text)
    except Exception as e:
        logger.error(
            json.dumps(
                {
                    "event": "EMERGENCY_bedrock_unreachable",
                    "step": "guardrail",
                    "event_id": req.event_id,
                    "error": str(e),
                },
                ensure_ascii=False,
            )
        )
        return ScreenResponse(
            event_id=req.event_id,
            is_safe=False,
            action_approved=False,
            screened_output="",
            guardrail_triggered=False,
            failure_reason=_failure_reason_for(e),
        )

    if guardrail_result["blocked"]:
        return ScreenResponse(
            event_id=req.event_id,
            is_safe=False,
            action_approved=False,
            screened_output="",
            guardrail_triggered=True,
            failure_reason="guardrail_block",
        )

    # Step 2: Claude Haiku tone-match + hallucination evaluation
    try:
        haiku_result = await _haiku_evaluate(text, req.context_severity)
    except Exception as e:
        logger.error(
            json.dumps(
                {
                    "event": "EMERGENCY_bedrock_unreachable",
                    "step": "haiku",
                    "event_id": req.event_id,
                    "error": str(e),
                },
                ensure_ascii=False,
            )
        )
        return ScreenResponse(
            event_id=req.event_id,
            is_safe=False,
            action_approved=False,
            screened_output="",
            guardrail_triggered=False,
            failure_reason=_failure_reason_for(e),
        )

    is_safe = bool(haiku_result.get("tone_appropriate")) and not bool(
        haiku_result.get("hallucination_detected")
    )
    return ScreenResponse(
        event_id=req.event_id,
        is_safe=is_safe,
        action_approved=is_safe,
        # Honest, not fabricated: this service only decides pass/fail, it
        # does not redact or rewrite -- same convention as
        # services/guardrails_service's /check/output safe_text field.
        screened_output=text if is_safe else "",
        guardrail_triggered=False,
        failure_reason="none" if is_safe else "guardrail_block",
    )


@app.get("/health")
async def health():
    """Fast, non-blocking connectivity check. Uses STS get_caller_identity
    (cheap, no Bedrock-specific IAM permission needed) rather than an actual
    Bedrock model/guardrail call, which would have real cost and require
    extra permissions just to answer a health probe."""

    async def _call():
        async with _aioboto3_session.client("sts") as client:
            return await client.get_caller_identity()

    try:
        await asyncio.wait_for(_call(), timeout=2.0)
        return {"status": "ok", "aws_reachable": True}
    except Exception as e:
        return {"status": "degraded", "aws_reachable": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005, workers=1)
