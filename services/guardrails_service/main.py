"""
Guardrails Service -- standalone FastAPI microservice (Layer 3, Service 3).

Two independent sources, NOT one:
- Input screening (POST /check/input, POST /api/v1/screening/input) is
  extracted verbatim from ml_comprehend.py's DistressScreeningPipeline
  (HeBERT for Hebrew + AWS Comprehend for other supported languages,
  cross-lingual cross-check via Bedrock Claude Haiku translation).
- Output screening (POST /check/output) is extracted from
  decision_agent_graph.py's _run_bedrock_guardrails / _run_hallucination_check
  -- the REAL, currently-active output guardrail in this project (AWS
  Bedrock Guardrails + Claude Haiku hallucination check). ml_comprehend.py
  itself has no output-side logic at all.

  The legacy safesignal.py `/api/v1/screening/output` endpoint (Comprehend
  PII + an untrained placeholder classifier) is NOT reproduced here --
  that file's own docstring already marks it "no longer called by
  decision_agent_graph.py", i.e. dead code superseded by the Bedrock-based
  checks this service now exposes over HTTP.

`/check/output`'s "safe_text" field is honest, not fabricated: this service
only makes a pass/fail decision plus a reason, it does not redact or rewrite
the text, so safe_text is just the original text on pass and empty on fail.
"""
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from input_screening import DistressScreeningPipeline
from output_screening import run_bedrock_guardrails, run_hallucination_check

distress_pipeline: DistressScreeningPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global distress_pipeline
    distress_pipeline = DistressScreeningPipeline()
    yield


app = FastAPI(title="SafeSignal Guardrails Service", lifespan=lifespan)

PLACEHOLDER_TEXT_CONTENT = "בדיקה: אני מרגיש מאוד לבד ומיואש בזמן האחרון"


class CheckInputRequest(BaseModel):
    text: str


class CheckOutputRequest(BaseModel):
    text: str
    rag_context: str = ""
    raw_input: str = ""
    distress_classification: str = ""
    tools_triggered: list[str] = []


@app.post("/check/input")
async def check_input(req: CheckInputRequest):
    assert distress_pipeline is not None
    result = distress_pipeline.analyze_post(req.text)
    return {"pass": result["passed_screening"], "reason": result.get("reason", "")}


@app.post("/check/output")
async def check_output(req: CheckOutputRequest):
    guardrail_result = run_bedrock_guardrails(req.text)
    if guardrail_result["blocked"]:
        hallucination_result = {"hallucination_detected": False, "reason": "skipped: blocked by guardrails"}
    else:
        hallucination_result = run_hallucination_check(
            req.text, req.rag_context, req.raw_input, req.distress_classification, req.tools_triggered
        )

    passed = not guardrail_result["blocked"] and not hallucination_result["hallucination_detected"]
    reason = hallucination_result["reason"] if not guardrail_result["blocked"] else guardrail_result["reason"]

    return {
        "pass": passed,
        "reason": reason,
        "safe_text": req.text if passed else "",
        # Additive, backward-compatible: callers that only read pass/reason/
        # safe_text are unaffected. langgraph_agent's output_screening_node
        # needs this breakdown to preserve its existing retry-message and
        # screening_logs granularity (which check specifically failed).
        "guardrail": guardrail_result,
        "hallucination_check": hallucination_result,
    }


@app.post("/api/v1/screening/input")
async def screening_input(req: dict):
    """Preserved verbatim from safesignal.py's screening_input handler."""
    assert distress_pipeline is not None
    real_text = req.get("text") or req.get("text_content") or PLACEHOLDER_TEXT_CONTENT
    result = distress_pipeline.analyze_post(real_text)
    return {
        **req,
        "passed": result["passed_screening"],
        "text_content": real_text,
        "screening_reason": result["reason"],
        "screening_metrics": result.get("raw_metrics", {}),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
