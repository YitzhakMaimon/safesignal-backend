"""
LangGraph Agent -- standalone FastAPI microservice (Layer 3, Service 4).

Two endpoints:
- POST /agent/run             Assignment-spec endpoint (guideline Section 4.4):
                               {"query": "<text>"} -> {"answer", "tools_used", "reasoning_steps"}
- POST /api/v1/decision-agent Existing n8n-facing contract, preserved verbatim
                               from safesignal.py's decision_agent() handler.

Honesty note on /agent/run's "reasoning_steps": this graph produces one
`thought_process` string per run, not a granular multi-step trace -- so
reasoning_steps is a single-item list, not fabricated multi-step reasoning.
"""
import asyncio
import uuid
from contextlib import AsyncExitStack, asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel

load_dotenv()

from agent_graph import create_decision_agent

decision_graph = None
_mcp_exit_stack: AsyncExitStack | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global decision_graph, _mcp_exit_stack

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


app = FastAPI(title="SafeSignal LangGraph Agent", lifespan=lifespan)

PLACEHOLDER_FIELDS = {
    "text_content": "בדיקה: אני מרגיש מאוד לבד ומיואש בזמן האחרון",
    "summary": "סיכום מדומה: המשתמש מביע תחושות בדידות ויאוש",
    "class": "medium",
    "risk_level": "medium",
    "context": "הקשר מדומה מתוך בסיס הידע: פנה לקו סיוע 105 או ער\"ן 1201",
    "decision": "route_to_human_review",
    "agent_output": "מומלץ לבצע בדיקה אנושית בהקדם (stub)",
    "passed": True,
}


def merged(body: dict, **overrides) -> dict:
    return {**PLACEHOLDER_FIELDS, **body, **overrides}


class AgentRunRequest(BaseModel):
    query: str


@app.post("/agent/run")
async def agent_run(req: AgentRunRequest):
    if decision_graph is None:
        return {"answer": "", "tools_used": [], "reasoning_steps": ["decision graph unavailable"]}

    initial_state = {
        "raw_input": req.query,
        "user_id": "anonymous",
        "incident_id": str(uuid.uuid4()),
        "distress_classification": "",
        "initial_rag_context": "",
    }
    result = await decision_graph.ainvoke(initial_state)

    thought_process = result.get("thought_process", "")
    return {
        "answer": result.get("summary_for_human_reviewer") or result.get("recommended_action") or "",
        "tools_used": result.get("tools_triggered", []),
        "reasoning_steps": [thought_process] if thought_process else [],
    }


@app.post("/api/v1/decision-agent")
async def decision_agent(req: Request):
    """Preserved verbatim from safesignal.py's decision_agent() handler."""
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
        passed=result.get("screening_passed", True),
        risk_level=result.get("risk_level", ""),
        screening_tags=result.get("screening_tags", []),
        screening_reason=result.get("screening_reason", ""),
        screening_logs=result.get("screening_logs", []),
        output_retry_count=result.get("output_retry_count", 0),
        decision=result.get("recommended_action", ""),
        agent_output=result.get("summary_for_human_reviewer", ""),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
