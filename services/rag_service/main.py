"""
RAG Service -- standalone FastAPI microservice (Layer 3, Service 1).

Wraps rag_core.RAGContextRetriever behind HTTP. Exposes two routes:

- POST /query          Assignment-spec endpoint (guideline Section 4.1):
                       {"description": "<text>"} -> {"similar_listings": [...], "insight": "<text>"}
- POST /api/v1/rag-context   Existing n8n-facing contract, preserved verbatim from
                       safesignal.py's rag_context handler so the current pipeline
                       keeps working unchanged once pointed at this service.

Note on "insight" in /query: this is currently the same human-readable
retrieved-context string build_context() already produced in the monolith --
NOT yet an LLM-generated, citation-aware synthesis. Writing that synthesis
prompt (cite which similar case it draws from, no fabrication) is the RAG
Retrieval Prompt Engineering Log surface (guideline Section 6, surface #3)
and is intentionally left for that task, not invented here.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_core import RAGContextRetriever

rag_retriever: RAGContextRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_retriever
    rag_retriever = RAGContextRetriever()
    rag_retriever.build_index()
    yield


app = FastAPI(title="SafeSignal RAG Service", lifespan=lifespan)

# Same three defaults safesignal.py's PLACEHOLDER_FIELDS used for these keys,
# kept local so this service has no import dependency on the monolith.
DEFAULT_CLASS = "medium"
DEFAULT_RISK_LEVEL = "medium"
DEFAULT_SUMMARY = "סיכום מדומה: המשתמש מביע תחושות בדידות ויאוש"
DEFAULT_TEXT_CONTENT = "בדיקה: אני מרגיש מאוד לבד ומיואש בזמן האחרון"


class QueryRequest(BaseModel):
    description: str


class QueryResponse(BaseModel):
    similar_listings: list[dict] = Field(default_factory=list)
    insight: str


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    assert rag_retriever is not None
    context, examples = rag_retriever.build_context(req.description)
    return QueryResponse(similar_listings=examples, insight=context)


@app.post("/api/v1/rag-context")
async def rag_context(req: dict):
    """Preserved verbatim from safesignal.py's rag_context handler."""
    assert rag_retriever is not None
    text = req.get("text_content") or req.get("text") or DEFAULT_TEXT_CONTENT
    context, retrieved_examples = rag_retriever.build_context(text)

    return {
        "text_content": text,
        "incident_id": req.get("incident_id", ""),
        "user_id": req.get("user_id", "anonymous"),
        "platform": req.get("platform", "telegram"),
        "class": req.get("class", DEFAULT_CLASS),
        "risk_level": req.get("risk_level", DEFAULT_RISK_LEVEL),
        "summary": req.get("summary", DEFAULT_SUMMARY),
        "context": context,
        "retrieved_examples": retrieved_examples,
        "names": req.get("names", []),
        "addresses": req.get("addresses", []),
        "ages": req.get("ages", []),
        "phone_numbers": req.get("phone_numbers", []),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
