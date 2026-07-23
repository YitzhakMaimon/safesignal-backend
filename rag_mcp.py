"""
FastMCP server exposing deep RAG history lookup for the SafeSignal Decision Agent.

Distinct from the passive RAG context retrieval already performed earlier in the
pipeline (rag_retrieval.py / "/api/v1/rag-context") -- this is an on-demand,
per-user probe of the FastAPI vector store, used only when the Decision Agent
decides the passive context isn't enough to assess this specific user's history.
Run standalone: `python rag_mcp.py` (serves Streamable HTTP on :8012/mcp by default).

Moved off :8002 (2026-07-22) -- that port is now services/image_analyser's HTTP
API in the 4-service microservices split; running both on :8002 would collide.

Also fixes a pre-existing dead reference: RAG_QUERY_URL used to point at
http://localhost:8000/query, an endpoint that never existed on the monolith
(only /api/v1/rag-context did) -- so this tool has been silently broken
(always returning a connection-error dict) since it was written. It now
points at services/rag_service's real POST /query endpoint (built and
verified working in this same refactor).
"""
import os

import requests
from mcp.server.fastmcp import FastMCP

RAG_QUERY_URL = os.environ.get("RAG_SERVICE_QUERY_URL", "http://localhost:8001/query")
MCP_PORT = int(os.environ.get("RAG_MCP_PORT", "8012"))

mcp = FastMCP("safesignal-rag-history", host="0.0.0.0", port=MCP_PORT)


@mcp.tool()
def query_rag_history(user_id: str, target_query: str) -> dict:
    """
    Fetch a specific user's historical distress patterns and past incident logs
    from the internal vector store. Call this when the user's own history --
    not just similar cases from the general knowledge base -- would materially
    change the urgency assessment.

    Args:
        user_id: Identifier of the user whose history should be searched.
        target_query: The specific question or topic to search the user's history for.
    """
    # NOTE (found 2026-07-22, not new behavior): rag_service's vector store has
    # no per-user partitioning -- there's no way to actually scope this search
    # to just `user_id`'s history. This was already true when this tool was
    # unreachable (dead URL); fixing the URL doesn't add that capability, it
    # just makes the same general-knowledge-base search actually reachable.
    # user_id is accepted for API-shape compatibility / future use, not applied.
    payload = {"description": target_query}

    try:
        response = requests.post(RAG_QUERY_URL, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        return {"error": f"could not reach RAG vector store: {e}"}

    if response.status_code != 200:
        return {"error": f"RAG vector store returned status {response.status_code}: {response.text}"}

    try:
        return response.json()
    except ValueError:
        return {"error": "RAG vector store returned a non-JSON response"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
