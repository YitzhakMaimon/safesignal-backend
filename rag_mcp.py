"""
FastMCP server exposing deep RAG history lookup for the SafeSignal Decision Agent.

Distinct from the passive RAG context retrieval already performed earlier in the
pipeline (rag_retrieval.py / "/api/v1/rag-context") -- this is an on-demand,
per-user probe of the FastAPI vector store, used only when the Decision Agent
decides the passive context isn't enough to assess this specific user's history.
Run standalone: `python rag_mcp.py` (serves Streamable HTTP on :8002/mcp).
"""
import requests
from mcp.server.fastmcp import FastMCP

RAG_QUERY_URL = "http://localhost:8000/query"

mcp = FastMCP("safesignal-rag-history", host="0.0.0.0", port=8002)


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
    payload = {"user_id": user_id, "target_query": target_query, "fetch_history_only": True}

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
