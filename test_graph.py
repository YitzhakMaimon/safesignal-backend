"""
Manual smoke test for decision_agent_graph.py.

Runs against the local mock tools (build_mock_tools() in decision_agent_graph.py)
instead of the real alert_mcp.py / rag_mcp.py MCP servers -- no network calls, no
async MCP client session negotiation, so this doesn't depend on either server
being up or on the MCP async transport working in this environment. Requires
only GOOGLE_API_KEY (decision_agent_graph.py uses Gemini via ChatGoogleGenerativeAI).
Run with: python test_graph.py

Note: even with mock tools, the graph itself is unchanged and still async, so
this drives it with `await graph.ainvoke(...)` rather than a synchronous
`.invoke()`.
"""
import asyncio
import os
from contextlib import AsyncExitStack

from decision_agent_graph import create_decision_agent


async def run_simulation(graph, test_name: str, payload: dict):
    print(f"\n=== מריץ בדיקה: {test_name} ===", flush=True)

    initial_state = {
        "raw_input": payload["raw_input"],
        "user_id": payload["user_id"],
        "incident_id": payload["incident_id"],
        "distress_classification": payload["distress_classification"],
        "initial_rag_context": "No initial context found.",
    }

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:
        # Isolate failures per scenario -- one case erroring out (rate limit,
        # transient API error, etc.) shouldn't prevent the other from running.
        print(f"בדיקה '{test_name}' נכשלה: {type(e).__name__}: {e}\n", flush=True)
        return

    print(f"Thought Process: {final_state.get('thought_process')}")
    print(f"Tools Triggered: {final_state.get('tools_triggered')}")
    print(f"Final Urgency: {final_state.get('final_urgency_assessment')}")
    print(f"Recommended Action: {final_state.get('recommended_action')}")
    print(f"Summary for Reviewer: {final_state.get('summary_for_human_reviewer')}\n", flush=True)


# --- תרחיש בדיקה 1: מקרה חירום קיצוני (צריך להפעיל את כלי החירום) ---
case_1 = {
    "raw_input": "תעזרו לי מהר יש מישהו שמנסה לפרוץ את הדלת עכשיו!",
    "user_id": "user_123",
    "incident_id": "inc_999",
    "distress_classification": "Critical",
}

# --- תרחיש בדיקה 2: הודעה מעורפלת (צריך להפעיל כלי RAG לבדיקת היסטוריה) ---
case_2 = {
    "raw_input": "הוא שוב פה מחוץ לבית שלי, בדיוק כמו פעם קודמת תסתכלו במערכת",
    "user_id": "user_123",
    "incident_id": "inc_888",
    "distress_classification": "Medium",
}


async def main():
    async with AsyncExitStack() as exit_stack:
        graph = await create_decision_agent(exit_stack, use_mock_tools=True)
        await run_simulation(graph, "תרחיש חירום מיידי", case_1)
        await run_simulation(graph, "תרחיש מידע עבר מעורפל", case_2)


if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("שגיאה: נא להגדיר את GOOGLE_API_KEY בטרמינל לפני ההרצה.")
    else:
        asyncio.run(main())
