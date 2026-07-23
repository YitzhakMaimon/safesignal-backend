"""
Output-side screening -- extracted from decision_agent_graph.py's
_run_bedrock_guardrails / _run_hallucination_check. NOT from ml_comprehend.py
(that file has no output-side logic at all -- see main.py's docstring).

Only the two pure check functions are extracted here. The LangGraph-specific
orchestration around them (output_screening_node: retry counting, routing to
decision_agent/human_review, reading/writing AgentState) stays in
decision_agent_graph.py, which currently calls these checks in-process.
Repointing it to call this service over HTTP instead is follow-up work for
Service #4 (langgraph_agent), not done here.
"""
import json
import os

import boto3
from anthropic import AnthropicBedrock
from botocore.exceptions import BotoCoreError, ClientError

BEDROCK_AWS_REGION = os.environ.get("BEDROCK_AWS_REGION", "us-east-1")
BEDROCK_GUARDRAIL_ID = os.environ.get("BEDROCK_GUARDRAIL_ID", "")
BEDROCK_GUARDRAIL_VERSION = os.environ.get("BEDROCK_GUARDRAIL_VERSION", "DRAFT")
HALLUCINATION_MODEL_ID = os.environ.get(
    "BEDROCK_HALLUCINATION_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)

HALLUCINATION_SYSTEM_PROMPT = """את/ה בודק/ת בקרת-איכות על תשובה שנוצרה ע"י סוכן AI במערכת \
טריאז' מצוקה (SafeSignal). קיבלת כמה קטעי טקסט: "הודעת המשתמש המקורית" (raw input), \
"הסיווג האוטומטי" (distress classification) שנקבע לה קודם בפייפליין, ה"הקשר" (context) \
שאותר ממאגר הידע (RAG), "כלים שהופעלו בפועל" (tools actually executed) - רשימת שמות \
הכלים שהסוכן קרא להם בפועל במהלך הריצה, ו"פלט הסוכן" (agent output) - הערכת המצב הסופית \
שהסוכן ניסח.

תפקידך: לבדוק אם פלט הסוכן כולל טענות עובדתיות שאינן נתמכות ע"י אף אחד מהמקורות שסופקו \
(הודעת המשתמש המקורית, הסיווג האוטומטי, הקשר ה-RAG, או רשימת הכלים שהופעלו בפועל) - כלומר \
פרטים, המלצות פעולה קונקרטיות או קביעות עובדתיות שהסוכן "המציא" ואינן מבוססות על אף אחד \
מהם. ציטוט או תיאור של הודעת המשתמש המקורית, הפניה לסיווג האוטומטי שכבר נקבע, או טענה \
שפעולה/התראה מסוימת בוצעה/נשלחה - כאשר שם הכלי המתאים (למשל trigger_immediate_alert) \
מופיע ברשימת "כלים שהופעלו בפועל" - אינם הזיה, גם אם אינם מופיעים במאגר ה-RAG. טענה על \
פעולה שבוצעה כש*אין* כלי תואם ברשימה כן נחשבת הזיה.

אל תסמן כהזיה: ניסוח מחדש סביר, מסקנות לוגיות ישירות מההקשר, או שימוש בשיקול דעת מקצועי \
כללי (כמו "מומלץ ליצור קשר עם קו סיוע") שאינו סותר את ההקשר.

החזר/י JSON בלבד: hallucination_detected (bool) ו-reason (משפט קצר בעברית המסביר את ההחלטה).
"""

HALLUCINATION_SCHEMA = {
    "type": "object",
    "properties": {
        "hallucination_detected": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["hallucination_detected", "reason"],
    "additionalProperties": False,
}


def run_bedrock_guardrails(text: str) -> dict:
    if not text or not text.strip():
        return {"blocked": False, "tags": [], "reason": "empty_output_nothing_to_screen"}

    if not BEDROCK_GUARDRAIL_ID:
        return {
            "blocked": False,
            "tags": [],
            "reason": "not_implemented: BEDROCK_GUARDRAIL_ID not configured -- guardrail check skipped",
        }

    try:
        client = boto3.client("bedrock-runtime", region_name=BEDROCK_AWS_REGION)
        response = client.apply_guardrail(
            guardrailIdentifier=BEDROCK_GUARDRAIL_ID,
            guardrailVersion=BEDROCK_GUARDRAIL_VERSION,
            source="OUTPUT",
            content=[{"text": {"text": text}}],
        )
    except (BotoCoreError, ClientError) as e:
        print(f"[Bedrock Guardrails Error] {e}")
        return {"blocked": True, "tags": ["guardrail_call_failed"], "reason": f"guardrail_call_failed: {e}"}

    action = response.get("action", "NONE")
    tags = sorted({key for assessment in response.get("assessments", []) for key in assessment.keys()})
    return {
        "blocked": action == "GUARDRAIL_INTERVENED",
        "tags": tags,
        "reason": f"bedrock_guardrail_action={action}",
    }


def run_hallucination_check(
    agent_output: str,
    rag_context: str,
    raw_input: str = "",
    distress_classification: str = "",
    tools_triggered: list[str] | None = None,
) -> dict:
    if not agent_output or not agent_output.strip():
        return {"hallucination_detected": False, "reason": "empty_output_nothing_to_check"}

    try:
        client = AnthropicBedrock(aws_region=BEDROCK_AWS_REGION)
        user_content = (
            f"הודעת המשתמש המקורית:\n{raw_input or '(לא זמינה)'}\n\n"
            f"הסיווג האוטומטי שנקבע לפנייה:\n{distress_classification or '(לא זמין)'}\n\n"
            f"הקשר (RAG context) שאותר עבור הפנייה:\n{rag_context or '(אין הקשר זמין)'}\n\n"
            f"כלים שהופעלו בפועל:\n{tools_triggered or '(לא הופעל אף כלי)'}\n\n"
            f"פלט הסוכן לבדיקה:\n{agent_output}"
        )
        response = client.messages.create(
            model=HALLUCINATION_MODEL_ID,
            max_tokens=512,
            system=HALLUCINATION_SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": HALLUCINATION_SCHEMA}},
            messages=[{"role": "user", "content": user_content}],
        )
        block = next(b for b in response.content if b.type == "text")
        parsed = json.loads(block.text)
        return {
            "hallucination_detected": bool(parsed["hallucination_detected"]),
            "reason": parsed["reason"],
        }
    except Exception as e:
        print(f"[Hallucination Check Error] {e}")
        return {"hallucination_detected": True, "reason": f"hallucination_check_failed: {e}"}
