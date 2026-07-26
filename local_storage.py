import os
import threading
from datetime import datetime

from openpyxl import Workbook, load_workbook

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ERROR_LOG_PATH = os.path.join(DATA_DIR, "error_log.xlsx")
ERROR_LOG_SHEET_NAME = "Errors"
ERROR_LOG_COLUMNS = [
    "timestamp", "source", "incident_id", "user_id", "error_message",
]

FALSE_POSITIVE_LOG_PATH = os.path.join(DATA_DIR, "false_positive_log.xlsx")
FALSE_POSITIVE_LOG_SHEET_NAME = "FalsePositives"
FALSE_POSITIVE_LOG_COLUMNS = [
    "timestamp", "incident_id", "user_id", "platform", "raw_text",
    "risk_level", "distress_classification", "summary",
]

TOKENS_LOG_PATH = os.path.join(DATA_DIR, "tokens_log.xlsx")
TOKENS_LOG_SHEET_NAME = "TokenUsage"

# One column per model/pipeline-stage that consumes real tokens anywhere in the
# pipeline (ml_comprehend.py, distress_classification.py, info_extraction.py,
# decision_agent_graph.py) -- a sentence gets ONE row here no matter how many of
# these it passes through in a full run; whichever ones it never reached stay 0.
TOKENS_LOG_MODEL_COLUMNS = [
    "HeBERT_screening_tokens",
    "Bedrock_Classification_tokens",
    "Bedrock_Translation_CrossCheck_tokens",
    "Ollama_Extraction_tokens",
    "Gemini_DecisionAgent_tokens",
    "Bedrock_HallucinationCheck_tokens",
]
TOKENS_LOG_COLUMNS = ["timestamp", "incident_id", "sentence", *TOKENS_LOG_MODEL_COLUMNS, "total_tokens"]

# Maps each call site's pipeline_stage string to its column above. Both Gemini calls
# inside decision_agent_node (tool-check pass + structured-output pass) share one
# column -- same model, same node -- and add together rather than overwrite.
_STAGE_TO_TOKENS_COLUMN = {
    "ml_comprehend_hebert_screening": "HeBERT_screening_tokens",
    "distress_classification": "Bedrock_Classification_tokens",
    "ml_comprehend_translation_crosscheck": "Bedrock_Translation_CrossCheck_tokens",
    "info_extraction": "Ollama_Extraction_tokens",
    "decision_agent_tool_check": "Gemini_DecisionAgent_tokens",
    "decision_agent_structured_output": "Gemini_DecisionAgent_tokens",
    "decision_agent_hallucination_check": "Bedrock_HallucinationCheck_tokens",
}

_lock = threading.Lock()


def _append_row(path: str, sheet_name: str, columns: list[str], row: list) -> None:
    """
    Shared append-with-init helper for all local Excel-backed tables in this
    module. Kept local-only (no S3/RDS) by design -- this data is sensitive
    enough that it stays on disk, not in the cloud, until a real encrypted
    store replaces it.
    """
    with _lock:
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            wb = Workbook()
            ws = wb.active
            ws.title = sheet_name
            ws.append(columns)
            wb.save(path)

        wb = load_workbook(path)
        ws = wb[sheet_name] if sheet_name in wb.sheetnames else wb.active
        ws.append(row)
        wb.save(path)


def save_error_record(
    source: str,
    error_message: str,
    incident_id: str = "",
    user_id: str = "",
    path: str = ERROR_LOG_PATH,
) -> None:
    """
    רשומת שגיאה מקומית (data/error_log.xlsx) - אותו דפוס בדיוק כמו
    review_queue/immediate_alert. יעד: מסך שגיאות עתידי ב-UI שיקרא מכאן
    דרך GET /api/v1/errors, כדי שכשל שרת (למשל n8n שלא הצליח להגיע ל-
    backend) לא ייבלע בשקט אלא יופיע למשתמש.
    """
    _append_row(path, ERROR_LOG_SHEET_NAME, ERROR_LOG_COLUMNS, [
        datetime.now().isoformat(timespec="seconds"),
        source,
        incident_id,
        user_id,
        error_message,
    ])


def get_error_records(path: str = ERROR_LOG_PATH) -> list[dict]:
    """קורא את data/error_log.xlsx בחזרה כרשימת dict-ים עבור מסך השגיאות ב-UI."""
    if not os.path.exists(path):
        return []

    with _lock:
        wb = load_workbook(path)
        ws = wb[ERROR_LOG_SHEET_NAME] if ERROR_LOG_SHEET_NAME in wb.sheetnames else wb.active
        rows = list(ws.iter_rows(min_row=2, values_only=True))

    return [dict(zip(ERROR_LOG_COLUMNS, row)) for row in rows]


def save_false_positive_record(
    incident_id: str,
    user_id: str = "",
    platform: str = "",
    raw_text: str = "",
    risk_level: str = "",
    distress_classification: str = "",
    summary: str = "",
    path: str = FALSE_POSITIVE_LOG_PATH,
) -> None:
    """
    רשומת False Positive מקומית (data/false_positive_log.xlsx) - אותו דפוס
    בדיוק כמו error_log. נכתב מ-update_incident_status ב-safesignal.py כשמפעיל
    המערכת מסמן אירוע כ-False Positive, כדי לשמור תיעוד מקומי (מזהה אירוע,
    הטקסט המקורי ועוד פרטי הסיווג) לצורך בדיקה/שיפור המודל בהמשך.
    """
    _append_row(path, FALSE_POSITIVE_LOG_SHEET_NAME, FALSE_POSITIVE_LOG_COLUMNS, [
        datetime.now().isoformat(timespec="seconds"),
        incident_id,
        user_id,
        platform,
        raw_text,
        risk_level,
        distress_classification,
        summary,
    ])


def save_token_usage_record(
    pipeline_stage: str,
    model_id: str,
    sentence: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    incident_id: str = "",
    path: str = TOKENS_LOG_PATH,
) -> None:
    """
    רשומת צריכת טוקנים מקומית (data/tokens_log.xlsx), נכתבת מכל אחת מנקודות
    הקריאה ל-LLM/מודל בפייפליין (ml_comprehend.py, distress_classification.py,
    info_extraction.py, decision_agent_graph.py), בין אם הריצה הגיעה מתנועה
    אמיתית דרך n8n/safesignal.py ובין אם מהרצה ידנית של בלוק הבדיקה (__main__)
    של אחד המודולים.

    בניגוד ל-error_log/false_positive_log (append-only, שורה חדשה בכל קריאה),
    כאן יש שורה אחת בלבד לכל משפט (התאמה לפי טקסט מדויק): אם המשפט כבר קיים
    בטבלה, מוסיפים את הטוקנים לעמודת המודל הרלוונטית באותה שורה במקום לפתוח
    שורה חדשה - כך שמשפט שעובר כמה שלבים בפייפליין (סינון/סיווג/חילוץ/סוכן
    ההחלטה) מסתכם בשורה אחת עם עמודה לכל מודל, ולא מתפצל לכמה שורות.
    """
    column = _STAGE_TO_TOKENS_COLUMN.get(pipeline_stage)
    if column is None:
        raise ValueError(f"Unknown pipeline_stage for token logging: {pipeline_stage!r}")
    tokens = input_tokens + output_tokens

    with _lock:
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            wb = Workbook()
            ws = wb.active
            ws.title = TOKENS_LOG_SHEET_NAME
            ws.append(TOKENS_LOG_COLUMNS)
            wb.save(path)

        wb = load_workbook(path)
        ws = wb[TOKENS_LOG_SHEET_NAME] if TOKENS_LOG_SHEET_NAME in wb.sheetnames else wb.active

        sentence_idx = TOKENS_LOG_COLUMNS.index("sentence")
        column_idx = TOKENS_LOG_COLUMNS.index(column)
        incident_idx = TOKENS_LOG_COLUMNS.index("incident_id")
        timestamp_idx = TOKENS_LOG_COLUMNS.index("timestamp")
        total_idx = TOKENS_LOG_COLUMNS.index("total_tokens")

        target_row = next(
            (row for row in ws.iter_rows(min_row=2) if row[sentence_idx].value == sentence),
            None,
        )
        if target_row is None:
            blank_row = ["" if c in ("timestamp", "incident_id", "sentence") else 0 for c in TOKENS_LOG_COLUMNS]
            ws.append(blank_row)
            target_row = ws[ws.max_row]

        target_row[timestamp_idx].value = datetime.now().isoformat(timespec="seconds")
        target_row[incident_idx].value = incident_id or target_row[incident_idx].value
        target_row[sentence_idx].value = sentence
        target_row[column_idx].value = (target_row[column_idx].value or 0) + tokens
        target_row[total_idx].value = sum(
            (target_row[TOKENS_LOG_COLUMNS.index(c)].value or 0) for c in TOKENS_LOG_MODEL_COLUMNS
        )

        wb.save(path)
