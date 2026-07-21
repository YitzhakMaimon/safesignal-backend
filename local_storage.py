import os
import threading
from datetime import datetime

from openpyxl import Workbook, load_workbook

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EXCEL_PATH = os.path.join(DATA_DIR, "screened_records.xlsx")
REVIEW_QUEUE_PATH = os.path.join(DATA_DIR, "review_queue.xlsx")

SHEET_NAME = "Screened Records"
COLUMNS = ["timestamp", "text_content", "screening_reason", "names", "addresses", "ages"]

REVIEW_QUEUE_SHEET_NAME = "Review Queue"
REVIEW_QUEUE_COLUMNS = [
    "timestamp", "incident_id", "user_id", "risk_level", "summary_for_human_reviewer", "screening_reason",
]

IMMEDIATE_ALERT_PATH = os.path.join(DATA_DIR, "immediate_alert_log.xlsx")
IMMEDIATE_ALERT_SHEET_NAME = "Immediate Alerts"
IMMEDIATE_ALERT_COLUMNS = [
    "timestamp", "incident_id", "user_id", "risk_level", "alert_status", "urgency_reason",
]

ERROR_LOG_PATH = os.path.join(DATA_DIR, "error_log.xlsx")
ERROR_LOG_SHEET_NAME = "Errors"
ERROR_LOG_COLUMNS = [
    "timestamp", "source", "incident_id", "user_id", "error_message",
]

_lock = threading.Lock()


def _append_row(path: str, sheet_name: str, columns: list[str], row: list) -> None:
    """
    Shared append-with-init helper for all local Excel-backed tables in this
    module. Kept local-only (no S3/RDS) by design -- see save_screened_record's
    docstring: this data is sensitive enough that it stays on disk, not in the
    cloud, until a real encrypted store replaces it.
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


def save_screened_record(
    text_content: str,
    screening_reason: str,
    names: list,
    addresses: list,
    ages: list,
    path: str = EXCEL_PATH,
) -> None:
    """
    מוסיף שורה לקובץ Excel מקומי (data/screened_records.xlsx - לא בענן)
    עבור כל הודעה שעברה סינון ראשוני (Input Screening) וחולצו ממנה פרטים ב-
    Relevant Information Extraction. מכיוון שמדובר במידע רגיש (שם/כתובת/גיל),
    האחסון נשאר מקומי בכוונה, ללא S3 או כל שירות ענן.
    """
    _append_row(path, SHEET_NAME, COLUMNS, [
        datetime.now().isoformat(timespec="seconds"),
        text_content,
        screening_reason,
        ", ".join(names),
        ", ".join(addresses),
        ", ".join(str(a) for a in ages),
    ])


def save_review_queue_record(
    incident_id: str,
    user_id: str,
    risk_level: str,
    summary_for_human_reviewer: str,
    screening_reason: str = "",
    path: str = REVIEW_QUEUE_PATH,
) -> None:
    """
    תור Human Review מקומי (data/review_queue.xlsx) - אין כרגע מופע Open WebUI
    אמיתי לדחוף אליו, אז זה משמש כמקום שממנו דשבורד עתידי (או אדם) יוכל לקרוא
    את התיקים הממתינים לבדיקה אנושית.
    """
    _append_row(path, REVIEW_QUEUE_SHEET_NAME, REVIEW_QUEUE_COLUMNS, [
        datetime.now().isoformat(timespec="seconds"),
        incident_id,
        user_id,
        risk_level,
        summary_for_human_reviewer,
        screening_reason,
    ])



def save_immediate_alert_record(
    incident_id: str,
    user_id: str,
    risk_level: str,
    alert_status: str,
    urgency_reason: str = "",
    path: str = IMMEDIATE_ALERT_PATH,
) -> None:
    """
    רשומת Immediate Alert מקומית (data/immediate_alert_log.xlsx). עד עכשיו
    immediate_alert_node תיעד את המקרים הכי דחופים (risk_level=high) רק
    בהדפסה לקונסולה - בניגוד ל-review_queue/incident_log שכן נשמרים בקובץ.
    זו רשומת האודיט הקבועה היחידה למקרי החירום, עד שיהיה יעד אמיתי (Open
    WebUI + Polly).
    """
    _append_row(path, IMMEDIATE_ALERT_SHEET_NAME, IMMEDIATE_ALERT_COLUMNS, [
        datetime.now().isoformat(timespec="seconds"),
        incident_id,
        user_id,
        risk_level,
        alert_status,
        urgency_reason,
    ])


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


if __name__ == "__main__":
    save_screened_record(
        text_content="בדיקה: אני בת 17, גרה בבאר שבע.",
        screening_reason="Distress probability 1.00 >= threshold 0.4 (HeBERT)",
        names=["אין שם"],
        addresses=["באר שבע"],
        ages=[17],
    )
    print(f"נשמרה שורת בדיקה ב-{EXCEL_PATH}")
