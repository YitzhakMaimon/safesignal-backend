"""
telegram_monitor.py
====================
כלי ניטור עצמאי לערוצי/קבוצות טלגרם, דרך ה-API של "Telegram155" ב-RapidAPI.

מטרת השלב הנוכחי: להוכיח יכולת משיכה - להריץ את הסקריפט ולראות בפלט את 20
ההודעות האחרונות מהערוץ שנבחר. מנגנון הסינון וההתרעה כבר מוכן ומחובר,
כך שבהמשך אפשר פשוט להזין את הפלט הזה למודל הניקוד של הפרויקט (safesignal.py)
במקום (או בנוסף ל) בדיקת מילות המפתח המקומית שכאן.

הרצה חד-פעמית (לבדיקה): python telegram_monitor.py --once
הרצה רציפה (כל 20 דקות): python telegram_monitor.py
"""

import os
import sys
import json
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()  # קורא את .env בשורש הפרויקט (git-ignored) לתוך os.environ


# ============================================================================
# 1. הגדרות (Config)
# ============================================================================

# מפתח ה-API נשלף מתוך .env (משתנה RAPIDAPI_KEY) ולא נכתב בקוד בשום מצב.
# הוסיפו שורה כזו לקובץ .env בשורש הפרויקט:
#   RAPIDAPI_KEY=your-key-here
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "telegram155.p.rapidapi.com")

# ה-Peer ID של הערוץ/קבוצה לניטור. ברירת מחדל: ערוץ Durov (לצורך בדיקה).
# אפשר לדרוס דרך משתנה סביבה TELEGRAM_PEER_ID או ישירות כאן.
PEER_ID = os.getenv("TELEGRAM_PEER_ID", "1006503122")

# כמה הודעות למשוך בכל סריקה (מקסימום מוגדר ע"י ה-API, אנחנו מבקשים 20)
MESSAGE_LIMIT = 20

# תדירות הסריקה במצב לולאה רציפה (בשניות). 20 דקות = 1200 שניות.
CHECK_INTERVAL_SECONDS = 20 * 60

# קובץ שבו נשמרים ה-IDs של הודעות שכבר נסרקו, כדי למנוע כפילויות בין ריצות
SEEN_IDS_FILE = os.path.join(os.path.dirname(__file__), "data", "telegram_seen_ids.json")


# ============================================================================
# 2. רשימת מילות/ביטויי מצוקה (Keyword Engine)
# ============================================================================
# רשימה מורחבת של ביטויים בעברית שעשויים להעיד על מצוקה, דיכאון או אובדנות.
# הרשימה אינה ממצה ואינה תחליף למודל NLP אמיתי (זה בדיוק התפקיד של
# safesignal.py בהמשך) - כאן היא רק שכבת סינון ראשונית וזולה.
DISTRESS_KEYWORDS = [
    # ביטויים ישירים לרצון למות / אובדנות
    "רוצה למות", "רוצה להתאבד", "אני רוצה למות", "מתכנן להתאבד",
    "סוף לחיים", "לשים סוף לחיים", "לא רוצה לחיות יותר", "עדיף שאמות",
    "החיים שלי לא שווים", "לא רוצה להיות יותר", "רוצה להיעלם",
    "רוצה שזה ייגמר", "רוצה לסיים הכל", "לגמור עם הכל", "לגמור עם החיים",

    # חוסר תקווה / חוסר כוח
    "אין לי כוח לחיות", "אין לי יותר כוח", "נמאס לי מהכל", "נמאס לי מהחיים",
    "אין טעם להמשיך", "אין לי סיבה להמשיך", "אין סיכוי שזה ישתפר",
    "לא רואה אור בקצה המנהרה", "הכל אבוד", "אין לי למי לפנות",
    "אף אחד לא יבחין אם אני אעלם", "אף אחד לא יתגעגע אליי",

    # פגיעה עצמית
    "לפגוע בעצמי", "פוגע בעצמי", "חותך את עצמי", "רוצה לחתוך את עצמי",
    "מזיק לעצמי", "הרגל להזיק לעצמי",

    # מצוקה נפשית כללית / דיכאון
    "אני שבור", "אני שבורה", "אני קורס", "אני קורסת", "אני בדיכאון עמוק",
    "לא מסוגל להמשיך ככה", "לא מסוגלת להמשיך ככה", "אני במצוקה קשה",
    "מרגיש חנוק", "מרגישה חנוקה", "כולם עדיף בלעדיי", "בלי לי כולם עדיף",

    # פרידה / סגירת מעגל
    "רוצה להיפרד מכולם", "זו ההודעה האחרונה שלי", "סליחה על הכל, ביי",
    "תשכחו ממני", "אני עוזב את העולם הזה", "אני עוזבת את העולם הזה",
]


# ============================================================================
# 3. ניהול הודעות שכבר נסרקו (Deduplication)
# ============================================================================

def load_seen_ids(peer_id: str) -> set:
    """
    טוען מהקובץ המקומי את סט ה-IDs של הודעות שכבר טופלו בעבר, עבור peer_id נתון.
    חשוב: מזהי הודעות הם מקומיים לכל ערוץ/קבוצה בנפרד (לא ייחודיים גלובלית),
    ולכן הקובץ שומר מיפוי peer_id -> רשימת IDs, ולא סט שטוח אחד לכל הערוצים.
    אחרת מעבר בין ערוצים היה עלול לגרום לדילוג שגוי על הודעות חדשות שבמקרה
    חולקות מספר ID עם ערוץ אחר שכבר נסרק.
    """
    if not os.path.exists(SEEN_IDS_FILE):
        return set()
    try:
        with open(SEEN_IDS_FILE, "r", encoding="utf-8") as f:
            all_seen = json.load(f)
        return set(all_seen.get(str(peer_id), []))
    except (json.JSONDecodeError, OSError):
        # קובץ פגום/ריק - מתחילים מסט ריק במקום לקרוס
        return set()


def save_seen_ids(peer_id: str, seen_ids: set) -> None:
    """שומר את סט ה-IDs של peer_id נתון לקובץ המקומי, לשימוש בריצה הבאה."""
    os.makedirs(os.path.dirname(SEEN_IDS_FILE), exist_ok=True)
    all_seen = {}
    if os.path.exists(SEEN_IDS_FILE):
        try:
            with open(SEEN_IDS_FILE, "r", encoding="utf-8") as f:
                all_seen = json.load(f)
        except (json.JSONDecodeError, OSError):
            all_seen = {}
    all_seen[str(peer_id)] = sorted(seen_ids)
    with open(SEEN_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_seen, f, ensure_ascii=False, indent=2)


# ============================================================================
# 4. משיכת הודעות מה-API
# ============================================================================

def fetch_messages(peer_id: str, limit: int = MESSAGE_LIMIT):
    """
    מושך את ה-limit ההודעות האחרונות מהערוץ/קבוצה peer_id, דרך
    v1/peers/{peer_id}/history (getHistory) של Telegram155.
    מחזיר את רשימת ההודעות (list) או None אם הייתה שגיאה.
    """
    if not RAPIDAPI_KEY:
        print("שגיאה: חסר RAPIDAPI_KEY. הוסיפו אותו לקובץ .env ונסו שוב.")
        return None

    url = f"https://{RAPIDAPI_HOST}/v1/peers/{peer_id}/history"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }
    params = {"limit": limit}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
    except requests.RequestException as e:
        print(f"שגיאת רשת בעת קריאה ל-API: {e}")
        return None

    if response.status_code == 429:
        print("הגעתם למגבלת הבקשות החודשית/הרגעית של RapidAPI (429). מדלגים על הסריקה הזו.")
        return None

    if not response.ok:
        print(f"שגיאה מה-API: {response.status_code} - {response.text[:300]}")
        return None

    try:
        data = response.json()
    except json.JSONDecodeError:
        print("שגיאה: התשובה מה-API לא הייתה JSON תקין.")
        return None

    return data.get("messages", [])


# ============================================================================
# 5. ניתוח טקסט - חיפוש מילות מפתח
# ============================================================================

def find_distress_keywords(text: str) -> list:
    """מחזיר רשימת מילות/ביטויי המצוקה שנמצאו בתוך הטקסט הנתון."""
    if not text:
        return []
    return [kw for kw in DISTRESS_KEYWORDS if kw in text]


# ============================================================================
# 6. התרעה והושטת יד
# ============================================================================

def get_sender_id(message: dict):
    """
    מחלץ את user_id של השולח, אם קיים. שים לב: בערוצי שידור (broadcast
    channels) הודעות מתפרסמות "בשם הערוץ" ולא בשם משתמש ספציפי, ולכן
    from_id הוא בדרך כלל None - זו התנהגות תקינה של ה-API, לא שגיאה.
    """
    from_id = message.get("from_id") or {}
    return from_id.get("user_id")


def get_sender_label(message: dict) -> str:
    """שם/כינוי לתצוגה בלבד: user_id אם יש, אחרת post_author (שם המפרסם בערוץ), אחרת 'לא ידוע'."""
    user_id = get_sender_id(message)
    if user_id:
        return str(user_id)
    return message.get("post_author") or "ערוץ (ללא שולח אישי)"


def build_profile_link(user_id, username=None) -> str:
    """
    בונה קישור קליקבילי לפרופיל המשתמש בטלגרם.
    אם יש username - זה הקישור האמין ביותר (t.me/username).
    אחרת, משתמשים במזהה המספרי (עובד רק בחלק מהלקוחות, אך עדיף על כלום).
    אם אין בכלל user_id (למשל פוסט בערוץ שידור) - אין למי לפנות אישית.
    """
    if username:
        return f"https://t.me/{username}"
    if user_id:
        return f"https://t.me/{user_id}"
    return "אין קישור אישי (זו הודעת ערוץ שידור, לא הודעה של משתמש בודד)"


def print_alert(message: dict, matched_keywords: list) -> None:
    """מדפיס התרעה בולטת לטרמינל, כולל תוכן הפוסט וקישור ישיר למתנדב."""
    msg_id = message.get("id")
    text = message.get("message", "")
    user_id = get_sender_id(message)
    username = (message.get("from_id") or {}).get("username")  # אם קיים בתשובת ה-API
    profile_link = build_profile_link(user_id, username)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 70)
    print("🚨 התרעת מצוקה אפשרית! 🚨")
    print("=" * 70)
    print(f"זמן זיהוי   : {timestamp}")
    print(f"מזהה הודעה  : {msg_id}")
    print(f"מזהה משתמש  : {get_sender_label(message)}")
    print(f"מילות מפתח  : {', '.join(matched_keywords)}")
    print(f"תוכן ההודעה : {text}")
    print(f"קישור לפנייה: {profile_link}")
    print("=" * 70 + "\n")


# ============================================================================
# 7. לולאת עבודה ראשית
# ============================================================================

def print_messages(messages: list) -> None:
    """מדפיס את ההודעות שנמשכו - שכבת נראות לבדיקת יכולת המשיכה."""
    print(f"\nנמשכו {len(messages)} הודעות מערוץ {PEER_ID}:\n")
    for msg in messages:
        msg_id = msg.get("id")
        text = (msg.get("message") or "").replace("\n", " ")
        sender = get_sender_label(msg)
        preview = text[:80] + ("..." if len(text) > 80 else "")
        print(f"  [{msg_id}] {sender}: {preview}")
    print()


def run_scan_cycle(seen_ids: set) -> set:
    """מבצע סריקה אחת: משיכה, הדפסה, סינון להודעות חדשות, וסינון מילות מפתח."""
    messages = fetch_messages(PEER_ID, MESSAGE_LIMIT)
    if messages is None:
        return seen_ids

    print_messages(messages)

    for message in messages:
        msg_id = message.get("id")
        if msg_id is None or msg_id in seen_ids:
            continue  # הודעה שכבר נבדקה - מדלגים, לא מתריעים שוב

        text = message.get("message", "")
        matched = find_distress_keywords(text)
        if matched:
            print_alert(message, matched)

        seen_ids.add(msg_id)

    save_seen_ids(PEER_ID, seen_ids)
    return seen_ids


def main():
    run_once = "--once" in sys.argv

    seen_ids = load_seen_ids(PEER_ID)
    print(f"טוענים {len(seen_ids)} מזהי הודעות שכבר נסרקו בעבר עבור ערוץ {PEER_ID} מ-{SEEN_IDS_FILE}")

    if run_once:
        # מצב בדיקה: סריקה אחת בלבד ויציאה
        run_scan_cycle(seen_ids)
        return

    # מצב רציף: סריקה כל CHECK_INTERVAL_SECONDS שניות, עד עצירה ידנית (Ctrl+C)
    print(f"מתחילים ניטור רציף של ערוץ {PEER_ID}, סריקה כל {CHECK_INTERVAL_SECONDS // 60} דקות...")
    while True:
        seen_ids = run_scan_cycle(seen_ids)
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
