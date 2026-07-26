import json

import requests

from local_storage import save_token_usage_record

# "127.0.0.1" ולא "localhost": על מכונות עם WSL2 מותקן, wslrelay.exe נצמד גם הוא
# לפורט 11434 על ::1 (IPv6) ומיירט חלק מהחיבורים ל"localhost" עוד לפני שהם מגיעים
# ל-Ollama האמיתי (שנצמד רק ל-127.0.0.1), מה שגורם ל-hang/connection-reset אקראיים.
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

# Ollama זורק מודל מהזיכרון כברירת מחדל אחרי 5 דקות חוסר פעילות (keep_alive
# דיפולטיבי), ואז הקריאה הבאה צריכה לטעון אותו מחדש מהדיסק - זה מה שגרם
# ל-timeout שראינו בפועל. מבקשים כאן שהמודל יישאר טעון הרבה יותר זמן; הטיימר
# הזה מתאפס בכל קריאה, אז כל עוד יש בקשה לפחות פעם ב-30 דקות המודל נשאר חם.
OLLAMA_KEEP_ALIVE = "30m"

# מילות יחס/תיאור נפוצות שה-LLM נוטה להתבלבל איתן ולחלץ כ"שם" בטעות - בעיקר
# כי הן מופיעות ממש לפני מספר, באותה תבנית תחבירית של שם (כמו "אני בת 17").
NON_NAME_WORDS = {
    "בן", "בת", "אני", "אתה", "את", "הוא", "היא", "אנחנו", "אתם", "אתן", "הם", "הן",
    "ילד", "ילדה", "נער", "נערה", "מר", "גברת", "אדון",
}

NO_NAME_PLACEHOLDER = "אין שם"

EXTRACTION_PROMPT = """You are a structured information extraction engine. Read the message below \
(it may be in Hebrew or English) and extract ONLY the following, if explicitly present in the text:
- names: full or partial person names mentioned. Do NOT treat gender/age descriptor words such as \
the Hebrew "בן" or "בת" (as in the common phrase "אני בת 17", meaning "I am a 17-year-old girl") as \
names - those are not names.
- addresses: street addresses, cities, or other location identifiers mentioned
- ages: ages mentioned, as integers
- phone_numbers: phone numbers mentioned, as they appear in the text (Israeli or international format)

Return ONLY a JSON object with exactly these four keys: "names", "addresses", "ages", "phone_numbers".
Each value must be a list (empty list if nothing found). Do not invent or infer information
that is not explicitly stated in the text.

Message:
\"\"\"{text}\"\"\"
"""


class RelevantInfoExtractor:
    """
    שולח את הטקסט למודל Ollama מקומי (llama3.1 כברירת מחדל, רץ על המכונה -
    לא נשלח לענן, כדי לשמור על פרטיות המידע המזהה) ומחלץ ממנו מידע מבני בלבד:
    שמות, כתובות וגילאים. אין כאן ניתוח סמנטי/רגשי - זה נעשה בשלבים אחרים
    (ml_comprehend.py / Bedrock); המטרה היחידה כאן היא NER ממוקד לצורך איתור פרטים
    שיעזרו לצוות האנושי לזהות/לאתר את הפונה.
    """

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_URL, timeout: int = 120):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def warm_up(self) -> None:
        """
        טוענת את המודל לזיכרון מיד (נקראת ב-startup של safesignal.py), כדי
        שהמשתמש הראשון בפועל לא ישלם את מחיר הטעינה הראשונית מהדיסק.
        """
        try:
            requests.post(
                self.base_url,
                json={"model": self.model, "prompt": "", "keep_alive": OLLAMA_KEEP_ALIVE},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            print(f"[Ollama Warm-up Error] {e}")

    def extract(self, text: str, incident_id: str = "") -> dict:
        empty = {"names": [NO_NAME_PLACEHOLDER], "addresses": [], "ages": [], "phone_numbers": []}
        if not text or not text.strip():
            return empty

        try:
            response = requests.post(
                self.base_url,
                json={
                    "model": self.model,
                    "prompt": EXTRACTION_PROMPT.format(text=text),
                    "format": "json",
                    "stream": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            response_body = response.json()
            save_token_usage_record(
                pipeline_stage="info_extraction",
                model_id=self.model,
                sentence=text,
                input_tokens=response_body.get("prompt_eval_count", 0),
                output_tokens=response_body.get("eval_count", 0),
                incident_id=incident_id,
            )
            parsed = json.loads(response_body.get("response", "{}"))
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            print(f"[Ollama Extraction Error] {e}")
            return empty

        names = [n for n in self._as_list(parsed.get("names")) if self._is_real_name(n)]
        return {
            "names": names or [NO_NAME_PLACEHOLDER],
            "addresses": self._as_list(parsed.get("addresses")),
            "ages": [int(a) for a in self._as_list(parsed.get("ages")) if self._is_intable(a)],
            "phone_numbers": self._as_list(parsed.get("phone_numbers")),
        }

    @staticmethod
    def _as_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _is_real_name(value) -> bool:
        return isinstance(value, str) and bool(value.strip()) and value.strip() not in NON_NAME_WORDS

    @staticmethod
    def _is_intable(value) -> bool:
        try:
            int(value)
            return True
        except (TypeError, ValueError):
            return False


# ==========================================
#      אזור בדיקת המודול (Main Simulation)
# ==========================================
if __name__ == "__main__":
    extractor = RelevantInfoExtractor()
    test_posts = [
        "קוראים לי דנה, אני בת 16 וגרה ברחוב הרצל 12 בתל אביב.",
        "אין לי כוח לקום מהמיטה כבר שבוע, לא יודעת מה לעשות.",
        "היי, רציתי לשאול באיזה שעה המשרדים שלכם נפתחים מחר?",
    ]

    print("\n--- מריץ בדיקות חילוץ מידע ---")
    for i, post in enumerate(test_posts, 1):
        result = extractor.extract(post)
        print(f"\n[פוסט בדיקה {i}]")
        print(f"טקסט: {post}")
        print(f"תוצאה: {result}")
        print("-" * 50)
