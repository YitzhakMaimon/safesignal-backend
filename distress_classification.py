import json
import os

from anthropic import AnthropicBedrock

# מודל Claude שנקרא דרך Amazon Bedrock (ולא מול ה-API הישיר של Anthropic), כך שהסיווג
# רץ תחת אותה תשתית AWS שכבר משמשת את הפרויקט (Comprehend ב-ml_comprehend.py). ה-"us." בתחילת
# המזהה הוא cross-region inference profile - Bedrock דורש אותו להזמנת מודלים חדשים
# (invoke עם מזהה המודל החשוף בלבד נכשל עם "on-demand throughput isn't supported").
# claude-opus-4-8/4-7 ו-claude-sonnet-5 עדיין לא זמינים לחשבון ה-AWS הנוכחי (403) -
# ברגע שתתבקש/י גישה אליהם בקונסולת Bedrock (Model access), אפשר לעדכן כאן.
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-opus-4-6-v1")
BEDROCK_AWS_REGION = os.environ.get("BEDROCK_AWS_REGION", "us-east-1")

# מיפוי קטגוריה -> רמת סיכון. נשמר כטבלת בקרה נפרדת (ולא מוחזר ישירות מהמודל) כדי
# שרמת הסיכון תמיד תהיה עקבית עם הקטגוריה, ולא תלויה בכך שה-LLM יבחר את שתיהן בהתאמה.
CLASS_TO_RISK = {
    "התאבדות/ארוע חירום": "high",
    "מצוקה רגשית": "medium",
    "חרם": "medium",
    "רגיל": "low",
}

# הטקסט שמגיע לכאן כבר עבר את שלב הסינון הראשוני (ml_comprehend.py: Comprehend/HeBERT) שסימן
# אותו כחשוד במצוקה - זהו שלב סיווג משני ומדויק יותר, לא הסינון הראשוני עצמו. קטגוריית
# "רגיל" נשמרת בכוונה כרשת ביטחון עבור מקרים שהסינון הראשוני סימן כ-false positive.
CLASSIFICATION_SYSTEM_PROMPT = """את/ה מסווג/ת מצוקה עבור מוקד תמיכה המטפל בפניות של בני נוער.
ההודעה שלפניך כבר עברה שלב סינון ראשוני אוטומטי שסימן אותה כחשודה כפנייה במצוקה -
תפקידך כעת הוא לתת סיווג מדויק יותר, ולא לחזור על הסינון הראשוני.

סווג/י את ההודעה (בעברית או באנגלית) לאחת מהקטגוריות הבאות בדיוק:

- "התאבדות/ארוע חירום": סימנים ברורים לכוונה אובדנית או לסכנת חיים מיידית (למשל תכנון
  לפגוע בעצמו/ה, פרידה/מכתב פרידה, "לסיים עם הכל"). להשתמש בקטגוריה זו רק כשיש רמז
  מפורש בטקסט - לא להסיק מעבר למה שכתוב.
- "מצוקה רגשית": ביטויי בדידות, ייאוש, עצב, חוסר אונים או מצוקה נפשית שאינם מגיעים
  לרמת סיכון מיידי לחיים.
- "חרם": הדרה חברתית, נידוי, בריונות חברתית ("כולם מתעלמים ממני", "מחרימים אותי").
- "רגיל": לאחר קריאה מדוקדקת, אין בטקסט בפועל סימני מצוקה משמעותיים (כלומר הסינון
  הראשוני היה כנראה false positive - למשל סרקזם, ביטוי סלנג לא מילולי, או הקשר תמים).

החזר/י אך ורק את קטגוריית המצוקה המתאימה ביותר (אחת מארבע הקטגוריות בדיוק, כפי שנכתבו
לעיל) ומשפט קצר אחד בעברית המסביר את הסיווג. אין להסיק מידע שלא נאמר במפורש בטקסט.
"""

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "class": {"type": "string", "enum": list(CLASS_TO_RISK)},
        "summary": {"type": "string", "description": "משפט אחד בעברית המסביר את הסיווג"},
    },
    "required": ["class", "summary"],
    "additionalProperties": False,
}


class BedrockDistressClassifier:
    """
    מסווג את רמת הדחיפות/המצוקה של פנייה שעברה את שלב הסינון הראשוני (ml_comprehend.py),
    באמצעות מודל Claude הנקרא דרך Amazon Bedrock.
    """

    def __init__(self, model: str = BEDROCK_MODEL_ID, aws_region: str = BEDROCK_AWS_REGION):
        self.model = model
        self.client = AnthropicBedrock(aws_region=aws_region)

    def classify(self, text: str) -> dict:
        if not text or not text.strip():
            return {"class": "רגיל", "risk_level": "low", "summary": "אין טקסט לניתוח"}

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=CLASSIFICATION_SYSTEM_PROMPT,
                output_config={
                    "format": {"type": "json_schema", "schema": CLASSIFICATION_SCHEMA}
                },
                messages=[{"role": "user", "content": text}],
            )
            block = next(b for b in response.content if b.type == "text")
            parsed = json.loads(block.text)
            label, summary = parsed["class"], parsed["summary"]
        except Exception as e:
            # כברירת מחדל בטוחה (בדומה ל-ml_comprehend.py בכשל Comprehend): מנתבים לבדיקה אנושית
            # ברמת "medium" ולא משמיטים את הפנייה, גם אם קריאת ה-LLM נכשלה.
            print(f"[Bedrock Classification Error] {e}")
            label = "מצוקה רגשית"
            summary = f"שגיאת סיווג מול Bedrock ({e}) - מנותב לבדיקה אנושית כברירת מחדל בטוחה"

        return {
            "class": label,
            "risk_level": CLASS_TO_RISK.get(label, "medium"),
            "summary": summary,
        }


# ==========================================
#      אזור בדיקת המודול (Main Simulation)
# ==========================================
if __name__ == "__main__":
    classifier = BedrockDistressClassifier()
    test_posts = [
        "אני מרגיש שאני רוצה לסיים עם הכל, כתבתי כבר מכתב פרידה.",
        "אני מרגיש מאוד לבד ומיואש בזמן האחרון.",
        "כולם בכיתה מתעלמים ממני ולא מדברים איתי בכלל.",
        "חחחחח סרט פשוט פצצה, הלכתי למות מצחוק!!",
    ]

    print("\n--- מריץ בדיקות סיווג מצוקה מול Bedrock ---")
    for i, post in enumerate(test_posts, 1):
        result = classifier.classify(post)
        print(f"\n[פוסט בדיקה {i}] {post}")
        print(f"תוצאה: {result}")
        print("-" * 50)
