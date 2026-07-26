import os
import boto3
import torch
from anthropic import AnthropicBedrock
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from local_storage import save_token_usage_record

# השפות שבהן Amazon Comprehend תומך עבור detect_sentiment (עברית אינה ביניהן)
COMPREHEND_SENTIMENT_LANGUAGES = {'en', 'es', 'fr', 'de', 'it', 'pt', 'ar', 'hi', 'ja', 'ko', 'zh', 'zh-TW'}
HEBREW_UNICODE_RANGE = ('֐', '׿')

# מודל התרגום לבדיקה חוצת-שפות (cross-lingual cross-check, ראו _cross_check_via_translation) -
# אותו Claude Haiku על Bedrock שכבר משמש את בדיקת ההזיות ב-decision_agent_graph.py, כדי לא
# להוסיף ספק/הרשאת AWS חדשה (Translate לא מורשה למשתמש הזה) ולא לצרוך את מכסת Gemini החינמית.
TRANSLATION_AWS_REGION = os.environ.get("BEDROCK_AWS_REGION", "us-east-1")
TRANSLATION_MODEL_ID = os.environ.get(
    "BEDROCK_TRANSLATION_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)

# סף ההסתברות למצוקה שמעליו הפוסט עובר הלאה (ל-LLM/בדיקה נוספת).
# נמוך במתכוון (לא 0.5): זהו שלב סינון ראשוני שמזין קריאת LLM בהמשך, לא החלטה סופית -
# false negative כאן (פספוס מצוקה אמיתית) יקר בהרבה מ-false positive (קריאת LLM מיותרת).
# לכייל מחדש כשיהיה דאטא אמיתי מהשטח.
DISTRESS_THRESHOLD = 0.4

# BERT-family models (HeBERT included) cap out at 512 tokens per input. Texts under
# this are analyzed in a single pass; longer ones (e.g. a scraped news homepage) are
# split into overlapping windows -- see chunk_text_by_tokens -- and scored chunk by
# chunk, so a distressing sentence buried deep in a long page isn't lost to truncation.
HEBERT_CHUNK_TOKEN_LIMIT = 450
HEBERT_CHUNK_OVERLAP = 50


def chunk_text_by_tokens(text: str, tokenizer, chunk_size: int = HEBERT_CHUNK_TOKEN_LIMIT,
                          overlap: int = HEBERT_CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping windows of exactly `chunk_size` real model tokens
    (not an LLM's guess at token count) using the model's own tokenizer, so each
    chunk safely fits under HeBERT's 512-token limit once re-tokenized with special
    tokens. The overlap means a sentence spanning a chunk boundary still appears
    whole in at least one chunk.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        if end >= len(tokens):
            break
        start += chunk_size - overlap

    return chunks


class DistressScreeningPipeline:
    def __init__(self, aws_region='eu-west-1'):
        """
        אתחול הצינור: חיבור ל-Comprehend (לזיהוי שפה + ניתוח שפות שאינן עברית)
        וטעינת HeBERT (לניתוח עברית, ש-Comprehend לא תומך בה בכלל).
        """
        print("=== [1/3] מתחבר ל-Amazon Comprehend (לזיהוי שפה + שפות שאינן עברית) ===")
        self.comprehend = boto3.client('comprehend', region_name=aws_region)

        print("=== [2/3] טוען מודל HeBERT (עשוי לקחת דקה בהרצה הראשונה) ===")
        # המודל המאומן (fine-tuned) שלנו לזיהוי מצוקה, לא מודל הסנטימנט הכללי מ-Hugging Face.
        # יש לחלץ את תיקיית המודל שהורדת מקולאב (hebert_distress_model.zip) לכאן, לתיקיית הפרויקט,
        # כך שהנתיב היחסי הבא יצביע עליה. אם התיקייה לא קיימת, נופלים חזרה למודל הסנטימנט המקורי.
        model_name = "hebert_distress_model"
        if not os.path.isdir(model_name):
            print(f"[Warning] '{model_name}' לא נמצא - נטען מודל סנטימנט כללי כברירת מחדל (לא מאומן למצוקה)")
            model_name = "avichr/heBERT_sentiment_analysis"

        # טעינת הטוקנייזר והמודל
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # הגדרת ה-Pipeline של Hugging Face (משתמש ב-GPU אם זמין)
        device = 0 if torch.cuda.is_available() else -1
        self.hf_classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )

        print("=== [3/3] המערכת מוכנה לפעולה! ===")

        # רשימת מילות מפתח מורחבת כרשת ביטחון לרמזים שקטים בעברית
        trigger_phrases_he = [
            "לסיים עם זה", "לסגור את האור", "אין לי כוח לקום", "שמישהו יכבה",
            "תודה על הכל", "סליחה מכולם", "בקרוב הכל ייגמר", "אין לי אוויר",
            "לא יכול יותר", "נמאס לי מהמשחק", "אני שקוף", "אין טעם",
            "השארתי לכם מכתב", "השארתי מכתב פרידה", "כתבתי מכתב פרידה",
            "הלילה האחרון שלי", "הפעם האחרונה שלי", "היום האחרון שלי"
        ]

        # אותה רשת ביטחון עבור אנגלית - נחוצה כי Comprehend מבוסס סנטימנט מילולי-שטחי
        # ולא בהכרח מזהה הצהרות סיכון שנאמרות בטון רגוע/עובדתי (למשל "I already have a
        # plan" נסווג בפועל כ-POSITIVE ע"י Comprehend, כי אין בו מילים "שליליות" קלאסיות).
        trigger_phrases_en = [
            "end my life", "ending my life", "kill myself", "want to die",
            "don't want to be here anymore", "no reason to keep going",
            "can't do this anymore", "can't take it anymore", "give up on life",
            "no point in living", "want to disappear", "better off without me",
            "thanks for everything", "sorry for everything", "said my goodbyes",
            "wrote a goodbye letter", "left a note",
        ]

        self.trigger_phrases = trigger_phrases_he + trigger_phrases_en

    def detect_language(self, text: str) -> str:
        """
        מזהה את שפת הטקסט. עברית מזוהה מקומית לפי טווח יוניקוד (מהיר, בלי קריאת API),
        אחרת נשלח ל-Comprehend detect_dominant_language (תומך במאות שפות לזיהוי,
        גם אם לא כולן נתמכות בפועל ל-detect_sentiment).
        """
        lo, hi = HEBREW_UNICODE_RANGE
        hebrew_chars = sum(1 for c in text if lo <= c <= hi)
        letter_chars = sum(1 for c in text if c.isalpha())
        if letter_chars and hebrew_chars / letter_chars > 0.3:
            return 'he'

        try:
            res = self.comprehend.detect_dominant_language(Text=text)
            languages = res.get('Languages', [])
            if languages:
                return languages[0]['LanguageCode']
        except Exception as e:
            print(f"[Language Detection Error] {e}")
        return 'unknown'

    def _score_hebrew_distress(self, text: str, incident_id: str = "") -> tuple[float, int]:
        """
        מריץ HeBERT על טקסט עברי ומחזיר (distress_probability, מספר ה-chunks).
        טקסטים ארוכים (מעל ~500 טוקנים אמיתיים) מפוצלים ל-chunks חופפים כדי שלא
        לחרוג מהמגבלה הקשיחה של BERT (ראו chunk_text_by_tokens); צבירה לפי MAX -
        אם ולו chunk אחד מציג מצוקה, זה גובר. משותף בין הבדיקה העברית הראשית
        ב-analyze_post לבין הבדיקה חוצת-השפות (_cross_check_via_translation).
        """
        token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
        text_chunks = (
            chunk_text_by_tokens(text, self.tokenizer)
            if token_count > HEBERT_CHUNK_TOKEN_LIMIT
            else [text]
        )

        # top_k=None מחזיר את ההסתברות של שתי המחלקות (לא רק את המנצחת),
        # כדי שנוכל להשוות את הסתברות "המצוקה" מול סף רציף ולא רק החלטת argmax בינארית.
        distress_probability = 0.0
        for chunk in text_chunks:
            hf_scores = self.hf_classifier(chunk, top_k=None, truncation=True, max_length=512)
            scores_by_label = {r['label'].upper(): r['score'] for r in hf_scores}
            # תומך גם בתיוג DISTRESS וגם בתיוג NEGATIVE (תלוי איך המודל המאומן נשמר)
            chunk_distress_probability = scores_by_label.get('DISTRESS', scores_by_label.get('NEGATIVE', 0.0))
            distress_probability = max(distress_probability, chunk_distress_probability)

        # HeBERT רץ מקומית (אין חיוב API), אבל עדיין "צריכת טוקנים" אמיתית של מודל -
        # נספר כאן לפי הטוקנייזר של המודל עצמו, לא ניחוש. אין טוקני פלט (סיווג, לא ייצור טקסט).
        save_token_usage_record(
            pipeline_stage="ml_comprehend_hebert_screening",
            model_id="HeBERT (local)",
            sentence=text,
            input_tokens=token_count,
            output_tokens=0,
            incident_id=incident_id,
        )

        return distress_probability, len(text_chunks)

    def _translate_text(self, text: str, target_language: str, incident_id: str = "") -> str | None:
        """
        מתרגם טקסט דרך Claude Haiku על Bedrock (אותו מודל/אזור שכבר מאומתים
        לעבודה בבדיקת ההזיות של decision_agent_graph.py). Fails safe: בכל
        שגיאה מחזיר None כדי שהקורא ידלג על הבדיקה חוצת-השפות בלי להפיל את
        תוצאת הסינון הראשית.
        """
        try:
            client = AnthropicBedrock(aws_region=TRANSLATION_AWS_REGION)
            response = client.messages.create(
                model=TRANSLATION_MODEL_ID,
                max_tokens=1024,
                system=f"Translate the user's message to {target_language}. "
                       f"Return ONLY the translated text, with no explanation or preamble.",
                messages=[{"role": "user", "content": text}],
            )
            save_token_usage_record(
                pipeline_stage="ml_comprehend_translation_crosscheck",
                model_id=TRANSLATION_MODEL_ID,
                sentence=text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                incident_id=incident_id,
            )
            block = next(b for b in response.content if b.type == "text")
            return block.text.strip()
        except Exception as e:
            print(f"[Translation Error] {e}")
            return None

    def _cross_check_via_translation(self, text: str, source_language: str, incident_id: str = "") -> dict | None:
        """
        בדיקה חוצת-שפות (cross-lingual second opinion) לפוסט שלא עבר את הסף
        בבדיקה הראשונית: מתרגם אותו לשפה השנייה הנתמכת ומריץ שם סינון נוסף,
        כדי לתפוס ניסוחי מצוקה (מטאפוריים, ניבים, פערי דאטא-אימון) שמודל/שפה
        אחד פספס אבל השני אולי לא. נקרא רק כשהתוצאה הראשית עדיין "לא עבר"
        (ראו analyze_post) - לעולם לא דורס פוסט שכבר הוסלם.
        """
        if source_language == 'he':
            translated = self._translate_text(text, target_language='English', incident_id=incident_id)
            if not translated:
                return None
            try:
                res = self.comprehend.detect_sentiment(Text=translated, LanguageCode='en')
                distress_probability = res['SentimentScore'].get('Negative', 0.0)
            except Exception as e:
                print(f"[Cross-check Comprehend Error] {e}")
                return None
            return {
                "distress_probability": distress_probability,
                "source": "Comprehend(en, cross-check)",
                "translated_text": translated,
            }

        # source_language == 'en'
        translated = self._translate_text(text, target_language='Hebrew', incident_id=incident_id)
        if not translated:
            return None
        distress_probability, num_chunks = self._score_hebrew_distress(translated, incident_id=incident_id)
        hebert_source = "HeBERT" if num_chunks == 1 else f"HeBERT({num_chunks} chunks)"
        return {
            "distress_probability": distress_probability,
            "source": f"{hebert_source}(cross-check)",
            "translated_text": translated,
        }

    def analyze_post(self, text: str, incident_id: str = "") -> dict:
        """
        מריץ את תהליך הסינון על הטקסט ומחליט האם להעבירו להמשך הצינור (True/False).
        מנתב לפי שפה: עברית -> HeBERT מקומי, שפה נתמכת אחרת -> Comprehend,
        שפה לא נתמכת -> מנותב תמיד ל-Human Review (לא ניתן לסמוך על ניחוש).
        """
        if not text or not text.strip():
            return {"passed_screening": False, "reason": "Empty input"}

        language = self.detect_language(text)

        if language == 'he':
            distress_probability, num_chunks = self._score_hebrew_distress(text, incident_id=incident_id)
            label = 'NEGATIVE' if distress_probability >= 0.5 else 'NEUTRAL'
            score = max(distress_probability, 1 - distress_probability)
            source = "HeBERT" if num_chunks == 1 else f"HeBERT({num_chunks} chunks)"
        elif language in COMPREHEND_SENTIMENT_LANGUAGES:
            try:
                res = self.comprehend.detect_sentiment(Text=text, LanguageCode=language)
                label = res['Sentiment']  # POSITIVE, NEGATIVE, NEUTRAL, MIXED
                score = res['SentimentScore'].get(label.capitalize(), 0.0)
                distress_probability = res['SentimentScore'].get('Negative', 0.0)
                source = f"Comprehend({language})"
            except Exception as e:
                print(f"[Comprehend Error] {e}. מנתב ל-Human Review כברירת מחדל בטוחה.")
                return {
                    "text": text,
                    "passed_screening": True,
                    "reason": f"Comprehend call failed ({e}) - routed to human review as a safe default",
                    "raw_metrics": {"detected_language": language}
                }
        else:
            # שפה לא נתמכת לא ע"י HeBERT ולא ע"י Comprehend - אי אפשר לסמוך על ניחוש,
            # מנתבים תמיד להמשך הצינור לבדיקה אנושית
            return {
                "text": text,
                "passed_screening": True,
                "reason": f"Unsupported language detected ({language}) - routed to human review",
                "raw_metrics": {"detected_language": language}
            }

        # בדיקת רמזים שקטים (Trigger Words) - הרשימה כרגע בעברית בלבד
        found_triggers = [phrase for phrase in self.trigger_phrases if phrase in text]
        has_trigger = len(found_triggers) > 0

        # מנוע החלטה - סף על הסתברות רציפה, לא רק על החלטת argmax בינארית.
        # זה שלב סינון (שמזין קריאת LLM בהמשך), לא החלטה סופית - לכן מוטה בכוונה
        # לכיוון recall: עדיף קריאת LLM מיותרת (false positive) מאשר לפספס מצוקה אמיתית.
        passed_screening = False
        reason = "Normal/Safe Post"

        if distress_probability >= DISTRESS_THRESHOLD:
            passed_screening = True
            reason = f"Distress probability {distress_probability:.2f} >= threshold {DISTRESS_THRESHOLD} ({source})"
        elif has_trigger:
            passed_screening = True
            reason = f"Trigger phrase detected: {found_triggers}"

        # לא עבר את הסף באף אחד מהמנגנונים הראשוניים (עברית<->אנגלית בלבד -
        # אלו שני המודלים שיש לנו) - בדיקה נוספת בשפה השנייה לפני שקובעים
        # סופית שהפוסט תקין (recall-first, ראו הערה למעלה).
        cross_check = None
        if not passed_screening and language in ('he', 'en'):
            cross_check = self._cross_check_via_translation(text, source_language=language, incident_id=incident_id)
            if cross_check and cross_check["distress_probability"] >= DISTRESS_THRESHOLD:
                passed_screening = True
                reason = (
                    f"Cross-lingual catch: original ({source}) scored {distress_probability:.2f} "
                    f"below threshold, but translated text scored "
                    f"{cross_check['distress_probability']:.2f} via {cross_check['source']}"
                )

        raw_metrics = {
            "detected_language": language,
            "label": label,
            "confidence": round(score, 4),
            "distress_probability": round(distress_probability, 4),
            "source": source,
            "triggers_found": found_triggers,
        }
        if cross_check:
            raw_metrics["cross_check"] = cross_check

        return {
            "text": text,
            "passed_screening": passed_screening,
            "reason": reason,
            "raw_metrics": raw_metrics,
        }

# ==========================================
#      אזור בדיקת המערכת (Main Simulation)
# ==========================================
if __name__ == "__main__":
    # יצירת מופע של הצינור
    pipeline_engine = DistressScreeningPipeline()

    # רשימת פוסטים מגוונים לבדיקת הלוגיקה
    test_posts = [
        "איזה יום מדהים, פשוט בא לי לסיים עם הכל וזהו.", # ציניות + רמז (Comprehend עלול להתבלבל מ'מדהים')
        "תודה לכל מי שהיה חלק מהחיים שלי, סליחה אם פגעתי במישהו.", # רמז שקט / פרידה
        "אני מרגיש ריקנות נוראית, אין לי כוח לקום יותר מהמיטה.", # מצוקה קלאסית
        "היי, רציתי לשאול באיזה שעה המשרדים שלכם נפתחים מחר?", # פוסט רגיל לחלוטין (אמור להידחות בסינון)
        "חחחחח סרט פשוט פצצה, הלכתי למות מצחוק!!" # סלנג חיובי (לא אמור לעבור כמצוקה)
    ]

    print("\n--- מריץ בדיקות על פוסטים מהשטח ---")
    for i, post in enumerate(test_posts, 1):
        result = pipeline_engine.analyze_post(post)
        
        print(f"\n[פוסט בדיקה {i}]")
        print(f"טקסט: {result['text']}")
        print(f"האם להעביר לשלב הבא (Passed)? -> ** {result['passed_screening']} **")
        print(f"סיבה: {result['reason']}")
        print(f"נתונים גולמיים: {result['raw_metrics']}")
        print("-" * 50)
