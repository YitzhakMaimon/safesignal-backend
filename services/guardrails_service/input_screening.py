"""
Input-side screening -- extracted verbatim from ml_comprehend.py (the ONLY
file that had guardrail logic under that name; it contains input screening
only, see main.py's module docstring for the output-side source).
"""
import os

import boto3
import torch
from anthropic import AnthropicBedrock
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

COMPREHEND_SENTIMENT_LANGUAGES = {'en', 'es', 'fr', 'de', 'it', 'pt', 'ar', 'hi', 'ja', 'ko', 'zh', 'zh-TW'}
HEBREW_UNICODE_RANGE = ('֐', '׿')

TRANSLATION_AWS_REGION = os.environ.get("BEDROCK_AWS_REGION", "us-east-1")
TRANSLATION_MODEL_ID = os.environ.get(
    "BEDROCK_TRANSLATION_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)

DISTRESS_THRESHOLD = 0.4

HEBERT_CHUNK_TOKEN_LIMIT = 450
HEBERT_CHUNK_OVERLAP = 50

# Configurable so the model directory doesn't have to live at this service's
# own CWD -- unlike the monolith, this can point at a mounted volume.
HEBERT_MODEL_PATH = os.environ.get("HEBERT_MODEL_PATH", "hebert_distress_model")


def chunk_text_by_tokens(text: str, tokenizer, chunk_size: int = HEBERT_CHUNK_TOKEN_LIMIT,
                          overlap: int = HEBERT_CHUNK_OVERLAP) -> list[str]:
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
        print("=== [1/3] מתחבר ל-Amazon Comprehend (לזיהוי שפה + שפות שאינן עברית) ===")
        self.comprehend = boto3.client('comprehend', region_name=aws_region)

        print("=== [2/3] טוען מודל HeBERT (עשוי לקחת דקה בהרצה הראשונה) ===")
        model_name = HEBERT_MODEL_PATH
        if not os.path.isdir(model_name):
            print(f"[Warning] '{model_name}' לא נמצא - נטען מודל סנטימנט כללי כברירת מחדל (לא מאומן למצוקה)")
            model_name = "avichr/heBERT_sentiment_analysis"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        device = 0 if torch.cuda.is_available() else -1
        self.hf_classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )

        print("=== [3/3] המערכת מוכנה לפעולה! ===")

        trigger_phrases_he = [
            "לסיים עם זה", "לסגור את האור", "אין לי כוח לקום", "שמישהו יכבה",
            "תודה על הכל", "סליחה מכולם", "בקרוב הכל ייגמר", "אין לי אוויר",
            "לא יכול יותר", "נמאס לי מהמשחק", "אני שקוף", "אין טעם",
            "השארתי לכם מכתב", "השארתי מכתב פרידה", "כתבתי מכתב פרידה",
            "הלילה האחרון שלי", "הפעם האחרונה שלי", "היום האחרון שלי"
        ]

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

    def _score_hebrew_distress(self, text: str) -> tuple[float, int]:
        token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
        text_chunks = (
            chunk_text_by_tokens(text, self.tokenizer)
            if token_count > HEBERT_CHUNK_TOKEN_LIMIT
            else [text]
        )

        distress_probability = 0.0
        for chunk in text_chunks:
            hf_scores = self.hf_classifier(chunk, top_k=None, truncation=True, max_length=512)
            scores_by_label = {r['label'].upper(): r['score'] for r in hf_scores}
            chunk_distress_probability = scores_by_label.get('DISTRESS', scores_by_label.get('NEGATIVE', 0.0))
            distress_probability = max(distress_probability, chunk_distress_probability)

        return distress_probability, len(text_chunks)

    def _translate_text(self, text: str, target_language: str) -> str | None:
        try:
            client = AnthropicBedrock(aws_region=TRANSLATION_AWS_REGION)
            response = client.messages.create(
                model=TRANSLATION_MODEL_ID,
                max_tokens=1024,
                system=f"Translate the user's message to {target_language}. "
                       f"Return ONLY the translated text, with no explanation or preamble.",
                messages=[{"role": "user", "content": text}],
            )
            block = next(b for b in response.content if b.type == "text")
            return block.text.strip()
        except Exception as e:
            print(f"[Translation Error] {e}")
            return None

    def _cross_check_via_translation(self, text: str, source_language: str) -> dict | None:
        if source_language == 'he':
            translated = self._translate_text(text, target_language='English')
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

        translated = self._translate_text(text, target_language='Hebrew')
        if not translated:
            return None
        distress_probability, num_chunks = self._score_hebrew_distress(translated)
        hebert_source = "HeBERT" if num_chunks == 1 else f"HeBERT({num_chunks} chunks)"
        return {
            "distress_probability": distress_probability,
            "source": f"{hebert_source}(cross-check)",
            "translated_text": translated,
        }

    def analyze_post(self, text: str) -> dict:
        if not text or not text.strip():
            return {"passed_screening": False, "reason": "Empty input"}

        language = self.detect_language(text)

        if language == 'he':
            distress_probability, num_chunks = self._score_hebrew_distress(text)
            label = 'NEGATIVE' if distress_probability >= 0.5 else 'NEUTRAL'
            score = max(distress_probability, 1 - distress_probability)
            source = "HeBERT" if num_chunks == 1 else f"HeBERT({num_chunks} chunks)"
        elif language in COMPREHEND_SENTIMENT_LANGUAGES:
            try:
                res = self.comprehend.detect_sentiment(Text=text, LanguageCode=language)
                label = res['Sentiment']
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
            return {
                "text": text,
                "passed_screening": True,
                "reason": f"Unsupported language detected ({language}) - routed to human review",
                "raw_metrics": {"detected_language": language}
            }

        found_triggers = [phrase for phrase in self.trigger_phrases if phrase in text]
        has_trigger = len(found_triggers) > 0

        passed_screening = False
        reason = "Normal/Safe Post"

        if distress_probability >= DISTRESS_THRESHOLD:
            passed_screening = True
            reason = f"Distress probability {distress_probability:.2f} >= threshold {DISTRESS_THRESHOLD} ({source})"
        elif has_trigger:
            passed_screening = True
            reason = f"Trigger phrase detected: {found_triggers}"

        cross_check = None
        if not passed_screening and language in ('he', 'en'):
            cross_check = self._cross_check_via_translation(text, source_language=language)
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
