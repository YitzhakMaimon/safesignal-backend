"""
Image Analyser -- standalone FastAPI microservice (Layer 3, Service 2).

Extracted verbatim from safesignal.py's _analyze_image / vision() handler.

IMPORTANT DEVIATION (see Deployment Notes): the guideline spec (Section 4.2)
calls for a PyTorch ResNet-50/EfficientNet-B0 classifier trained to output
{"room_type": ..., "condition_score": 1-5, "confidence": ...}. No such model
exists in this project -- this service still uses Gemini Vision (multimodal
OCR + free-text visual-signal description), exactly as it did inside the
monolith. This is a deliberate, explicit choice (confirmed with the project
owner), not an oversight, but it means the guideline's literal /analyse
endpoint and its structured room_type/condition_score schema are NOT
implemented here -- inventing fake room_type/condition_score numbers on top
of a text description would misrepresent what the service actually does.
Only the existing, real endpoint is exposed below.
"""
import base64
import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Standalone service (no longer imported alongside decision_agent_graph.py,
# which used to load .env as an import side-effect in the monolith) -- loads
# its own env directly. No-op in Docker if no .env is present; real deploys
# pass GOOGLE_API_KEY via `docker run -e`.
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")

# Identical to safesignal.py's PLACEHOLDER_FIELDS -- kept in full (not just
# DetectedText) so this endpoint's response shape is byte-for-byte unchanged
# for n8n, which currently merges this response with earlier-stage fields.
PLACEHOLDER_FIELDS = {
    "text_content": "בדיקה: אני מרגיש מאוד לבד ומיואש בזמן האחרון",
    "summary": "סיכום מדומה: המשתמש מביע תחושות בדידות ויאוש",
    "class": "medium",
    "risk_level": "medium",
    "context": "הקשר מדומה מתוך בסיס הידע: פנה לקו סיוע 105 או ער\"ן 1201",
    "decision": "route_to_human_review",
    "agent_output": "מומלץ לבצע בדיקה אנושית בהקדם (stub)",
    "passed": True,
    "DetectedText": "טקסט מדומה שזוהה מתוך תמונה (stub)",
    "TranscriptionText": "תמלול מדומה מהודעה קולית (stub)",
}
DETECTED_TEXT_FALLBACK = PLACEHOLDER_FIELDS["DetectedText"]

VISION_PROMPT = """את/ה כלי סינון במערכת טריאז' מצוקה (SafeSignal). קיבלת תמונה שהועלתה \
או צולמה על ידי משתמש. שתי מטרות, בסדר הזה:
1. אם יש בתמונה טקסט כתוב (צילום מסך של הודעה, פתק, שיחה וכו') - תעתיק אותו כלשונו, \
במדויק ובשלמותו.
2. בנוסף (או אם אין טקסט), תאר/י בקצרה כל רמז חזותי במצוקה עצמה - סימני פגיעה עצמית, \
מצב מסוכן, סביבה מדאיגה וכו'. אם אין כזה, אין צורך לציין זאת במפורש.

החזר/י תשובה אחת, טקסטואלית וקצרה ככל האפשר, המשלבת את שני הדברים (אם רלוונטיים). \
אל תמציא/י תוכן שאינו נראה בתמונה בפועל."""


def _analyze_image(image_url: str) -> str:
    """Real vision analysis via Gemini (multimodal). Fails *open* to stub text on any error."""
    if not GOOGLE_API_KEY or not image_url:
        return DETECTED_TEXT_FALLBACK

    try:
        image_response = requests.get(image_url, timeout=30)
        image_response.raise_for_status()
        mime_type = image_response.headers.get("Content-Type", "image/jpeg").split(";")[0]
        b64_image = base64.b64encode(image_response.content).decode("utf-8")

        llm = ChatGoogleGenerativeAI(model=GEMINI_VISION_MODEL, google_api_key=GOOGLE_API_KEY)
        message = HumanMessage(content=[
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
        ])
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"[Gemini Vision Error] {e}")
        return DETECTED_TEXT_FALLBACK


app = FastAPI(title="SafeSignal Image Analyser")


@app.post("/api/v1/vision/textract-rekognition")
async def vision(req: Request):
    """Preserved verbatim from safesignal.py's vision() handler (merged() helper inlined)."""
    body = await req.json()
    detected_text = _analyze_image(body.get("image_url", ""))
    return {**PLACEHOLDER_FIELDS, **body, "DetectedText": detected_text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
