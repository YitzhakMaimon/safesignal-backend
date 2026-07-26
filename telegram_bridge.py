"""
telegram_bridge.py
====================
Bridges a Telegram bot's live group messages into the SafeSignal backend.

Runs on the EC2 instance (langgraph-ec2), NOT on the local Windows dev
machine: Amdocs' Zscaler policy blocks api.telegram.org outright (category
"Suspicious URLs") at the machine/agent level, so it's blocked regardless
of which network the dev machine connects through. EC2 isn't on the Amdocs
network, so it's unaffected.

Reaches the real backend (safesignal.py, running on the dev machine) via
the existing SSH reverse tunnel (scripts/ssh-reverse-tunnel-loop.ps1:
EC2's own port 8000 -> back to the dev machine's localhost:8000) -- the
same tunnel decision_agent_graph.py's ALERT_WEBHOOK_URL/REVIEW_QUEUE_URL
already depend on. So BACKEND_BASE_URL below (localhost, from the EC2
side) really reaches the real backend.

Requires the bot's Privacy Mode to be disabled via @BotFather
(/setprivacy -> Disable), or it will only ever see messages that start
with "/" or that mention/reply to it -- not the group's ambient traffic.

Deploy: copy this file + a .env (TELEGRAM_BOT_TOKEN=...) to the EC2
instance and run it (nohup + restart-loop, same pattern as the other
scripts/*-loop.ps1 scripts use for their own long-running processes).
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv isn't installed everywhere this runs (e.g. the EC2
          # deploy target) -- TELEGRAM_BOT_TOKEN is exported directly there.

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "http://localhost:8000")

# How many messages' ingestion chains may run at once. Bounded (not
# unlimited) on purpose: classify-distress and decision-agent both call AWS
# Bedrock, and firing every message in a batch at Bedrock simultaneously
# risks throttling -- which would make things slower and less reliable, not
# faster. Each chain is I/O-bound (waiting on HTTP responses), so a thread
# pool is enough here without rewriting the whole script onto asyncio/httpx.
MAX_CONCURRENT_MESSAGES = int(os.environ.get("MAX_CONCURRENT_MESSAGES", "3"))


def call_backend(path: str, payload: dict) -> dict:
    resp = requests.post(f"{BACKEND_BASE_URL}{path}", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_ingestion_chain(text_content: str, user_id: str, incident_id: str, message_timestamp) -> None:
    """
    Runs one Telegram message through the same chain n8n drives for every
    other ingestion source (website/image/voice) -- raw-log, then screening
    (gate: only continue if it passes, same as n8n's "Passed Input
    Screening?" IF node), then extraction/classification/RAG/decision-agent.
    Calling the real HTTP endpoints (rather than reimplementing the logic
    here) keeps behavior identical to the n8n-driven paths.
    """
    body = {
        "text_content": text_content,
        "user_id": user_id,
        "incident_id": incident_id,
        "platform": "telegram",
        "message_timestamp": message_timestamp,
    }
    body = call_backend("/api/v1/ingest/raw-log", body)
    body = call_backend("/api/v1/screening/input", body)
    if not body.get("passed"):
        return

    body = call_backend("/api/v1/extract-info", body)
    body = call_backend("/api/v1/classify-distress", body)
    body = call_backend("/api/v1/rag-context", body)
    call_backend("/api/v1/decision-agent", body)


def _process_message(text: str, user_id: str, incident_id: str, message_date) -> None:
    """Worker-thread entry point -- run_ingestion_chain() itself has no
    exception handling, so a failure here must be caught locally or it just
    vanishes into the pool (ThreadPoolExecutor only surfaces exceptions via
    Future.result(), which nothing here calls)."""
    try:
        run_ingestion_chain(text, user_id, incident_id, message_date)
    except Exception as e:
        print(f"[telegram-bridge] failed to process message {incident_id}: {e}")


def main():
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN not set -- exiting.")
        return

    print(f"[telegram-bridge] polling started (max {MAX_CONCURRENT_MESSAGES} concurrent messages).")
    offset = None
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_MESSAGES) as pool:
        while True:
            try:
                params = {"timeout": 30, "allowed_updates": ["message"]}
                if offset is not None:
                    params["offset"] = offset
                resp = requests.get(f"{TELEGRAM_API_BASE}/getUpdates", params=params, timeout=40)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"[telegram-bridge] getUpdates failed ({e}); retrying in 5s")
                time.sleep(5)
                continue

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message")
                if not message:
                    continue
                chat = message.get("chat", {})
                if chat.get("type") not in ("group", "supergroup"):
                    continue
                text = message.get("text")
                if not text:
                    continue

                sender = message.get("from") or {}
                user_id = f"telegram:{sender['id']}" if sender.get("id") else "anonymous"
                incident_id = f"telegram-{chat['id']}-{message['message_id']}"

                # Submit and move on -- don't block the polling loop waiting
                # for this message's chain to finish. Offset already
                # advanced above, so the next getUpdates call is unaffected
                # by how long processing takes.
                pool.submit(_process_message, text, user_id, incident_id, message.get("date"))


if __name__ == "__main__":
    main()
