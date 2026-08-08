# SafeSignal

SafeSignal is a youth distress-detection and triage pipeline. It watches
incoming messages — a live Telegram bot feed, plus Reddit and X (Twitter)
monitored via RSS — screens them for genuine distress signals, classifies
risk level with an LLM, retrieves supporting context from a knowledge base
(RAG), and routes each incident through an autonomous Decision Agent that
decides whether it needs an immediate alert, a human review, or can be
logged and closed — with guardrails and hallucination checking on every
AI-generated output before it reaches a human.

## How it works

```
message in
  ├── Telegram bot (live, long-polled by safesignal.py)
  ├── Reddit  (r/offmychest + r/mentalhealth, polled via RSS every 40 min)
  └── X / Twitter (RSS feed, currently disabled in the n8n workflow)
      │
      ▼
Input Screening  ──── HeBERT (fine-tuned Hebrew distress model) + AWS Comprehend
      │
      ▼
Relevant Info Extraction ── names / addresses / ages / phone numbers (PII, for the human reviewer only)
      │
      ▼
Distress Classification ── AWS Bedrock LLM → class + risk_level + summary
      │
      ▼
RAG Context Retrieval ──── FAISS vector index over a knowledge base of similar cases
      │
      ▼
Decision Agent (LangGraph) ── tool-calling agent (AWS Bedrock / Gemini) that reads
      │                       everything above, calls real tools (alert / rag lookup),
      │                       and produces a final risk assessment + recommended action
      ▼
Output Screening ────────── AWS Bedrock Guardrails + hallucination check on the
      │                     agent's own output before anything reaches a human
      ▼
Routing by Risk Level
      ├── high    → Immediate Alert  (n8n → AWS SES email, optional Twilio call)
      ├── medium  → Human Review     (dashboard)
      └── low     → Log & Close      (S3 incident archive)
```

Every incident, its classification, and the agent's reasoning are persisted
to a local SQLite database and pushed live to a dashboard over WebSocket.

## Architecture

The system has two parts:

1. **The root backend** (`safesignal.py`) — a single FastAPI app that owns the
   SQLite database, the WebSocket layer for the dashboard, the Decision Agent
   (`decision_agent_graph.py`, a LangGraph graph), and a handful of endpoints
   that the microservices below don't yet own independently (audio
   transcription, vision/OCR, incident history, error logging).

2. **Microservices** (`services/`), each an independent FastAPI app in its own
   Docker container:

   | Service              | Port  | Responsibility                                   |
   |-----------------------|-------|---------------------------------------------------|
   | `rag_service`         | 8001  | FAISS-backed retrieval over the knowledge base     |
   | `image_analyser`      | 8002  | Image distress-signal detection (Gemini Vision)    |
   | `guardrails_service`  | 8003  | Input screening (HeBERT + Comprehend)              |
   | `langgraph_agent`     | 8004  | Standalone copy of the Decision Agent graph        |
   | `alert_mcp`           | 8011  | MCP tool server — fires the immediate-alert webhook|
   | `rag_mcp`              | 8012  | MCP tool server — exposes RAG lookup as a tool      |
   | `output_screening`    | 18005 (→8005 in-container) | Bedrock Guardrails + Claude Haiku output check |

   `alert_mcp` and `rag_mcp` are the actual tools the Decision Agent calls at
   runtime (via MCP, the Model Context Protocol) — they're what makes the
   agent's tool calls real rather than simulated.

Orchestration between all of the above — pulling messages in, chaining the
HTTP calls between stages, and branching on the final risk level — is done by
an [n8n](https://n8n.io) workflow running in its own Docker container.

## Tech stack

- **API layer**: FastAPI, Uvicorn, WebSockets
- **AI / ML**: AWS Bedrock (Claude models), Google Gemini, HeBERT
  (Hebrew BERT, fine-tuned for distress classification, via 🤗 Transformers +
  PyTorch), Groq (Whisper transcription)
- **Agent framework**: LangGraph + LangChain, MCP (Model Context Protocol)
  for real tool calling
- **Retrieval**: FAISS vector index
- **Data**: SQLite (via SQLAlchemy async + aiosqlite), AWS S3 (incident
  archive), AWS SES (email alerts), Twilio (optional voice alerts)
- **Orchestration**: n8n
- **Infra**: Docker / Docker Compose

## Project structure

```
safesignal.py              root FastAPI app: DB, WebSocket, endpoints, app lifespan
decision_agent_graph.py    the Decision Agent (LangGraph graph + MCP tool wiring)
ml_comprehend.py           HeBERT + AWS Comprehend input screening
distress_classification.py Bedrock-based distress classifier
info_extraction.py         PII extraction (names/addresses/ages/phones)
rag_retrieval.py           FAISS index build + context retrieval
local_storage.py           S3 incident archive (Excel workbook)
database.py / models.py    SQLAlchemy models + async engine
realtime.py                WebSocket connection manager + broadcast payloads
schemas.py                 Pydantic request/response schemas
telegram_bridge.py         Telegram ingestion bridge (see below)
services/                  the microservices described above, one directory each
training/                  HeBERT training notebook + datasets used to build it
n8n.json                   exported n8n workflow (reference only — the live
                            workflow in your n8n instance is the source of truth)
docker-compose.yml         brings up all services/ containers
requirements.txt           root backend dependencies (see Setup)
PROMPT_ENGINEERING_LOG.md  design rationale + validation for every LLM prompt
                            used in the pipeline (see below)
```

**On `PROMPT_ENGINEERING_LOG.md`:** every prompt in this system controls a
real triage decision about a vulnerable user, so prompt wording isn't
treated as a throwaway implementation detail. This log documents, per
prompt: where it came from, what structural/engineering work (schema
design, error handling, safe defaults) surrounds it beyond the wording
itself, what integration bugs were found and fixed while wiring it up, and
what live validation was run against it. It's written to be read
independently of the code — the record of *why* the prompts are shaped the
way they are, not just *what* they say.

## Prerequisites

What a brand-new machine needs before any of this will run, in full:

- **Docker Desktop**, running. This is the only thing the seven
  microservices actually need — see
  [A note on the microservices](#a-note-on-the-microservices) below for why
  you don't need to install or configure them individually.
- **Python 3.12** (a virtual environment is strongly recommended), for the
  root backend.
- **Node.js 20+**, for the dashboard — a separate repository, see
  "The operator dashboard" under [Running the system](#running-the-system)
  below.
- **Ollama**, running locally with the `llama3.1` model pulled
  (`ollama pull llama3.1`). The live n8n workflow's PII-extraction step
  (`Relevant Information Extraction`) calls a local Ollama instance rather
  than a paid API for that one step — without it, that step fails and no
  names/addresses/ages/phone numbers get extracted (the rest of the
  pipeline still runs). Easiest to also run it in Docker:
  ```bash
  docker run -d --name ollama -p 11434:11434 ollama/ollama
  docker exec ollama ollama pull llama3.1
  ```
- **An AWS account** with access to Bedrock (with the Claude models you
  intend to use enabled — this requires manually requesting model access in
  the Bedrock console the first time), Comprehend, S3, and SES, available on
  the machine via the standard AWS credential chain (e.g.
  `~/.aws/credentials`, set up via `aws configure`) — the containers that
  need AWS mount this directory read-only.
- API keys as needed for the providers you want to use: Google Gemini, Groq,
  Twilio (all optional — each feature degrades gracefully to a stub/skip if
  its key is missing; see `.env.example` for exactly what each one gates).
  Gemini is used more than the others — both the image-ingestion path and
  one step inside the n8n workflow depend on it.
- The fine-tuned HeBERT model weights (`hebert_distress_model/`, ~420 MB —
  not in this repository; see "Model weights" under [Setup](#setup) below).
- n8n itself provisioned with its own credentials (AWS, Gemini, Ollama) set
  up by hand in the n8n UI — n8n never exports credentials into `n8n.json`,
  so this step can't be automated by cloning the repo. See step 1 under
  [Running the system](#running-the-system).

None of the above is machine-specific — anyone with these installed and
their own credentials filled in can clone this repo (and the dashboard repo)
onto a fresh machine and bring the whole system up.

### A note on the microservices

The seven `services/` containers are **not** pre-built images pulled from
somewhere — `docker-compose.yml` points each one at `build: ./services/...`,
so `docker compose up --build` compiles all seven from source, on whatever
machine you run it on, using the `Dockerfile` and `requirements.txt` already
committed in each service's directory. A fresh machine with nothing but
Docker Desktop installed builds the exact same images you'd get anywhere
else — there's no separate "set up the microservices" step, no private
registry, and no dependency on anything that only exists on the original
development machine.

## Setup

1. **Clone this repo, and the dashboard repo alongside it** (as sibling
   directories — the dashboard instructions below assume that layout):
   ```bash
   git clone <this-repo> final_project_safesignal
   git clone <dashboard-repo> safesignal-dashboard
   cd final_project_safesignal
   python -m venv .venv
   ./.venv/Scripts/python.exe -m pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Fill in `.env` with your own keys/credentials. Every variable in
   `.env.example` documents exactly what it gates and what happens if it's
   left empty.

3. **Model weights.** Place the fine-tuned HeBERT model directory at
   `hebert_distress_model/` (containing `config.json`, `model.safetensors`,
   `tokenizer.json`, `tokenizer_config.json`) in the project root. It's
   produced by `training/train_hebert_colab.ipynb` and is deliberately
   excluded from git (`.gitignore`) — at ~420 MB it exceeds GitHub's
   per-file push limit. Download the pre-trained weights from
   `<Google Drive link — TODO>`, or reproduce them yourself by running the
   training notebook. Both the root backend and `guardrails_service` load
   the model from this same local path.

## Running the system

Bring the pieces up in this order:

1. **n8n** (workflow orchestration):
   ```bash
   docker run -d --name n8n --restart unless-stopped -p 5678:5678 n8nio/n8n
   ```
   Then open `http://localhost:5678`, import `n8n.json` (or build the
   workflow from scratch — the exported file is a reference snapshot, not
   guaranteed to match whatever you've since edited in the UI), and activate
   it. Provide n8n's own credentials for AWS Bedrock/SES, Google Gemini, and
   Ollama inside the n8n UI (Settings → Credentials) — these are separate
   from the `.env` file, which only configures the Python services.

2. **The microservices** (rag_service, image_analyser, guardrails_service,
   output_screening, alert_mcp, rag_mcp, langgraph_agent):
   ```bash
   docker compose up -d --build
   ```
   First build takes a few minutes (PyTorch + transformers for
   `guardrails_service` is the slow one). Confirm everything is up:
   ```bash
   docker compose ps
   ```
   All seven services should show `Up`.

3. **The root backend**, only after step 2 is fully up — it connects to
   `alert_mcp`/`rag_mcp` at startup, and falls back to simulated tool calls
   if it can't reach them yet:
   ```bash
   PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe safesignal.py
   ```
   Startup loads the HeBERT model and builds the FAISS index, which takes
   roughly a minute the first time. You should see
   `Uvicorn running on http://127.0.0.1:8000` when it's ready.

4. **(Optional) Telegram ingestion.** `telegram_bridge.py` polls a Telegram
   bot for live messages and forwards them into the pipeline. It can run
   locally or on a separate always-on host; either way it just needs
   `TELEGRAM_BOT_TOKEN` set and network access to wherever your backend is
   reachable. See the comments at the top of `telegram_bridge.py` for setup.

5. **The operator dashboard.** This is a separate Next.js project
   (`safesignal-dashboard`, not part of this repo) that connects to the
   backend from step 3 over WebSocket to show live incidents, history, and
   system status. It has no login — once it's running, opening it in a
   browser is the whole "entry point" into the system:
   ```bash
   cd ../safesignal-dashboard
   npm install   # first time only
   npm run dev
   ```
   Open **`http://localhost:3000`** — that's the click. It reads
   `NEXT_PUBLIC_REALTIME_URL` from `safesignal-dashboard/.env.local`
   (defaults to `ws://localhost:8000/api/v1/realtime`, i.e. the backend from
   step 3) to receive incidents live as they're created; `/history` on the
   same dashboard shows past incidents with filtering/stats. Without a
   backend connection it silently falls back to a simulated demo feed, so if
   the dashboard looks "live" but empty, double check step 3 is actually up.

## Verifying it works

Check every port is listening:

```bash
netstat -ano | findstr "8000 8001 8002 8003 8004 8011 8012 18005 5678"
```

Hit the backend directly with a sample message:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/classify-distress \
  -H "Content-Type: application/json" \
  -d "{\"text_content\": \"some distress text\", \"incident_id\": \"test-1\"}"
```

A healthy response returns `class`, `risk_level`, and a `summary` grounded in
the input text. For the full pipeline (including the Decision Agent's tool
calls and guardrail checks), POST the same shape to
`/api/v1/decision-agent` instead.

Interactive API docs for the root backend are at `http://127.0.0.1:8000/docs`
(and `http://127.0.0.1:<port>/docs` for each microservice except the MCP
tool servers, which speak MCP over `/mcp` rather than REST). To see it
end-to-end as an operator would, open the dashboard at
`http://localhost:3000` (step 5 above) and send a message through Telegram
(or one of the `curl` calls above) — it should appear there live.

## Environment variables

See `.env.example` — every variable is documented there with what it
controls and what happens if it's left unset (each integration degrades
independently rather than failing the whole app).

## Notes

- There is currently no automated test suite; verification is done by
  exercising the endpoints above directly.
- `services/langgraph_agent/agent_graph.py` is a standalone copy of the
  Decision Agent for the microservices architecture; `safesignal.py` itself
  still uses `decision_agent_graph.py` directly. Keep both in sync when
  changing routing/risk logic.
