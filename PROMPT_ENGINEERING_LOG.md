# SafeSignal — Prompt Engineering Log (n8n Native AI Node Migration)

## Scope and honesty note

This document covers only the two prompt surfaces touched during the **n8n native-node migration session** (replacing `HTTP Request` nodes with native `Information Extractor` and `AI Agent` nodes). It intentionally does **not** follow a "Version 1 baseline → 5 iterations" narrative for either surface, because that is not what happened here, and inventing one would misrepresent the work.

Both prompts were **ported verbatim from an already-designed, already-in-production Python implementation** (`info_extraction.py`, `distress_classification.py`) into their new n8n-native homes. No prompt wording was iterated on during this session — the wording was already fixed before this work began. What *did* happen, and what this log documents instead, is:

- the **structural adaptation** required to move a prompt from a raw HTTP call into an n8n-native schema-constrained node,
- the **design decisions deliberately preserved** (not reinvented) from the original implementation, and
- the **real bugs found and fixed** during integration testing — which were architecture/data-flow bugs in JavaScript Code nodes, not prompt defects.

Where the requested log format (Section 6, "Version 1–5") does not apply, this document says so explicitly rather than fabricating iterations.

---

## Surface 1 — n8n Information Extractor Node (Local Ollama / Llama 3.1)

**Goal:** Structured PII/NER metadata extraction — `names`, `addresses`, `ages`, `phone_numbers` — from a message that has already passed initial distress screening.

### Provenance

The system prompt is an unmodified port of `EXTRACTION_PROMPT` from `info_extraction.py` (`RelevantInfoExtractor` class), which was already running in production via a direct Ollama HTTP call (`/api/generate`, model `llama3.1`, `format: "json"`). This session did not design new prompt wording; it re-hosted the existing prompt inside n8n's native `Information Extractor` node, backed by an `Ollama Chat Model` sub-node pointed at the same local model.

### Iteration history: none (by design)

There is no "Version 1 → Version 5" wording history for this surface. The prompt text is identical before and after migration. **Iteration did not happen because it was not needed** — the prompt was already validated in production before this session began.

### What actually required engineering work (structural, not prompt wording)

1. **Schema type gap.** n8n's Information Extractor offers two schema modes: `fromAttributes` (a simple UI builder) and `manual` (raw JSON Schema). Inspection of the node's compiled source (`InformationExtractor.node.js`) showed `fromAttributes` only supports scalar types (`Boolean`, `Date`, `Number`, `String`) — **no array type**. Since all four target fields are arrays, `schemaType: "manual"` with an explicit JSON Schema was required instead of the simpler builder.
2. **Business-rule enforcement moved from Python to a downstream Code node, not into the prompt.** The original Python implementation applies two rules *after* the LLM call, in code, not via prompt instruction alone:
   - filtering out Hebrew gender/age descriptor words ("בן", "בת", etc.) that the model tends to mis-extract as names, and
   - substituting the placeholder `"אין שם"` when no real name is found.

   These rules were re-implemented **identically, in a new `Restore Extraction Context` Code node**, rather than trusting the LLM to honor them purely through prompt phrasing. The system prompt does include an instruction not to treat these words as names (inherited from the original), but the *enforcement* is deterministic code, matching the original design's safety posture.

### Failure discovered during integration testing (data-flow, not prompt)

Live testing surfaced a real defect: the `Restore Extraction Context` Code node, written in this same session, used `$json`/`.item` (an n8n Code-node idiom that only addresses the first item in a batch) and returned a hardcoded single-item array. In a batch containing multiple extracted messages, this silently dropped every item after the first. This is **not a prompt defect** — the LLM's per-item output was correct; the surrounding merge code was not. Fixed by iterating with `$input.all()` and correlating each output to its source item by index. Verified with a standalone Node.js unit test (3 items in → 3 items out, correct per-item field values) before deployment, then reconfirmed on live data.

### Final production prompt (unchanged from `info_extraction.py`)

```
You are a structured information extraction engine. Read the message below (it may be in Hebrew or English) and extract ONLY the following, if explicitly present in the text:
- names: full or partial person names mentioned. Do NOT treat gender/age descriptor words such as the Hebrew "בן" or "בת" (as in the common phrase "אני בת 17", meaning "I am a 17-year-old girl") as names - those are not names.
- addresses: street addresses, cities, or other location identifiers mentioned
- ages: ages mentioned, as integers
- phone_numbers: phone numbers mentioned, as they appear in the text (Israeli or international format)

Return ONLY a JSON object with exactly these four keys: "names", "addresses", "ages", "phone_numbers".
Each value must be a list (empty list if nothing found). Do not invent or infer information that is not explicitly stated in the text.
```

n8n-side wrapper (`options.systemPromptTemplate`) adapts this to the node's own format-instruction mechanism (the node appends its own `{format_instructions}` derived from the attached JSON Schema, so the explicit "Return ONLY a JSON object..." instruction is redundant but harmless in the n8n version).

### Validation ("pass rate" — live, not synthetic)

One live end-to-end test via the website ingestion webhook, deliberately including a name, address, age, and phone number:

| Field | Input | Extracted | Correct? |
|---|---|---|---|
| names | "דנה כהן" | `["דנה", "כהן"]` | ✅ (split into two tokens — cosmetic difference, not an error) |
| addresses | "רחוב הרצל 12, תל אביב" | `["רחוב הרצל 12", "תל אביב"]` | ✅ |
| ages | "17" | `[17]` | ✅ |
| phone_numbers | "0501234567" | `["0501234567"]` | ✅ |

1/1 live test correct on all four fields (with one cosmetic name-tokenization difference noted). This is not a statistically meaningful "pass rate" — it is a single confirmatory smoke test of a prompt that was already validated in production prior to migration.

---

## Surface 2 — n8n AI Agent Node (AWS Bedrock, Claude via cross-region inference profile)

**Goal:** Secondary distress triage — assign one of four categories and derive a risk level, for messages that already passed initial screening.

### Provenance

The system prompt is an unmodified port of `CLASSIFICATION_SYSTEM_PROMPT` from `distress_classification.py` (`BedrockDistressClassifier` class), already in production via a direct `AnthropicBedrock` client call with native structured-output enforcement (`output_config.format.json_schema`). As with Surface 1, no prompt wording was authored or iterated in this session.

### Iteration history: none (by design)

Same as Surface 1 — this is a verbatim port. No "Version 1 → 5" wording history exists for this surface either.

### What actually required engineering work (structural, not prompt wording)

1. **`risk_level` is deliberately never asked of the model.** This is the single most important design decision in this surface, and it was explicitly preserved rather than re-derived: the original code maps category → risk level through a fixed lookup table (`CLASS_TO_RISK`) *after* classification, specifically so the risk level can never be inconsistent with the category (i.e., the model cannot pick a "high risk" category and independently assign "low" risk, or vice versa). The n8n JSON Schema given to the `Structured Output Parser` only requests `class` and `summary` — never `risk_level` — and the downstream Code node applies the same fixed mapping table in JavaScript.
2. **Node type selection.** Two native n8n node types could plausibly host this prompt: `Information Extractor` (used for Surface 1) and `AI Agent` (used here, per explicit requirement). Both support Bedrock as a language model and both can be constrained to structured JSON output — `Information Extractor` via its own built-in schema field, `AI Agent` via an attached `Structured Output Parser` sub-node (`hasOutputParser: true`, connected over the `ai_outputParser` connection type). Both were implemented and live-tested in this session; the current production node is the `AI Agent` variant.
3. **Safe-default error handling preserved.** The original Python code catches any Bedrock call failure and falls back to `"מצוקה רגשית"` (medium risk) rather than dropping the message — a deliberate "never silently drop a flagged message" safety behavior. This was replicated in the n8n version via `onError: "continueRegularOutput"` on the Agent node plus an explicit fallback branch in the downstream Code node that checks for `item.json.error` and applies the identical fallback category and an equivalent explanatory summary.

### Failure discovered during integration testing (data-flow, not prompt — and not in this node)

This is the finding worth being precise about, since an earlier draft of this request described it as belonging to this surface's prompt. **It does not.** The actual defect: `Distress Classification (Bedrock)` node (this surface) itself behaved correctly and produced the right classification for every input tested. The state-loss bug was located in **three separate downstream Code nodes** — `Output Screening (Comprehend + ML) - Result`, `Routing by Risk Level - Result`, and (for Surface 1) `Restore Extraction Context` — each of which used the single-item `$json` idiom on a multi-item batch and silently discarded every item after the first. This was discovered by tracing a real production incident (a Telegram message correctly classified `high` risk by the Decision Agent, which nonetheless never reached the email-alert step) back through the n8n execution log, node by node, until the exact node discarding the item was identified. All three were fixed with the same `$input.all()` pattern, each verified with a standalone Node.js test before deployment.

### Final production prompt (unchanged from `distress_classification.py`)

```
את/ה מסווג/ת מצוקה עבור מוקד תמיכה המטפל בפניות של בני נוער.
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
```

### Validation ("pass rate" — live, not synthetic)

Two live end-to-end tests via the website ingestion webhook, run **twice** — once against the `Information Extractor`-based implementation, once against the final `AI Agent`-based implementation, to confirm the node-type change produced identical clinical behavior:

| Input (language) | Expected category | `class` returned | `risk_level` derived | Correct? |
|---|---|---|---|---|
| "אני מרגיש ממש לבד לאחרונה, קשה לי להירדם בלילות והרגשות שלי מבולבלים כל הזמן." (Hebrew) | מצוקה רגשית | מצוקה רגשית | medium | ✅ |
| "I've been feeling really overwhelmed and isolated lately, like nobody understands what I'm going through." (English) | מצוקה רגשית | מצוקה רגשית | medium | ✅ |

4/4 live tests correct (2 inputs × 2 node implementations), with identical classification, RAG retrieval, and downstream Decision Agent reasoning in both implementations — confirming the node-type migration (Information Extractor → AI Agent) changed only the n8n architecture, not the clinical output.

---

## Raw prompt extraction — RAG service and LangGraph agent

As requested: raw text only, extracted directly from the current codebase, **no iteration log written yet**.

### RAG service (`rag_retrieval.py`) — finding: no LLM prompt exists

`RAGContextRetriever.build_context()` performs pure vector similarity retrieval (embeddings + nearest-neighbor lookup against the FAISS index) and returns a formatted string of the top-k matches with their similarity scores. **There is no LLM call and no system prompt in this module.** The "RAG Retrieval Prompt" surface described in generic prompt-engineering rubrics does not apply to this codebase as currently built — there is nothing to log for this surface, and nothing should be fabricated to fill the gap. If an LLM-synthesized "insight" narrative (as opposed to raw retrieved matches) is later added to `rag_retrieval.py`, that would become a real, loggable surface at that time.

### LangGraph Decision Agent (`decision_agent_graph.py`, `_build_system_prompt`)

This is a **dynamically constructed** prompt (an f-string rebuilt per invocation with live incident data interpolated), not a static template:

```
You are the Decision Agent in SafeSignal, an emergency and distress triage system. Evaluate the incident context below and decide whether an external tool must be called before you can produce a final assessment.

Incident ID: {state.incident_id}
User ID: {state.user_id}
Raw input: {state.raw_input}
Automated distress classification: {state.distress_classification}
Passive RAG context already retrieved: {state.initial_rag_context}

Available tools:
- trigger_immediate_alert: use ONLY for Critical/High urgency situations that need immediate human/Amazon Polly voice intervention.
- query_rag_history: use when this specific user's own historical incident pattern would materially change the urgency assessment and isn't already covered by the passive RAG context above.

If no tool is needed, respond directly -- your final answer will be parsed into a structured assessment.
```

### Bonus finding — adjacent prompt surface not explicitly requested: hallucination check (`decision_agent_graph.py`, `HALLUCINATION_SYSTEM_PROMPT`)

Discovered during the same scan; flagged here for completeness since it is part of the same output-screening path and will likely need its own log entry later:

```
את/ה בודק/ת בקרת-איכות על תשובה שנוצרה ע"י סוכן AI במערכת טריאז' מצוקה (SafeSignal). קיבלת כמה קטעי טקסט: "הודעת המשתמש המקורית" (raw input), "הסיווג האוטומטי" (distress classification) שנקבע לה קודם בפייפליין, ה"הקשר" (context) שאותר ממאגר הידע (RAG), "כלים שהופעלו בפועל" (tools actually executed) - רשימת שמות הכלים שהסוכן קרא להם בפועל במהלך הריצה, ו"פלט הסוכן" (agent output) - הערכת המצב הסופית שהסוכן ניסח.

תפקידך: לבדוק אם פלט הסוכן כולל טענות עובדתיות שאינן נתמכות ע"י אף אחד מהמקורות שסופקו (הודעת המשתמש המקורית, הסיווג האוטומטי, הקשר ה-RAG, או רשימת הכלים שהופעלו בפועל) - כלומר פרטים, המלצות פעולה קונקרטיות או קביעות עובדתיות שהסוכן "המציא" ואינן מבוססות על אף אחד מהם. ציטוט או תיאור של הודעת המשתמש המקורית, הפניה לסיווג האוטומטי שכבר נקבע, או טענה שפעולה/התראה מסוימת בוצעה/נשלחה - כאשר שם הכלי המתאים (למשל trigger_immediate_alert) מופיע ברשימת "כלים שהופעלו בפועל" - אינם הזיה, גם אם אינם מופיעים במאגר ה-RAG. טענה על פעולה שבוצעה כש*אין* כלי תואם ברשימה כן נחשבת הזיה.

אל תסמן כהזיה: ניסוח מחדש סביר, מסקנות לוגיות ישירות מההקשר, או שימוש בשיקול דעת מקצועי כללי (כמו "מומלץ ליצור קשר עם קו סיוע") שאינו סותר את ההקשר.

החזר/י JSON בלבד: hallucination_detected (bool) ו-reason (משפט קצר בעברית המסביר את ההחלטה).
```

---

## Summary table

| Surface | Prompt authored/iterated this session? | Real engineering work this session | Live validation |
|---|---|---|---|
| Information Extractor (NER) | No — verbatim port | Schema-type gap (array support), business-rule enforcement moved to Code node, single-item batch bug found+fixed | 1 live test, 4/4 fields correct |
| AI Agent (Bedrock classification) | No — verbatim port | Node-type selection (Information Extractor vs. AI Agent, both built and tested), risk-mapping table preserved, safe-default error handling preserved, single-item batch bug found+fixed **in three unrelated downstream nodes** | 4 live tests (2 languages × 2 node implementations), 4/4 correct |
| RAG service | N/A | Confirmed no LLM prompt exists in current codebase | N/A |
| LangGraph Decision Agent | N/A (raw extraction only, per request) | Raw prompt extracted, not yet iterated | N/A |
| Hallucination check | N/A (bonus finding) | Raw prompt extracted, not yet iterated | N/A |
