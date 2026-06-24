# HealthBridgeAI — System Design & Technical Specification

**Version:** 2.1  
**Date:** June 2026  
**Status:** Design — Pre-Implementation  
**Authors:** AHFID AI Labs  
**Scope:** WhatsApp Bot · Twilio · GCP Cloud Run · Multi-Disease · Production-Grade

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Decisions](#2-design-decisions)
3. [Technology Recommendations](#3-technology-recommendations)
4. [Codebase Audit](#4-codebase-audit)
5. [Bug Inventory (31 defects)](#5-bug-inventory)
6. [System Architecture](#6-system-architecture)
7. [RAG Pipeline Design](#7-rag-pipeline-design)
8. [Source Citation System](#8-source-citation-system)
9. [Semantic Response Cache](#9-semantic-response-cache)
10. [Web Search Architecture (Tavily)](#10-web-search-architecture)
11. [Disease Routing & Intent Classification](#11-disease-routing--intent-classification)
12. [Project Structure (Clean Architecture)](#12-project-structure)
13. [GCP Infrastructure](#13-gcp-infrastructure)
14. [Configuration & Secrets](#14-configuration--secrets)
15. [Data Models](#15-data-models)
16. [API Design](#16-api-design)
17. [Security Design](#17-security-design)
18. [Monitoring & Observability](#18-monitoring--observability)
19. [Implementation Roadmap](#19-implementation-roadmap)
20. [Migration Plan](#20-migration-plan)

---

## 1. Executive Summary

HealthBridgeAI is a multilingual WhatsApp chatbot that gives patients and healthcare workers in West Africa expert-level, **source-cited** answers on infectious diseases — diagnosis, treatment, drug interactions, complications, and prevention. It is designed for the general population: low-cost, fast, reliable, and available in local languages.

**What exists today:** A TB-specific bot with a RAG pipeline (Pinecone), multilingual support (English, Yoruba, Igbo, Hausa, Pidgin), voice I/O, and a Flask WhatsApp webhook. It contains a Streamlit UI and Render deployment configuration. A full codebase audit identified **31 bugs** — 6 critical, 9 high, 10 medium, 6 low — several of which prevent the app from starting at all.

**What we are building (v2.1 additions over v2.0):**
- Multi-disease architecture extensible to any condition (TB, HIV/AIDS, Malaria, and future diseases)
- **Twilio WhatsApp API** replacing Meta's direct Cloud API
- **Source citations** on every response — grounded, verifiable, trustworthy
- **Tavily** for authoritative web search (replaces DuckDuckGo), scoped to disease-specific trusted domains only
- **Semantic response caching** via Pinecone — reuses responses for similar queries at zero LLM cost
- **Two-dimensional routing**: disease classification AND query intent classification
- **Advanced RAG pipeline**: BAAI/bge-m3 embeddings, hybrid search (dense + BM25), cross-encoder reranking, HyDE for complex queries
- **FastAPI** (async) replacing Flask (sync) for better I/O throughput
- **Clean Hexagonal Architecture** with strict layer separation
- GCP Cloud Run deployment with Firestore, GCS, Pub/Sub, and Secret Manager
- All 31 bugs fixed as a prerequisite to any new feature work

**Disease roadmap:**

| Disease | Status | Pinecone Namespace |
|---------|--------|-------------------|
| Tuberculosis | Active | `tb` |
| HIV / AIDS | Planned (KB needed) | `hiv` |
| Malaria | Planned (KB needed) | `malaria` |

---

## 2. Design Decisions

### D1 — WhatsApp via Twilio (not Meta direct API)
Twilio's WhatsApp Business API provides a managed abstraction over Meta's Cloud API: built-in retry logic, message queuing, a developer sandbox for testing, and a Python SDK. Twilio's `X-Twilio-Signature` HMAC-SHA1 validation is well-documented and easier to implement correctly than Meta's custom signature scheme. Cost: ~$0.005–$0.015/message (Twilio fee) — acceptable for a healthcare-focused application.

### D2 — WhatsApp only: remove Streamlit and Render
Streamlit adds 200MB+ of unused dependencies. Render's free tier sleeps after 24-hour inactivity — fatal for a real-time health service. Both removed entirely. Flask is also being replaced (see D3).

### D3 — FastAPI (async) over Flask (sync)
Flask is WSGI — each request occupies a thread while waiting on I/O (Pinecone, LLM API, GCS, Twilio). For a bot where the response pipeline is entirely I/O-bound, async/await via FastAPI + Uvicorn is significantly more efficient. FastAPI also provides automatic Pydantic request validation and OpenAPI documentation.

### D4 — GCP Cloud Run for deployment
Scales to zero between message bursts (cost-efficient), scales up instantly during peaks, and integrates natively with Firestore, GCS, Cloud Build, Secret Manager, and Cloud Logging. Minimum instances = 1 in production to eliminate cold-start webhook latency.

### D5 — Pinecone namespaces per disease + one namespace for semantic cache
A single Pinecone index with a namespace per disease provides isolation, independent ingestion, and metadata filtering without separate index costs. A fourth namespace `response-cache` stores semantically embedded prior responses for reuse.

### D6 — Firestore replaces Redis
Firestore is GCP-native, serverless, sub-millisecond for user preference lookups, and supports native TTL on documents. No separate service to manage. Replaces Redis for user preferences and adds full conversation history.

### D7 — GCS replaces local filesystem
Cloud Run containers have ephemeral local storage that resets on restart. Audio files and knowledge base ZIPs live in GCS with lifecycle rules (audio: 24h, KB: permanent with versioning).

### D8 — Cloud Pub/Sub for async processing
Twilio expects a response within 15 seconds. LLM + RAG + optional TTS can take 5–30 seconds. Publishing to Pub/Sub returns immediately; a subscriber Cloud Run service processes and replies asynchronously via the Twilio REST API. Enables retry on failure and dead-letter queuing.

### D9 — Pydantic BaseSettings as the single configuration system
Replaces the two conflicting config systems (`config.py` and `config_manager.py`) and the missing `config.json`. Validates at startup — missing required values raise a clear error before any request is served.

### D10 — YAML-driven disease registry
Adding a new disease requires editing `diseases.yaml` only — no Python code changes. Each disease defines its Pinecone namespace, system prompt, trusted search domains, chunk settings, retrieval threshold, and whether it is enabled.

### D11 — Source citations on every response
Every response must include a clearly formatted "Sources" section listing the documents or web pages used. This is non-negotiable for a medical information service. Citations are extracted from Pinecone chunk metadata (for KB responses) and Tavily results (for web search responses). The LLM is instructed to cite by document number, and only use what was retrieved.

### D12 — Tavily replaces DuckDuckGo for web search
Tavily is purpose-built for AI/RAG applications: it returns structured results (title, URL, extracted content, score), supports `include_domains` filtering for domain-scoped searches, and offers an `advanced` search depth. DuckDuckGo's unofficial API has no domain filtering, unreliable rate limits, and unstructured output. Tavily is the correct tool for this use case.

### D13 — Web search scoped to disease-specific trusted domains only
The fallback is not a general web search. Each disease has a curated list of authoritative sources defined in `diseases.yaml`. Tavily's `include_domains` parameter enforces that every web search result comes from an approved domain. A result from an unapproved domain is discarded even if retrieved.

### D14 — Semantic response caching (Pinecone namespace `response-cache`)
Since the general population will ask similar questions repeatedly, caching at the semantic level (not exact string match) can serve 40–60% of queries at zero LLM cost. Threshold: cosine similarity ≥ 0.92 against the normalised English query embedding. TTL: 7 days. Cache lookup happens in the English query space, so a Yoruba and an Igbo user asking the same question share the same cache entry.

### D15 — Two-dimensional routing: disease + query intent
Every query is classified on two axes: (1) which disease(s) it addresses, and (2) what kind of answer the user wants (symptoms, treatment, prevention, complications, etc.). The query intent drives the response format — structured bullet list for symptoms, numbered protocol for treatment, plain answer for definition. Both classifications use a single LLM call returning a structured JSON object.

### D16 — BAAI/bge-m3 embeddings replace paraphrase-multilingual-MiniLM-L12-v2
BGE-M3 supports 100+ languages, produces 1024-dimensional vectors (vs 384), and consistently outperforms MiniLM on multilingual retrieval benchmarks. It also natively supports sparse output for hybrid search. This is the current state-of-the-art open-source multilingual embedding.

### D17 — Hybrid search: dense + BM25 sparse via Pinecone
Medical queries require both semantic understanding (dense) and exact keyword matching (sparse) — drug names like "rifampicin", "lopinavir", or "artemisinin" must match precisely. Pinecone's native hybrid search combines BAAI/bge-m3 dense vectors with BM25 sparse vectors. Alpha = 0.7 (70% dense, 30% sparse) for general queries; alpha = 0.3 for drug-name queries.

### D18 — Cross-encoder re-ranking after retrieval
Initial retrieval returns 20 candidates. A cross-encoder model (`ms-marco-MiniLM-L-12-v2` via FlashRank) re-ranks them by relevance to the actual query. Top 5 go to the LLM. This dramatically improves precision without increasing LLM context cost.

### D19 — HyDE (Hypothetical Document Embeddings) for complex queries
When initial retrieval scores are low (< 0.5), generate a hypothetical expert answer, embed it, and use that embedding for retrieval. Medical expert writing style matches KB document style better than a patient-phrased question does. Only used as a fallback to avoid latency overhead on standard queries.

### D20 — Clean Hexagonal Architecture
The codebase follows a strict three-layer model: `core/` (domain models, abstract ports, business services — zero external dependencies), `infrastructure/` (concrete adapters that implement ports), and `api/` (HTTP interface). This enables testing with mock adapters, swapping infrastructure components without touching business logic, and clear import boundaries enforced by linting.

---

## 3. Technology Recommendations

This section documents the specific library/model/service choices and the reasoning behind each, beyond what is covered in design decisions.

### Web Framework

| Recommendation | Choice | Why |
|----------------|--------|-----|
| Web framework | **FastAPI 0.115+** | Native async, automatic Pydantic validation, OpenAPI docs |
| ASGI server | **Uvicorn + Gunicorn** | Production-grade, concurrency model matches Cloud Run |
| Replace | Flask (current) | Synchronous only; no async support for I/O-bound pipeline |

### Embeddings

| Recommendation | Choice | Why |
|----------------|--------|-----|
| Primary model | **BAAI/bge-m3** | 100+ languages, 1024-dim, best multilingual quality, sparse+dense |
| Inference | HuggingFace pipeline (CPU) or Inference API | Cloud Run has no GPU; bge-m3 is usable on CPU for batch indexing |
| Dimension | 1024 | Pinecone index must be configured to 1024 |
| Replace | paraphrase-multilingual-MiniLM-L12-v2 (384-dim) | Lower quality, lower dimension |

### Retrieval

| Recommendation | Choice | Why |
|----------------|--------|-----|
| Chunking strategy | **Semantic chunking** | Groups semantically coherent sentences rather than splitting at fixed token count |
| Library | `langchain-experimental` `SemanticChunker` | Production-ready, configurable thresholds |
| Search type | **Hybrid (dense + BM25)** | Medical terminology requires keyword precision; semantic alone misses drug names |
| Sparse encoding | `pinecone-text` BM25Encoder | Native Pinecone sparse format |
| Re-ranking | **FlashRank** (`ms-marco-MiniLM-L-12-v2`) | Fast CPU cross-encoder; open source; significant precision improvement |
| HyDE fallback | LLM-generated hypothetical answer | Used only when initial retrieval score < 0.5 |

### Web Search

| Recommendation | Choice | Why |
|----------------|--------|-----|
| Search API | **Tavily API** (`tavily-python`) | Built for AI/RAG; structured output; `include_domains` filtering |
| Replace | DuckDuckGo (`duckduckgo-search`) | No domain filter, unofficial API, unstructured output |
| Tier | Tavily paid tier | Free tier: 1,000 calls/month; insufficient for production |
| Alternative | **Brave Search API** | Open web, reliable, free developer tier (2,000 queries/month) |

### LLM

| Recommendation | Choice | Why |
|----------------|--------|-----|
| Primary model | **`anthropic/claude-haiku-4-5`** | Fast, cheap, excellent instruction following for structured JSON |
| Router model | Same (`claude-haiku-4-5`) | Routing + intent is a cheap call |
| Heavy queries | **`anthropic/claude-sonnet-4-6`** | Use for complex drug-interaction or multi-disease queries |
| Provider | OpenRouter | Access to Anthropic, OpenAI, and fallback models via single API |
| Structured output | Pydantic model + instructor library | Reliable JSON parsing; retry on malformed output |

### Response Caching

| Recommendation | Choice | Why |
|----------------|--------|-----|
| Strategy | **Semantic cache in Pinecone** | Reuses existing infrastructure; no extra service |
| Alternative | **GPTCache** library | Drop-in semantic cache; supports multiple backends including Faiss |
| Similarity threshold | 0.92 | High precision required for medical content — wrong cached answers are dangerous |
| TTL | 7 days | Medical guidelines change slowly; 7 days is safe and efficient |
| Cache namespace | Pinecone `response-cache` | Isolated from KB namespaces |

### Audio

| Recommendation | Choice | Why |
|----------------|--------|-----|
| Transcription (primary) | **N-ATLAS** | Trained specifically on Nigerian languages (Hausa, Yoruba, Igbo, Pidgin) |
| Transcription (fallback 1) | **OpenAI Whisper** (large-v3) | Excellent multilingual; handles code-switching |
| Transcription (fallback 2) | Google Speech-to-Text | Reliable fallback; charged per 15 seconds |
| Synthesis (primary) | **Yarn GPT** | Nigerian language TTS with natural prosody |
| Synthesis (fallback 1) | Meta MMS | Open-source, supports Hausa/Yoruba/Igbo |
| Synthesis (fallback 2) | **gTTS** | Always available; English quality |

### Chunking (KB ingestion)

| Recommendation | Choice | Why |
|----------------|--------|-----|
| Strategy | Semantic chunking | Coherent semantic units improve retrieval relevance |
| Chunk labelling | LLM-based chunk type classification | Label each chunk as `symptoms` / `treatment` / `prevention` etc. during ingestion |
| Metadata | Full source metadata per chunk | Enables citations and intent-filtered retrieval |

### Testing

| Recommendation | Choice | Why |
|----------------|--------|-----|
| Unit tests | pytest + `unittest.mock` | Standard; works with dependency injection |
| RAG evaluation | **RAGAS** | Measures faithfulness, answer relevance, context precision, recall |
| Coverage target | ≥ 80% core services, ≥ 60% infrastructure | Enforce in CI |
| Integration tests | pytest with real Pinecone test namespace | Test retrieval quality end-to-end |

---

## 4. Codebase Audit

### Current file inventory
```
HealthBridgeAI/
├── app.py                     # Flask webhook + Streamlit launcher
├── app-st.py                  # Streamlit UI — DELETE
├── main.py                    # Streamlit entry point — DELETE
├── config.py                  # Old flat config — DELETE
├── Dockerfile
├── render.yaml                # Render.com config — DELETE
├── requirements.txt
├── populate_kb.py
├── install_packages.py        # Non-standard installer — DELETE
├── modules/
│   ├── llm_handler.py
│   ├── knowledge_base_manager.py
│   ├── vector_store_manager.py
│   ├── audio_transcriber.py
│   ├── audio_synthesizer.py   # SYNTAX ERROR
│   ├── audio_handler.py       # Incomplete wrappers — DELETE
│   ├── language_utils.py
│   ├── language_service.py
│   ├── user_preferences.py    # Redis-based — replace with Firestore
│   ├── cache_manager.py
│   ├── config_manager.py      # Duplicate config — DELETE
│   ├── session_manager.py     # Streamlit state — DELETE
│   ├── response_strategies.py # Streamlit dependency — DELETE
│   ├── exceptions.py
│   ├── utils.py
│   └── utills.py              # TYPO FILENAME — merge then delete
└── data/
    └── TB_knowledge_base.zip  # Move to GCS
```

### What works
- Flask WhatsApp webhook with GET verification and POST message handling
- RAG pipeline: PDF loading, chunking, HuggingFace embeddings, Pinecone upsert and retrieval
- Multilingual support: language detection, LLM-based translation (EN ↔ Yoruba/Igbo/Hausa/Pidgin)
- Audio transcription: N-ATLAS → Whisper → Google Speech Recognition fallback chain
- Audio synthesis: Yarn GPT → Meta MMS → gTTS fallback chain
- File-based caching for translations and KB results
- Web search fallback via DuckDuckGo

### What must be removed
| File | Reason |
|------|--------|
| `app-st.py` | Streamlit — removed per D2 |
| `main.py` | Streamlit entry point |
| `render.yaml` | Render.com deployment — replaced by Cloud Run |
| `session_manager.py` | Streamlit session state — irrelevant |
| `config.py` | Old config — replaced by Pydantic Settings |
| `config_manager.py` | Duplicate config — merged into Settings |
| `utills.py` | Typo filename — merged into `message_parser.py` |
| `audio_handler.py` | Incomplete backwards-compat wrappers |
| `install_packages.py` | Non-standard; `requirements.txt` is sufficient |
| `response_strategies.py` | Streamlit-coupled |

---

## 5. Bug Inventory

### Critical — App cannot start or core execution path is broken

| # | File | Bug | Fix |
|---|------|-----|-----|
| C1 | `app.py:60–64` | `config.json` does not exist — Flask cannot start | Load all credentials from env vars via Pydantic Settings |
| C2 | `audio_synthesizer.py:49–52` | Docstring placed after variable assignments — invalid Python | Remove misplaced docstring |
| C3 | `audio_synthesizer.py:49–50` | Hardcoded `ogg_path = "test.ogg"` — concurrent requests corrupt each other's audio | UUID-based GCS paths per request |
| C4 | `config.py` + `config_manager.py` | Two config systems, no clear authority; `app.py` uses neither | Delete both; create `config/settings.py` |
| C5 | `app.py` POST handler | No signature verification on inbound messages (META or Twilio) | Validate `X-Twilio-Signature` on every POST (Twilio) |
| C6 | `audio_synthesizer.py:126–134` | Unreachable code after `return` — MP3 fallback path permanently dead | Restructure control flow with explicit format selection |

### High — Significant risk in production

| # | File | Bug | Fix |
|---|------|-----|-----|
| H1 | `llm_handler.py` | No input length limit — LLM token overflow possible | Truncate at 2,000 characters with user notice |
| H2 | `app.py:139–142` | Audio temp files not reliably cleaned up | GCS lifecycle rules (24h auto-delete) |
| H3 | `utills.py:24–81` | Direct dict key access without `.get()` — KeyError on any non-standard Twilio payload | Defensive parsing with explicit type checks |
| H4 | `llm_handler.py`, `vector_store_manager.py` | No retry logic for transient API failures | Exponential backoff (tenacity), max 3 retries |
| H5 | `app.py` POST handler | No rate limiting per user | Firestore sliding-window rate limit (20 msg/min) |
| H6 | `user_preferences.py:46–51` | Redis in-memory fallback loses all preferences on restart | Replace with Firestore |
| H7 | `llm_handler.py:374`, `app.py:248` | Language code validation inconsistent across modules | Single source of truth in `settings.py` |
| H8 | All modules | Multiple `logging.basicConfig()` calls — only first takes effect | Single logger in main, JSON structured output |
| H9 | `config.py:80` | `MIN_SCORE = 1.5` too strict — forces excessive web search fallback | Configurable per disease in `diseases.yaml`, default 0.6 |

### Medium — Degraded experience or technical debt

| # | Bug | Fix |
|---|-----|-----|
| M1 | `utills.py` filename typo | Merge into `modules/bot/message_parser.py`, delete |
| M2 | `language_utils.py` — 30% word-match threshold fails on short messages | Require ≥4 words for rule-based; LLM fallback for short input |
| M3 | `llm_handler.py:295` — DuckDuckGo search prefix hardcoded to "tuberculosis" | Replace with Tavily + disease-specific `search_domains` from `diseases.yaml` |
| M4 | Web search results not cached | Add 1-hour TTL cache entry or handle via semantic cache |
| M5 | All translation LLM calls synchronous | Use `asyncio.gather` for parallel translation + generation (FastAPI) |
| M6 | No conversation history — context lost between messages | Firestore conversation collection, inject last 5 turns |
| M7 | `populate_kb.py:76–78` — hard `sys.exit(1)` on missing KB | Log error, disable KB for that disease, continue |
| M8 | Chunk size hardcoded 1000/200 for all document types | Semantic chunking with per-disease configuration |
| M9 | No analytics, conversation logging, or usage metrics | Structured events to Cloud Logging; Cloud Monitoring dashboard |
| M10 | `audio_handler.py` wrappers incomplete | Remove wrapper; import transcriber/synthesizer directly |

### Low — Code quality and maintainability

| # | Bug | Fix |
|---|-----|-----|
| L1 | Cache TTL hardcoded 24h for all cache types | Per-type TTL: translations 7d, KB 24h, semantic cache 7d |
| L2 | `except Exception:` used broadly | Catch specific exceptions; re-raise unknown with context |
| L3 | No `.env.example` | Add with all required and optional variables documented |
| L4 | Render health-check points to Streamlit `/_stcore/health` | Removed with `render.yaml` |
| L5 | No Pinecone connection pooling | Create client once at startup, share as singleton |
| L6 | `response_strategies.py` references Streamlit | Remove with other Streamlit files |

---

## 6. System Architecture

### Two-service Cloud Run design

**Service 1 — Webhook Receiver**  
Accepts Twilio POST requests, validates `X-Twilio-Signature`, publishes raw payload to Pub/Sub `messages-inbound`, returns empty TwiML `<Response></Response>` within 2 seconds. Min instances: 1. 512MB RAM, 1 vCPU.

**Service 2 — Message Processor**  
Pub/Sub push subscriber. Runs the full pipeline: parse → detect language → check semantic cache → route disease + intent → retrieve KB → rerank → generate response + citations → optionally synthesize audio → reply via Twilio REST API → store in Firestore → update cache. Min instances: 1, max: 20, 2GB RAM, 2 vCPU, 5-minute timeout.

### Twilio message flow

```
User (WhatsApp)
      │
      │ POST (form-encoded)
      ▼
Twilio WhatsApp API
      │
      │ POST /webhook + X-Twilio-Signature
      ▼
Webhook Service (Cloud Run)
  1. Validate X-Twilio-Signature (HMAC-SHA1)
  2. Parse From, Body, MediaUrl0, NumMedia
  3. Publish to Pub/Sub
  4. Return <Response></Response> (empty TwiML)
      │
      │ Pub/Sub push
      ▼
Processor Service (Cloud Run)
  1. Parse Pub/Sub message
  2. Load user (Firestore) + rate limit check
  3. If audio: download from Twilio MediaUrl → GCS → transcribe
  4. Detect language
  5. Translate to English (if needed)
  6. Check semantic cache (Pinecone response-cache namespace)
  7. [Cache hit] → translate to user lang → Twilio REST API → done
  8. [Cache miss] → DiseaseRouter (disease + intent) →
  9. RAGService (hybrid search → rerank → HyDE fallback if needed)
  10. Tavily search (if retrieval score < threshold)
  11. ResponseGenerator (LLM + citation assembly)
  12. Translate response (if needed) + synthesize audio (if requested)
  13. Send via Twilio REST API
  14. Store turn in Firestore
  15. Store in semantic cache
  16. Acknowledge Pub/Sub message
      │
      │ POST (Twilio REST API)
      ▼
Twilio WhatsApp API → User
```

### User commands (plain text, processed before pipeline)

| Command | Action |
|---------|--------|
| `language en` / `yo` / `ha` / `ig` / `pidgin` | Set preferred response language |
| `audio on` / `audio off` | Enable or disable voice responses |
| `about` | Show bot capabilities and disease list |
| `help` | Show available commands |
| `feedback [message]` | Submit feedback stored in Firestore |

---

## 7. RAG Pipeline Design

This is the most important technical section. Every component choice here directly affects the accuracy and reliability of health information delivered to the general population.

### 7.1 Knowledge Base Ingestion Pipeline

```
KB ZIP (GCS)
    ↓
Extract PDFs/docs
    ↓
Semantic Chunking (SemanticChunker, threshold=0.85)
    ↓
Chunk Type Labelling (LLM call: symptoms/treatment/prevention/etc.)
    ↓
Source Metadata Extraction (doc title, section, page, URL)
    ↓
Dense Embedding (BAAI/bge-m3, 1024-dim)
    ↓
Sparse Encoding (BM25Encoder from pinecone-text)
    ↓
Upsert to Pinecone (namespace=disease_id, with full metadata)
```

#### Semantic chunking
Instead of splitting at fixed token counts (which breaks mid-sentence and mid-concept), semantic chunking groups sentences by semantic similarity. Adjacent sentences with cosine similarity > 0.85 remain in the same chunk. This produces coherent, topically unified chunks that align better with query semantics.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85
)
chunks = chunker.split_text(document_text)
```

#### Chunk type labelling during ingestion
Each chunk is labelled with its content type using a fast LLM call. This enables intent-filtered retrieval (see Section 11).

Labels: `symptoms`, `treatment`, `prevention`, `complications`, `drug_interaction`, `transmission`, `testing`, `statistics`, `definition`, `general`

#### Chunk metadata schema (stored in Pinecone)

```json
{
  "text": "Full chunk text",
  "disease": "tb",
  "doc_id": "who_tb_guidelines_2022",
  "source_doc": "WHO Consolidated Guidelines on Tuberculosis, Module 4 (2022)",
  "source_url": "https://www.who.int/publications/i/item/9789240048928",
  "source_type": "guideline",
  "section": "Chapter 4: Treatment of drug-susceptible TB",
  "page_number": 45,
  "chunk_type": "treatment",
  "chunk_index": 12,
  "language": "en"
}
```

### 7.2 Query-Time Retrieval Pipeline

```
User query (English)
      ↓
[1] Hybrid Encoding
    ├── Dense: BAAI/bge-m3 → 1024-dim vector
    └── Sparse: BM25Encoder → sparse vector
      ↓
[2] Pinecone Hybrid Search
    ├── Namespace: relevant disease(s)
    ├── top_k = 20
    ├── alpha = 0.7 (dense) / 0.3 (sparse) [drug queries: 0.3/0.7]
    └── Optional metadata filter: chunk_type = query_intent
      ↓
[3] Cross-Encoder Re-ranking (FlashRank)
    ├── Input: top-20 candidates
    ├── Model: ms-marco-MiniLM-L-12-v2
    └── Output: top-5 by cross-encoder score
      ↓
[4] Score threshold check
    ├── Best score ≥ 0.6: use KB results
    └── Best score < 0.6: trigger HyDE fallback or Tavily search
      ↓
[5] (Fallback A) HyDE — if score < 0.5
    ├── Generate hypothetical expert answer via LLM
    ├── Embed hypothetical answer (BAAI/bge-m3)
    └── Re-query Pinecone with hypothetical embedding → re-rank → repeat step 4
      ↓
[6] (Fallback B) Tavily web search — if score < 0.6 after HyDE
    ├── Search only approved domains for this disease
    ├── max_results = 5, search_depth = "advanced"
    └── Validate all results come from approved domain list
      ↓
[7] Contextual compression (optional, for long chunks)
    ├── Use LLM to extract only the relevant sentence(s) from each chunk
    └── Reduces context noise before generation
      ↓
Context documents → ResponseGenerator
```

### 7.3 Multi-Query Retrieval (for ambiguous queries)

When query intent is classified as `GENERAL` or when the disease router returns low confidence, generate 3 query variations and retrieve for each, then deduplicate by chunk ID:

```python
variations = llm.generate(
    f"Generate 3 different ways a patient might ask: '{query}'"
    "Return JSON array of 3 strings."
)
all_results = [retrieve(v, disease_ids) for v in variations]
unique = deduplicate_by_chunk_id(flatten(all_results))
```

### 7.4 RAG Quality Evaluation (RAGAS)

Run RAGAS evaluation on a test question set for each disease KB before enabling it in production:

| Metric | Target | What it measures |
|--------|--------|-----------------|
| Faithfulness | ≥ 0.85 | Response is grounded in retrieved context (no hallucinations) |
| Answer Relevancy | ≥ 0.80 | Response actually answers the question |
| Context Precision | ≥ 0.70 | Retrieved chunks are relevant to the question |
| Context Recall | ≥ 0.70 | Retrieved chunks cover the answer |

---

## 8. Source Citation System

Every response must include citations. This is the single most important trust signal for a medical information service targeting the general population.

### 8.1 Citation extraction

**From KB retrieval:** Extract metadata from each Pinecone chunk used in the response:
- `source_doc` → document title
- `source_url` → direct URL to the document (if available)
- `section` → chapter or section name
- `page_number` → page reference

**From Tavily web search:** Extract from each result used:
- `title` → article or page title
- `url` → source URL
- `published_date` → recency indicator

### 8.2 LLM citation grounding

The LLM prompt is structured so that each context document is numbered. The LLM is instructed to:
1. Only use information from the provided numbered context
2. Cite each claim with the document number(s) used
3. List the document numbers it actually used at the end of its answer
4. Say "I don't have information on this in my knowledge base" if context is insufficient

**Prompt structure:**
```
System: You are a medical information expert specializing in {disease_name}.
Answer using ONLY the numbered context documents below.
Cite each claim with [Doc N]. At the end, list "Sources Used: [1], [3]" etc.
If the context is insufficient, say so — do not guess.
{disease_specific_guidelines}
{medical_disclaimer}

Context Documents:
[1] WHO TB Treatment Guidelines 2022, Chapter 4 (p.45):
    "{chunk_text}"

[2] CDC | Tuberculosis Treatment | cdc.gov:
    "{web_search_result_content}"

[3] StopTB Partnership Technical Brief 2023:
    "{chunk_text}"

Question: {user_query}
Conversation history: {last_5_turns}
```

**LLM structured output (Pydantic):**
```python
class LLMResponse(BaseModel):
    answer: str
    sources_used: list[int]    # Indices of context docs actually cited
    confidence: str            # "high" | "medium" | "low"
    needs_professional: bool   # True if professional consultation is advised
    caveat: str | None         # Any important qualification
```

### 8.3 WhatsApp citation format

WhatsApp message length limit: 4,096 characters. Sources are appended after the answer:

```
[Answer text with inline citations where appropriate...]

─────────────────────
📚 *Sources:*
1. WHO Consolidated Guidelines on TB Module 4 (2022) — who.int
2. CDC | Tuberculosis Treatment — cdc.gov

⚠️ _This information is for educational purposes. Always consult a qualified 
healthcare professional for medical advice._
```

For audio responses: sources are sent as a follow-up text message immediately after the audio.

### 8.4 Medical disclaimer

Appended to every response regardless of source:

> *This information is for educational purposes only. Always consult a qualified healthcare professional for diagnosis, treatment, or any medical decision. If you are experiencing a medical emergency, seek immediate help.*

For audio: synthesized and appended to the audio response.

---

## 9. Semantic Response Cache

### 9.1 How it works

The cache stores prior responses indexed by the semantic embedding of the normalised English query. When a new query arrives with similarity ≥ 0.92 to a cached query, the cached response is returned — translated to the user's language if needed.

**Cache hit rate expectation:** 40–60% for a general-population health bot, since common questions ("How is TB treated?", "What are the symptoms of malaria?") recur constantly.

### 9.2 Cache flow

```
User message (any language)
      ↓
Translate to English (LLM)
      ↓
Normalize: lowercase, strip punctuation
      ↓
Embed with BAAI/bge-m3 (1024-dim)
      ↓
Search Pinecone namespace: response-cache
  filter: { disease_ids: ["tb"] }
  top_k: 1
      ↓
Score ≥ 0.92?
  YES → Deserialize cached response
        Translate to user's language (if different)
        Return with cache marker
        Increment hit_count in Firestore
  NO  → Run full RAG + LLM pipeline
        Store result in cache
        Return to user
```

### 9.3 Cache schema (Pinecone namespace `response-cache`)

```json
{
  "id": "sha256(normalized_english_query + '|' + sorted_disease_ids_joined)",
  "values": [1024-dim BAAI/bge-m3 embedding],
  "metadata": {
    "english_query": "What are the symptoms of tuberculosis?",
    "disease_ids": ["tb"],
    "query_intent": "symptoms",
    "english_response": "The main symptoms of tuberculosis include...",
    "sources_json": "[{\"doc\": \"WHO Guidelines\", \"url\": \"who.int/...\"}, ...]",
    "confidence": "high",
    "created_at": 1719100000,
    "expires_at": 1719705000,
    "hit_count": 0
  }
}
```

### 9.4 Cache management

| Operation | Trigger | Action |
|-----------|---------|--------|
| Write | Every cache miss that produces a response | Store embedding + response |
| Read | Every query, before pipeline | Search response-cache namespace |
| Invalidate | When KB is updated for a disease | Delete all entries with matching `disease_ids` |
| Expire | Scheduled cleanup (daily Cloud Scheduler job) | Delete entries where `expires_at` < now |
| Warm-up | After KB indexing | Pre-cache responses to the top-50 common questions per disease |

### 9.5 What is NOT cached

- User command responses (`help`, `language`, `about`, `feedback`)
- Drug dosage and emergency responses (safety: always re-generate)
- Responses where `confidence` was `low` or `needs_professional` was `true`
- Personal health questions (identified by presence of "I have", "my doctor", "my symptoms")

---

## 10. Web Search Architecture

### 10.1 Tavily integration

```python
from tavily import TavilyClient

client = TavilyClient(api_key=settings.TAVILY_API_KEY) 

results = client.search(
    query=f"{query}",                          # English query
    include_domains=disease.search_domains,    # From diseases.yaml
    max_results=5,
    search_depth="advanced",                   # More thorough extraction
    include_raw_content=False,                 # Use Tavily's extracted content
)
```

Tavily returns: `title`, `url`, `content` (extracted), `score`, `published_date`.

### 10.2 Result validation

After Tavily returns results, all results are validated:
1. Check `url` domain is in `disease.search_domains` — discard if not
2. Check `score` ≥ 0.5 — discard low-relevance results
3. Check `published_date` — if older than 3 years, flag as potentially outdated (note in response)
4. If < 2 results pass validation, return "no reliable external source found" rather than using unvalidated content

### 10.3 Trusted domains per disease (in diseases.yaml)

**TB:**
```
who.int, cdc.gov, stoptb.org, tbfacts.org, nhs.uk, mayoclinic.org,
tbonline.info, iuatld.org, nicd.ac.za
```

**HIV / AIDS:**
```
who.int, unaids.org, aidsinfo.nih.gov, cdc.gov, aidsmap.com,
nhs.uk, mayoclinic.org, hiv.gov
```

**Malaria:**
```
who.int, cdc.gov, rollbackmalaria.org, malariaconsortium.org,
malaria.org, nicd.ac.za, nmcp.gov.ng
```

### 10.4 When to trigger web search

Web search is triggered when **both** conditions are true:
1. Best KB retrieval score (after re-ranking) < `disease.min_retrieval_score` (default 0.6)
2. HyDE re-retrieval also scores below threshold

Web search is **never** triggered for:
- Questions requiring personalised medical advice
- Drug dosage questions (too risky without verified prescription context)
- Emergency triage questions (direct to emergency services immediately)

---

## 11. Disease Routing & Intent Classification

### 11.1 Two-dimensional classification

A single LLM call classifies both dimensions simultaneously, returning a structured JSON object:

```python
class RouteResult(BaseModel):
    disease_ids: list[str]           # e.g., ["tb"], ["hiv", "tb"], []
    disease_confidence: float        # 0.0 – 1.0
    query_intent: QueryIntent        # Enum: see below
    intent_confidence: float         # 0.0 – 1.0
    is_general_health: bool          # True if no specific disease detected
    is_emergency: bool               # True if symptoms suggest medical emergency
    is_personal: bool                # True if user is asking about themselves
```

### 11.2 Query intent types

```python
class QueryIntent(str, Enum):
    SYMPTOMS         = "symptoms"          # What are the symptoms?
    TREATMENT        = "treatment"         # How is it treated?
    PREVENTION       = "prevention"        # How to prevent?
    COMPLICATIONS    = "complications"     # What complications can occur?
    DRUG_INTERACTION = "drug_interaction"  # Drug effects or interactions
    TRANSMISSION     = "transmission"      # How does it spread?
    TESTING          = "testing"           # How to test / diagnose?
    STATISTICS       = "statistics"        # Prevalence, mortality data
    DEFINITION       = "definition"        # What is [term]?
    GENERAL          = "general"           # Any other query
```

### 11.3 Intent affects retrieval and response format

| Intent | Pinecone filter | Response format |
|--------|----------------|-----------------|
| SYMPTOMS | `chunk_type: "symptoms"` | Bullet list, organized by severity |
| TREATMENT | `chunk_type: "treatment"` | Numbered protocol, include drug names and duration |
| PREVENTION | `chunk_type: "prevention"` | Numbered list of measures |
| COMPLICATIONS | `chunk_type: "complications"` | Categorized list (pulmonary / systemic / etc.) |
| DRUG_INTERACTION | `chunk_type: "drug_interaction"` | Bold warning if dangerous; specific guidance |
| TRANSMISSION | `chunk_type: "transmission"` | Brief factual answer + prevention link |
| TESTING | `chunk_type: "testing"` | List of tests, where to access them |
| STATISTICS | `chunk_type: "statistics"` | Numbers with context + year |
| DEFINITION | `chunk_type: "definition"` | Plain explanation, avoid jargon |
| GENERAL | No filter | Standard prose response |

### 11.4 Emergency detection

If `is_emergency: true`:
- Skip the full pipeline entirely
- Return an immediate emergency response:
  > *If you are experiencing a medical emergency, please call your nearest hospital or emergency service immediately. In Nigeria: NEMA Emergency Hotline 080097000010. Do not wait for an online response.*
- Log the event to Cloud Logging with severity=CRITICAL
- Do not cache this response

### 11.5 Multi-disease queries

When `disease_ids` contains more than one disease (e.g., `["tb", "hiv"]` for a co-infection query):
- Retrieve from all relevant namespaces
- Merge and re-rank across namespaces
- Use a combined system prompt: both disease contexts + co-infection specific guidance
- Sources from all retrieved namespaces are included

### 11.6 Routing fallback priority

```
1. Alias match (keyword in query matches disease.aliases) → zero cost
2. LLM classification (if alias match ambiguous or no match)
3. Conversation history context (inherit previous turn's disease_ids for follow-up)
4. General health response (if all above fail)
```

---

## 12. Project Structure

Following Clean Hexagonal Architecture: `core` (domain logic, zero external dependencies) ↔ `infrastructure` (adapters for external services) ↔ `api` (HTTP interface). Dependencies point inward only: `api` → `core`, `infrastructure` → `core`, never `core` → `infrastructure`.

```
healthbridgeai/
├── pyproject.toml                       # PEP 517/518 — replaces setup.py
├── .env.example                         # All vars documented, no values
├── .gitignore
├── SYSTEM_DESIGN.md
│
├── src/
│   └── healthbridgeai/
│       ├── __init__.py
│       ├── main.py                      # FastAPI app factory + lifespan startup
│       │
│       ├── core/                        # Domain layer — zero external dependencies
│       │   ├── __init__.py
│       │   ├── models/                  # Pydantic domain models (data shapes only)
│       │   │   ├── __init__.py
│       │   │   ├── message.py           # InboundMessage, ParsedMessage, MessageType
│       │   │   ├── disease.py           # DiseaseDomain, QueryIntent, RouteResult
│       │   │   ├── retrieval.py         # Chunk, Source, RetrievalResult, WebResult
│       │   │   ├── response.py          # BotResponse, CachedResponse, LLMResponse
│       │   │   └── user.py              # User, ConversationTurn, RateLimit
│       │   ├── ports/                   # Python Protocol interfaces (structural typing)
│       │   │   ├── __init__.py
│       │   │   ├── messaging.py         # IMessagingProvider (send_text, send_audio, download_media)
│       │   │   ├── vector_store.py      # IVectorStore (upsert, hybrid_search, delete_namespace)
│       │   │   ├── llm.py               # ILLMClient (generate, generate_structured)
│       │   │   ├── search.py            # IWebSearch (search, with domain_filter)
│       │   │   ├── storage.py           # IUserStore, IConversationStore
│       │   │   └── cache.py             # IResponseCache (get, set, invalidate)
│       │   └── services/                # Business logic — depends only on ports
│       │       ├── __init__.py
│       │       ├── pipeline.py          # MessagePipeline — main orchestrator
│       │       ├── router.py            # DiseaseRouter — disease + intent classification
│       │       ├── rag.py               # RAGService — hybrid search, rerank, HyDE
│       │       ├── generator.py         # ResponseGenerator — LLM + citation assembly
│       │       └── language.py          # LanguageService — detection + translation
│       │
│       ├── infrastructure/              # Adapters (implement ports — external deps live here)
│       │   ├── __init__.py
│       │   ├── messaging/
│       │   │   ├── __init__.py
│       │   │   └── twilio.py            # TwilioAdapter — implements IMessagingProvider
│       │   ├── vector_store/
│       │   │   ├── __init__.py
│       │   │   └── pinecone.py          # PineconeAdapter — implements IVectorStore
│       │   ├── llm/
│       │   │   ├── __init__.py
│       │   │   └── openrouter.py        # OpenRouterAdapter — implements ILLMClient
│       │   ├── search/
│       │   │   ├── __init__.py
│       │   │   └── tavily.py            # TavilyAdapter — implements IWebSearch
│       │   ├── storage/
│       │   │   ├── __init__.py
│       │   │   ├── firestore.py         # FirestoreAdapter — IUserStore + IConversationStore
│       │   │   └── gcs.py               # GCSAdapter — file storage
│       │   ├── cache/
│       │   │   ├── __init__.py
│       │   │   └── semantic.py          # PineconeSemanticCacheAdapter — IResponseCache
│       │   └── audio/
│       │       ├── __init__.py
│       │       ├── transcriber.py       # N-ATLAS → Whisper → Google Speech-to-Text
│       │       └── synthesizer.py       # YarnGPT → Meta MMS → gTTS (UUID-based GCS paths)
│       │
│       ├── api/                         # HTTP layer — thin, no business logic
│       │   ├── __init__.py
│       │   ├── webhook.py               # POST /webhook (Twilio validation + Pub/Sub publish)
│       │   └── health.py                # GET /health, GET /ready
│       │
│       └── config/
│           ├── __init__.py
│           ├── settings.py              # Pydantic BaseSettings — single config source
│           └── diseases.yaml            # Disease registry (no code changes to add disease)
│
├── processor/                           # Pub/Sub subscriber — separate Cloud Run service
│   ├── __init__.py
│   └── main.py                          # FastAPI app for Pub/Sub push endpoint
│
├── scripts/
│   ├── populate_kb.py                   # --disease flag, GCS upload + Pinecone indexing
│   ├── evaluate_kb.py                   # RAGAS evaluation per disease
│   ├── warm_cache.py                    # Pre-cache top-N common questions per disease
│   └── setup_gcp.sh                     # Provision all GCP resources
│
├── deploy/
│   ├── Dockerfile                       # Multi-stage build: webhook service
│   ├── Dockerfile.processor             # Multi-stage build: processor service
│   ├── cloudbuild.yaml                  # CI/CD pipeline
│   └── cloudrun.yaml                    # Cloud Run service configs
│
└── tests/
    ├── conftest.py                      # Shared fixtures, mock adapters
    ├── unit/
    │   ├── core/
    │   │   ├── test_router.py           # DiseaseRouter unit tests
    │   │   ├── test_rag.py              # RAGService unit tests (mocked vector store)
    │   │   ├── test_generator.py        # ResponseGenerator unit tests
    │   │   └── test_pipeline.py         # Pipeline unit tests (all ports mocked)
    │   └── infrastructure/
    │       ├── test_twilio.py           # Twilio adapter tests
    │       └── test_pinecone.py         # Pinecone adapter tests
    └── integration/
        ├── test_retrieval.py            # Real Pinecone test namespace
        ├── test_cache.py                # Semantic cache integration
        └── test_end_to_end.py           # Full pipeline with test WhatsApp number
```

### pyproject.toml structure

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "healthbridgeai"
version = "2.1.0"
requires-python = ">=3.11"

[tool.hatch.build.targets.wheel]
packages = ["src/healthbridgeai"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
strict = true
```

---

## 13. GCP Infrastructure

### Services

| Service | GCP Resource | Configuration |
|---------|-------------|---------------|
| Cloud Run — webhook | `healthbridge-webhook` | 512MB, 1 vCPU, min 1, max 10 instances |
| Cloud Run — processor | `healthbridge-processor` | 2GB, 2 vCPU, min 1, max 20 instances, 5-min timeout |
| Cloud Pub/Sub | topic: `messages-inbound` | Dead-letter topic after 5 retries |
| Firestore | Native mode, `africa-south1` | Collections: users, conversations, feedback |
| Cloud Storage | `gs://healthbridge-assets` | Buckets: `knowledge-bases/`, `audio-cache/` (24h), `media-downloads/` (1h) |
| Secret Manager | `secrets/healthbridge-*` | All API keys and tokens |
| Artifact Registry | `healthbridge-images` | Docker images tagged by git SHA |
| Cloud Build | `cloudbuild.yaml` trigger | On push to `main` |
| Cloud Scheduler | `cache-cleanup` job | Daily: delete expired cache entries |
| Cloud Logging + Monitoring | `healthbridge-dashboard` | Structured JSON logs + custom metrics |

### IAM service accounts

| Service Account | Roles |
|----------------|-------|
| `webhook-sa` | `pubsub.publisher`, `secretmanager.secretAccessor`, `logging.logWriter` |
| `processor-sa` | `datastore.user`, `storage.objectAdmin`, `secretmanager.secretAccessor`, `logging.logWriter` |
| `cloudbuild-sa` | `run.admin`, `artifactregistry.writer`, `iam.serviceAccountUser` |
| `scheduler-sa` | `run.invoker` (for cache cleanup Cloud Run job) |

### Cloud Build pipeline (cloudbuild.yaml)

Steps in order:
1. `pip install` dev dependencies + `ruff lint` + `mypy` type check
2. `pytest tests/unit/ --cov=src/healthbridgeai --cov-report=xml`
3. `docker build` webhook image → push to Artifact Registry
4. `docker build` processor image → push to Artifact Registry
5. `gcloud run deploy healthbridge-webhook`
6. `gcloud run deploy healthbridge-processor`

---

## 14. Configuration & Secrets

### config/settings.py

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Twilio WhatsApp
    TWILIO_ACCOUNT_SID: str
    TWILIO_AUTH_TOKEN: str
    TWILIO_WHATSAPP_FROM: str          # e.g. "+14155238886" (Twilio sandbox) or approved number
    TWILIO_WEBHOOK_URL: str            # Full URL of /webhook endpoint (used for validation)

    # LLM via OpenRouter
    OPENROUTER_API_KEY: str
    LLM_PRIMARY_MODEL: str = "anthropic/claude-haiku-4-5"
    LLM_HEAVY_MODEL: str = "anthropic/claude-sonnet-4-6"   # For complex queries
    LLM_ROUTER_MODEL: str = "anthropic/claude-haiku-4-5"
    LLM_TIMEOUT_SECONDS: int = 30
    MAX_USER_INPUT_CHARS: int = 2000

    # Vector DB
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "healthbridge"
    PINECONE_REGION: str = "us-east-1"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    PINECONE_INDEX_DIMENSION: int = 1024

    # Web Search
    TAVILY_API_KEY: str

    # Audio (all optional — fallbacks active if empty)
    YARNGPT_API_KEY: str = ""
    HUGGINGFACE_TOKEN: str = ""
    NATLAS_API_KEY: str = ""

    # GCP
    GCP_PROJECT_ID: str
    GCS_BUCKET_NAME: str = "healthbridge-assets"
    FIRESTORE_DATABASE: str = "(default)"

    # App behaviour
    SUPPORTED_LANGUAGES: list[str] = Field(
        default=["en", "yo", "ig", "ha", "pidgin"]
    )
    RATE_LIMIT_MESSAGES_PER_MINUTE: int = 20
    CONVERSATION_HISTORY_TURNS: int = 5
    SEMANTIC_CACHE_THRESHOLD: float = 0.92
    CACHE_TTL_DAYS: int = 7
    MIN_RETRIEVAL_SCORE_DEFAULT: float = 0.6
    HYDE_FALLBACK_THRESHOLD: float = 0.5

settings = Settings()
```

### .env.example

```bash
# Twilio WhatsApp
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # Twilio Console
TWILIO_AUTH_TOKEN=                                      # Twilio Console → Auth Token
TWILIO_WHATSAPP_FROM=+14155238886                       # Sandbox: +14155238886
TWILIO_WEBHOOK_URL=https://healthbridge-webhook-xxx-uc.a.run.app/webhook

# LLM via OpenRouter
OPENROUTER_API_KEY=                # openrouter.ai → Keys
LLM_PRIMARY_MODEL=anthropic/claude-haiku-4-5
LLM_HEAVY_MODEL=anthropic/claude-sonnet-4-6
LLM_ROUTER_MODEL=anthropic/claude-haiku-4-5

# Pinecone
PINECONE_API_KEY=                  # pinecone.io → API Keys
PINECONE_INDEX_NAME=healthbridge
PINECONE_REGION=us-east-1

# Web Search
TAVILY_API_KEY=                    # tavily.com → API

# Audio (all optional — fallbacks activate if empty)
YARNGPT_API_KEY=
HUGGINGFACE_TOKEN=
NATLAS_API_KEY=

# GCP
GCP_PROJECT_ID=
GCS_BUCKET_NAME=healthbridge-assets
```

### diseases.yaml — complete structure

```yaml
diseases:
  tb:
    name: "Tuberculosis"
    short_name: "TB"
    aliases:
      - "TB"
      - "tuberculosis"
      - "lung disease"
      - "rifampicin"
      - "isoniazid"
      - "DOTS"
      - "MDR-TB"
      - "pulmonary"
    pinecone_namespace: "tb"
    kb_gcs_path: "knowledge-bases/tb/TB_knowledge_base.zip"
    chunk_size_hint: 400        # Target tokens per semantic chunk
    min_retrieval_score: 0.6
    search_domains:
      - "who.int"
      - "cdc.gov"
      - "stoptb.org"
      - "tbfacts.org"
      - "nhs.uk"
      - "mayoclinic.org"
      - "iuatld.org"
    emergency_keywords:
      - "coughing blood"
      - "hemoptysis"
      - "difficulty breathing"
      - "chest pain"
    system_prompt_extra: |
      Always emphasise treatment adherence — TB is curable when the full course
      is completed. Never suggest stopping treatment early. For drug-resistant TB
      (MDR-TB, XDR-TB), always emphasise the need for specialist care.
    enabled: true

  hiv:
    name: "HIV / AIDS"
    short_name: "HIV"
    aliases:
      - "HIV"
      - "AIDS"
      - "antiretroviral"
      - "ARV"
      - "ART"
      - "CD4"
      - "viral load"
      - "PrEP"
      - "PEP"
    pinecone_namespace: "hiv"
    kb_gcs_path: "knowledge-bases/hiv/HIV_knowledge_base.zip"
    chunk_size_hint: 350
    min_retrieval_score: 0.6
    search_domains:
      - "who.int"
      - "unaids.org"
      - "aidsinfo.nih.gov"
      - "cdc.gov"
      - "aidsmap.com"
      - "hiv.gov"
    emergency_keywords:
      - "PEP"
      - "post-exposure"
      - "needle stick"
      - "unprotected exposure"
    system_prompt_extra: |
      Always recommend HIV testing and counselling. Maintain sensitivity around
      stigma — HIV is manageable with treatment and people live full, healthy lives.
      PEP (post-exposure prophylaxis) must be started within 72 hours of exposure —
      always treat PEP queries as urgent. Emphasise privacy and confidentiality.
    enabled: false

  malaria:
    name: "Malaria"
    short_name: "Malaria"
    aliases:
      - "malaria"
      - "Plasmodium"
      - "ACT"
      - "artemisinin"
      - "RDT"
      - "rapid test"
      - "mosquito"
      - "falciparum"
    pinecone_namespace: "malaria"
    kb_gcs_path: "knowledge-bases/malaria/malaria_knowledge_base.zip"
    chunk_size_hint: 400
    min_retrieval_score: 0.6
    search_domains:
      - "who.int"
      - "cdc.gov"
      - "rollbackmalaria.org"
      - "malariaconsortium.org"
      - "nmcp.gov.ng"
    emergency_keywords:
      - "convulsions"
      - "unconscious"
      - "severe fever"
      - "cerebral malaria"
      - "vomiting blood"
    system_prompt_extra: |
      Always urge rapid diagnostic testing (RDT) before treatment — do not
      treat presumptively without a test. Emphasise danger signs that require
      immediate hospital care (convulsions, loss of consciousness, inability to
      drink). For children under 5 and pregnant women, always flag higher risk.
    enabled: false
```

---

## 15. Data Models

### Firestore: `users/{phone_number}`

```json
{
  "phone_number": "+2348012345678",
  "language_code": "yo",
  "audio_enabled": false,
  "first_seen_at": "Timestamp",
  "last_seen_at": "Timestamp",
  "message_count": 47,
  "rate_limit": {
    "window_start": "Timestamp",
    "count": 3
  }
}
```

### Firestore: `conversations/{phone_number}/turns/{turn_id}`

```json
{
  "turn_id": "uuid-v4",
  "phone_number": "+2348012345678",
  "timestamp": "Timestamp",
  "input_type": "text",
  "user_message": "What are the side effects of rifampicin?",
  "user_language": "en",
  "detected_diseases": ["tb"],
  "query_intent": "drug_interaction",
  "retrieval_score": 0.82,
  "used_web_search": false,
  "cache_hit": false,
  "bot_response_english": "Rifampicin can cause...",
  "sources": [
    {"doc": "WHO TB Guidelines 2022", "url": "https://who.int/...", "section": "Chapter 4"}
  ],
  "llm_model": "anthropic/claude-haiku-4-5",
  "latency_ms": 2840
}
```

### Firestore: `feedback/{feedback_id}`

```json
{
  "feedback_id": "uuid-v4",
  "phone_number": "+2348012345678",
  "timestamp": "Timestamp",
  "message": "The malaria dosage info was unclear",
  "conversation_turn_id": "uuid-v4"
}
```

### Pinecone chunk metadata schema

```json
{
  "text": "Full chunk text (stored for retrieval display)",
  "disease": "tb",
  "doc_id": "who_tb_guidelines_2022",
  "source_doc": "WHO Consolidated Guidelines on TB, Module 4 (2022)",
  "source_url": "https://www.who.int/publications/i/item/9789240048928",
  "source_type": "guideline",
  "section": "Chapter 4: Treatment of drug-susceptible TB",
  "page_number": 45,
  "chunk_type": "treatment",
  "chunk_index": 12,
  "language": "en"
}
```

### Pinecone cache entry schema (namespace: `response-cache`)

```json
{
  "id": "sha256(english_query|disease_ids)",
  "values": "[1024-dim bge-m3 embedding]",
  "metadata": {
    "english_query": "What are the symptoms of tuberculosis?",
    "disease_ids": "tb",
    "query_intent": "symptoms",
    "english_response": "The main symptoms of tuberculosis include...",
    "sources_json": "[{\"doc\": \"WHO Guidelines\", \"url\": \"...\"}]",
    "confidence": "high",
    "created_at": 1719100000,
    "expires_at": 1719705000,
    "hit_count": 0
  }
}
```

---

## 16. API Design

### Webhook Service endpoints

| Method · Path | Purpose | Auth |
|---------------|---------|------|
| `POST /webhook` | Receive Twilio WhatsApp events | `X-Twilio-Signature` HMAC-SHA1 |
| `GET /webhook` | (Optional) Twilio status page | Internal |
| `GET /health` | Cloud Run liveness probe | None |
| `GET /ready` | Readiness — checks Pub/Sub + Twilio connectivity | None |

### Twilio webhook payload (form-encoded)

| Field | Description |
|-------|-------------|
| `AccountSid` | Twilio Account SID (validate == settings.TWILIO_ACCOUNT_SID) |
| `Body` | Text message content |
| `From` | Sender: `whatsapp:+2348012345678` |
| `To` | Your Twilio number: `whatsapp:+14155238886` |
| `MessageSid` | Unique Twilio message ID |
| `NumMedia` | Number of media attachments |
| `MediaUrl0` | URL of first media file (audio, image) |
| `MediaContentType0` | MIME type: `audio/ogg`, `audio/mp4`, `image/jpeg`, etc. |

### Twilio signature validation

```python
from twilio.request_validator import RequestValidator

validator = RequestValidator(settings.TWILIO_AUTH_TOKEN)
is_valid = validator.validate(
    url=settings.TWILIO_WEBHOOK_URL,
    params=dict(request.form),
    signature=request.headers.get("X-Twilio-Signature", "")
)
if not is_valid:
    raise HTTPException(status_code=401, detail="Invalid signature")
```

### Key function signatures

```python
# src/healthbridgeai/core/services/pipeline.py
async def process_message(parsed: ParsedMessage) -> None:
    """Full pipeline: cache check → route → retrieve → generate → cite → reply."""

# src/healthbridgeai/core/services/router.py
async def route(query: str, history: list[ConversationTurn]) -> RouteResult:
    """Return disease IDs, query intent, emergency flag, personal flag."""

# src/healthbridgeai/core/services/rag.py
async def retrieve(query: str, route: RouteResult) -> RetrievalResult:
    """Hybrid search → rerank → HyDE fallback → Tavily fallback."""

# src/healthbridgeai/core/services/generator.py
async def generate(query: str, context: RetrievalResult, route: RouteResult,
                   history: list[ConversationTurn], lang: str) -> BotResponse:
    """LLM generation with citation grounding. Returns response + sources."""

# src/healthbridgeai/core/ports/cache.py (Protocol)
async def get(self, query_embedding: list[float], disease_ids: list[str]) -> CachedResponse | None: ...
async def set(self, query_embedding: list[float], response: BotResponse) -> None: ...
async def invalidate_disease(self, disease_id: str) -> None: ...
```

### Sending a reply via Twilio REST API

```python
from twilio.rest import Client

client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

# Text reply
client.messages.create(
    body=response_text,
    from_=f"whatsapp:{settings.TWILIO_WHATSAPP_FROM}",
    to=f"whatsapp:{user_phone}"
)

# Audio reply (GCS signed URL must be publicly accessible for Twilio)
client.messages.create(
    body="",
    from_=f"whatsapp:{settings.TWILIO_WHATSAPP_FROM}",
    to=f"whatsapp:{user_phone}",
    media_url=[gcs_signed_url]
)
```

---

## 17. Security Design

### Twilio webhook authentication
Every POST to `/webhook` must carry a valid `X-Twilio-Signature` header. Computed as HMAC-SHA1 of the full webhook URL + sorted POST parameters, keyed by `TWILIO_AUTH_TOKEN`. Any request failing this check returns 401 immediately — no Pub/Sub publish, no processing.

Additionally: verify `AccountSid` in the payload matches `settings.TWILIO_ACCOUNT_SID`.

### Secret management
All credentials stored exclusively in GCP Secret Manager. Cloud Run services access secrets via dedicated service accounts. Secrets are never logged, never appear in environment dumps, and never exist in source control. `.env` is gitignored; `.env.example` contains only key names.

### Rate limiting
Per-phone-number rate limiting enforced in Firestore before any LLM call. If a user exceeds 20 messages per 60-second sliding window, they receive: *"You have sent too many messages. Please wait a minute."* Counter resets automatically.

### Input validation
- User text truncated at 2,000 characters
- Language codes validated against `settings.SUPPORTED_LANGUAGES`
- Media file MIME type checked before download (only `audio/*` accepted for transcription)
- Media file size checked before download (reject if > 16MB)
- Pub/Sub message schema validated before processing (structured parse with error handling)

### Cache safety
Certain response types are never cached (see Section 9.5):
- Emergency responses
- Drug dosage responses
- Low-confidence responses
- Personalised medical advice

### No PII in logs
Phone numbers are hashed (SHA-256 truncated to 8 chars) in all log entries. Full phone numbers stored only in Firestore (GCP-encrypted at rest). Message content is never logged — only metadata.

### HTTPS enforcement
Cloud Run enforces HTTPS on all ingress. Twilio sends webhooks over HTTPS only. Twilio will reject any webhook URL that is not HTTPS.

---

## 18. Monitoring & Observability

### Structured log format

```json
{
  "severity": "INFO",
  "message": "message_processed",
  "user_hash": "a3f9c12b",
  "input_type": "text",
  "detected_language": "yo",
  "detected_diseases": ["tb"],
  "query_intent": "treatment",
  "retrieval_score": 0.78,
  "used_web_search": false,
  "cache_hit": true,
  "llm_model": "anthropic/claude-haiku-4-5",
  "latency_ms": 410,
  "trace_id": "pubsub-message-id"
}
```

### Custom Cloud Monitoring metrics

| Metric | Type | Alert threshold |
|--------|------|----------------|
| `messages_processed_per_minute` | Counter | Alert if > 500/min |
| `llm_latency_p95_ms` | Distribution | Alert if p95 > 15,000ms |
| `retrieval_score_avg` | Gauge | Alert if 1-hr avg < 0.4 |
| `web_search_fallback_rate` | Gauge | Alert if > 40% |
| `cache_hit_rate` | Gauge | Informational (target: > 40%) |
| `error_rate` | Counter | Alert if > 5 errors/min |
| `emergency_queries_count` | Counter | Alert if > 0 (review immediately) |
| `disease_query_distribution` | Counter by disease | Dashboard only |
| `language_distribution` | Counter by language | Dashboard only |
| `pubsub_dead_letter_count` | Counter | Alert if > 0 |

---

## 19. Implementation Roadmap

### Phase 1 — Foundation & Bug Fixes (Weeks 1–2)

All 31 bugs resolved. All removed files deleted. FastAPI replacing Flask.

- Delete: `app-st.py`, `main.py`, `render.yaml`, `config.py`, `config_manager.py`, `utills.py`, `session_manager.py`, `audio_handler.py`, `install_packages.py`, `response_strategies.py`
- Set up `pyproject.toml` and `src/` layout
- Create `src/healthbridgeai/config/settings.py` (Pydantic BaseSettings) — resolves all 6 critical config bugs
- Create Clean Architecture skeleton: `core/`, `infrastructure/`, `api/`
- Implement `infrastructure/messaging/twilio.py` with HMAC-SHA1 signature validation
- Fix `audio/synthesizer.py`: remove misplaced docstring, UUID-based GCS output paths, fix unreachable code
- Implement `core/services/language.py` with ≥4-word rule + LLM fallback
- Add retry logic (`tenacity`) to all external API calls
- Add input validation: 2,000-char limit, language code whitelist, media type/size check
- Write `.env.example`
- Set up single structured logger; remove all `basicConfig()` calls
- Replace Flask with FastAPI + Uvicorn
- Verify TB RAG pipeline works end-to-end with Twilio sandbox

### Phase 2 — Advanced RAG + Citations (Weeks 3–4)

- Upgrade embeddings to BAAI/bge-m3 (1024-dim) — update Pinecone index dimension
- Re-index TB KB with:
  - Semantic chunking (SemanticChunker)
  - Chunk type labelling (LLM call during ingestion)
  - Full source metadata per chunk (doc title, URL, section, page)
- Implement hybrid search in `infrastructure/vector_store/pinecone.py`:
  - BM25Encoder from `pinecone-text` for sparse vectors
  - Hybrid query with alpha = 0.7
- Implement cross-encoder re-ranking (FlashRank `ms-marco-MiniLM-L-12-v2`)
- Implement HyDE fallback in `core/services/rag.py`
- Implement source citation assembly in `core/services/generator.py`:
  - Numbered context documents in prompt
  - LLM structured output: `LLMResponse` with `sources_used` list
  - WhatsApp citation format with sources + medical disclaimer
- Run RAGAS evaluation on TB KB: all four metrics must meet target thresholds
- Replace DuckDuckGo with Tavily in `infrastructure/search/tavily.py`:
  - Domain filtering per disease
  - Result validation (domain check + score check)
  - Sources formatted for citation

### Phase 3 — Multi-Disease + Semantic Cache (Weeks 5–6)

- Implement `config/diseases.yaml` schema and `DiseaseRegistry`
- Implement `core/services/router.py`: two-dimensional routing (disease + intent)
  - Alias matching (zero cost)
  - LLM classification with `RouteResult` structured output
  - Emergency detection → immediate emergency response + CRITICAL log
  - Multi-disease namespace merging
- Implement intent-filtered retrieval (Pinecone metadata filter on `chunk_type`)
- Implement intent-specific response formats in `generator.py`
- Implement semantic response cache in `infrastructure/cache/semantic.py`:
  - Pinecone `response-cache` namespace
  - Cache lookup before pipeline
  - Cache write after pipeline
  - Cache invalidation on KB update
  - Never-cache rules (emergency, dosage, personal, low-confidence)
- Implement conversation context: `firestore.get_recent_turns()`, `save_turn()`
- Inject last 5 conversation turns into LLM prompt
- Add user command handling: `language`, `audio`, `about`, `help`, `feedback`
- Implement `scripts/warm_cache.py` for top-50 common questions per disease
- Integration tests: routing, retrieval, citation, cache hit/miss

### Phase 4 — GCP Migration (Weeks 7–8)

- Provision GCP resources via `scripts/setup_gcp.sh`
- Migrate audio to GCS with 24h lifecycle rules
- Migrate KB ZIP loading to GCS download-on-startup
- Create `processor/main.py` (FastAPI Pub/Sub push subscriber)
- Write `deploy/Dockerfile` and `deploy/Dockerfile.processor` (multi-stage)
- Write `deploy/cloudbuild.yaml`: ruff → mypy → pytest → build → push → deploy
- Write `deploy/cloudrun.yaml`: secrets from Secret Manager, min instances, timeouts
- Register Cloud Run webhook URL with Twilio (update webhook in Twilio Console)
- End-to-end smoke test on production Twilio WhatsApp number

### Phase 5 — Observability, Hardening & Disease Expansion (Weeks 9–10)

- Implement structured JSON logging throughout
- Create Cloud Monitoring dashboard with 10 custom metrics + alerting policies
- Load test: 100 concurrent messages, verify p95 latency < 15s, no errors
- Implement per-user rate limiting with sliding window in Firestore
- Implement Pub/Sub dead-letter alerting
- Upload HIV KB to GCS; run RAGAS evaluation; set `enabled: true` if metrics pass
- Upload Malaria KB to GCS; run RAGAS evaluation; set `enabled: true` if metrics pass
- Phone number hashing in all log entries
- Run `scripts/warm_cache.py` for HIV and Malaria
- Final security review: Twilio signature rejection test, rate limit test, secret rotation test

---

## 20. Migration Plan

### File-to-file mapping

| Current file | Action | Replaced by |
|-------------|--------|-------------|
| `app.py` | Rewrite | `src/healthbridgeai/main.py` + `api/webhook.py` |
| `app-st.py` | Delete | — |
| `main.py` | Delete | — |
| `config.py` | Delete | `src/healthbridgeai/config/settings.py` |
| `modules/config_manager.py` | Delete | `config/settings.py` |
| `render.yaml` | Delete | `deploy/cloudrun.yaml` |
| `modules/utills.py` | Delete | `infrastructure/messaging/twilio.py` + `api/webhook.py` |
| `modules/utils.py` | Merge | `infrastructure/storage/gcs.py` |
| `modules/session_manager.py` | Delete | — |
| `modules/response_strategies.py` | Delete | `core/services/generator.py` |
| `modules/audio_handler.py` | Delete | Direct import of transcriber/synthesizer |
| `modules/audio_transcriber.py` | Fix + move | `infrastructure/audio/transcriber.py` |
| `modules/audio_synthesizer.py` | Fix + move | `infrastructure/audio/synthesizer.py` |
| `modules/llm_handler.py` | Refactor | `core/services/generator.py` + `infrastructure/llm/openrouter.py` |
| `modules/knowledge_base_manager.py` | Refactor | `core/services/rag.py` + `infrastructure/vector_store/pinecone.py` |
| `modules/vector_store_manager.py` | Refactor | `infrastructure/vector_store/pinecone.py` |
| `modules/language_utils.py` | Refactor | `core/services/language.py` |
| `modules/language_service.py` | Merge | `core/services/language.py` |
| `modules/user_preferences.py` | Delete | `infrastructure/storage/firestore.py` |
| `modules/cache_manager.py` | Refactor | `infrastructure/cache/semantic.py` |
| `modules/exceptions.py` | Keep + expand | `src/healthbridgeai/core/exceptions.py` |
| `populate_kb.py` | Refactor | `scripts/populate_kb.py` (--disease flag, semantic chunking, GCS) |
| `install_packages.py` | Delete | — |
| `Dockerfile` | Rewrite | `deploy/Dockerfile` (multi-stage, non-root user) |
| `data/TB_knowledge_base.zip` | Move | `gs://healthbridge-assets/knowledge-bases/tb/` |
| — | New | `scripts/evaluate_kb.py` (RAGAS evaluation) |
| — | New | `scripts/warm_cache.py` (pre-warm semantic cache) |
| — | New | `scripts/setup_gcp.sh` |

### Pinecone index migration

The current TB index uses 384-dimensional MiniLM vectors. BAAI/bge-m3 uses 1024 dimensions. The index must be recreated:
1. Create new Pinecone index `healthbridge` (dimension=1024, metric=dotproduct, serverless)
2. Re-ingest the TB KB using the new pipeline (semantic chunking, bge-m3 embeddings, BM25 sparse)
3. Run RAGAS evaluation — confirm quality ≥ old baseline before deleting old index
4. Delete old index

### Twilio number setup

If using Twilio sandbox (development):
- Enrol developers by having them send "join [sandbox code]" to the sandbox number
- Configure sandbox webhook URL in Twilio Console → Messaging → Try it out → Send a WhatsApp message → Sandbox settings

If using an approved Twilio WhatsApp Business number (production):
- Apply for WhatsApp Business through Twilio Console
- Approval typically takes 2–5 business days
- No change to phone number visible to users

### WhatsApp continuity

If migrating from an existing Meta direct API setup:
- The WhatsApp phone number itself is preserved
- The webhook delivery changes (Meta → Twilio) during Twilio onboarding
- This is a one-time operational change, not a user-visible change
- Users see the same number; only the backend delivery path changes

> **Before going live:** Test every message type in the Twilio sandbox: text query, audio query, command (`language yo`), multi-disease query, emergency keyword. All must work correctly before switching production traffic.
