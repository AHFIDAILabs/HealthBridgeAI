# HealthBridgeAI

A multilingual WhatsApp AI companion for integrated infectious disease management in Nigeria and West Africa. Patients and healthcare workers can ask expert-level questions about HIV/AIDS, Tuberculosis, Malaria, and other diseases — and receive source-cited, trustworthy answers in English, Yoruba, Igbo, Hausa, or Pidgin.

> Built by [AHFID AI Labs](https://ahfid.org) · Powered by Claude (Anthropic) via OpenRouter · Delivered over WhatsApp via Twilio

---

## Overview

HealthBridgeAI is a conversational AI platform designed for the general population. Users interact entirely through WhatsApp — no app to install, no account to create. The system combines a curated medical knowledge base with a hybrid retrieval pipeline, large language models, and source citation to ensure every response is grounded, verifiable, and safe.

**Key capabilities**

- Ask questions about symptoms, treatment, prevention, complications, drug interactions, and more
- Receive answers with numbered source citations (WHO, CDC, and other authoritative bodies)
- Send voice messages — the bot transcribes, responds, and optionally replies in audio
- Converse in English, Yoruba, Igbo, Hausa, or Nigerian Pidgin
- Emergency keyword detection routes users to emergency services instantly

---

## Diseases

| Disease | Status | Knowledge Base |
|---------|--------|----------------|
| Tuberculosis (TB) | Active | WHO, CDC, StopTB, IUATLD |
| HIV / AIDS | Planned | WHO, UNAIDS, AIDSINFO NIH |
| Malaria | Planned | WHO, CDC, NMCP Nigeria |

Additional diseases can be added through configuration only — no code changes required.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| WhatsApp delivery | Twilio WhatsApp Business API |
| Web framework | FastAPI + Uvicorn (async) |
| LLM | Claude Haiku 4.5 / Sonnet 4.6 via OpenRouter |
| Embeddings | BAAI/bge-m3 (1024-dim, 100+ languages) |
| Vector store | Pinecone (hybrid dense + BM25 sparse search) |
| Re-ranking | FlashRank cross-encoder |
| Web search fallback | Tavily (trusted domains only) |
| Async messaging | Google Cloud Pub/Sub |
| Persistence | Google Cloud Firestore |
| File storage | Google Cloud Storage |
| Deployment | Google Cloud Run (two services) |
| Secrets | Google Secret Manager |

---

## Architecture

The system uses a Clean Hexagonal Architecture with two Cloud Run services:

```
User (WhatsApp)
      │ HTTPS
      ▼
Twilio API ──POST /webhook──► Webhook Service (Cloud Run)
                                │  Validates signature → Pub/Sub → TwiML 200
                                ▼ Pub/Sub push
                            Processor Service (Cloud Run)
                                │  Semantic cache lookup (Pinecone)
                                │  Disease + intent routing
                                │  Hybrid RAG → re-rank → HyDE → Tavily
                                │  LLM generation + source citations
                                └──► Twilio REST API reply to user
```

Source code layers follow dependency inversion — the `core/` domain layer has zero external dependencies:

```
src/healthbridgeai/
├── core/           # Domain logic: models, ports (interfaces), services
├── infrastructure/ # Adapters: Twilio, Pinecone, OpenRouter, Tavily, GCS, Firestore
├── api/            # HTTP layer: webhook endpoint, health check
└── config/         # Pydantic settings, diseases.yaml registry
```

---

## Project Structure

```
healthbridgeai/
├── src/healthbridgeai/     # Application package (src layout)
├── processor/              # Pub/Sub push subscriber (separate Cloud Run service)
├── scripts/                # KB ingestion, RAGAS evaluation, cache warming
├── deploy/                 # Dockerfiles, Cloud Build, Cloud Run configs
├── tests/                  # Unit (mocked ports) and integration tests
├── data/                   # Local KB ZIPs (uploaded to GCS for production)
├── pyproject.toml          # Dependencies and tooling config
└── .env.example            # All required environment variables documented
```

---

## Getting Started (Local Development)

### Prerequisites

- Python 3.11+
- A [Twilio account](https://twilio.com) with the WhatsApp sandbox enabled
- API keys: Twilio, OpenRouter, Pinecone, Tavily
- GCP project (Firestore, GCS, Pub/Sub provisioned via `scripts/setup_gcp.sh`)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/AHFIDAILabs/HealthBridgeAI.git
cd HealthBridgeAI

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 3. Configure environment variables
cp .env.example .env
# Edit .env and fill in all required values

# 4. Ingest the TB knowledge base into Pinecone
python scripts/populate_kb.py --disease tb

# 5. Start the webhook service (uses Twilio sandbox)
uvicorn src.healthbridgeai.main:app --reload --port 8000

# 6. Expose locally for Twilio webhook (e.g. using ngrok)
ngrok http 8000
# Set the ngrok URL as TWILIO_WEBHOOK_URL in .env and in Twilio Console sandbox settings
```

### Running tests

```bash
pytest                          # Full suite with coverage
pytest tests/unit/              # Unit tests only (no external services)
pytest tests/integration/       # Requires real Pinecone test namespace
```

---

## Configuration

All environment variables are documented in [.env.example](.env.example).

Disease configuration (aliases, trusted search domains, emergency keywords, system prompt additions) lives in [src/healthbridgeai/config/diseases.yaml](src/healthbridgeai/config/diseases.yaml). Adding a new disease requires only a YAML entry and a KB upload — no Python changes.

---

## Deployment

Deployment targets Google Cloud Run via Cloud Build. On every push to `main`:

1. `ruff` (lint) → `mypy` (type check) → `pytest` (≥ 80% coverage)
2. Docker images built and pushed to Artifact Registry
3. Both Cloud Run services deployed with zero-downtime rollout

See [deploy/](deploy/) for Dockerfiles and pipeline configuration.

---

## License

[MIT](LICENSE) — AHFID AI Labs, 2025–2026.
