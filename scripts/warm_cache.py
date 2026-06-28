"""
warm_cache.py — Pre-populate the semantic response cache.

For each enabled disease, sends the top-N most common questions through
the full pipeline.  The resulting responses are stored in the Pinecone
semantic cache so first real users get sub-second responses.

Questions are loaded from  data/cache_warmup/<disease_id>.yaml:

    questions:
      - "What are the symptoms of TB?"
      - "How is tuberculosis spread?"
      ...

Usage:
    python scripts/warm_cache.py                # all enabled diseases, 50 Qs each
    python scripts/warm_cache.py --disease tb --n 20
    python scripts/warm_cache.py --dry-run      # print questions, no API calls
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
import textwrap
from pathlib import Path
from typing import Optional

import yaml

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import structlog

log = structlog.get_logger(__name__)

_WARMUP_DIR = _REPO_ROOT / "data" / "cache_warmup"

# Synthetic caller for cache warmup — hashed in logs, never stored in plain text
_WARMUP_PHONE = "+2340000000001"


def _configure_logging() -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _load_questions(disease_id: str, n: int) -> list[str]:
    path = _WARMUP_DIR / f"{disease_id}.yaml"
    if not path.exists():
        log.warning("warm_cache.no_question_file", disease=disease_id, path=str(path))
        return []
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return (data.get("questions") or [])[:n]


async def _warm_disease(disease_id: str, questions: list[str], dry_run: bool) -> int:
    if dry_run:
        for q in questions:
            print(f"  [{disease_id}] {q}")
        return len(questions)

    from healthbridgeai.config import get_disease_registry
    from healthbridgeai.core.models.message import InboundMessage, MessageType
    from healthbridgeai.core.services import (
        LanguageService, MessagePipeline, RAGService, ResponseGenerator, RouterService,
    )
    from healthbridgeai.infrastructure.cache import SemanticCache
    from healthbridgeai.infrastructure.llm import OpenRouterClient
    from healthbridgeai.infrastructure.search import TavilyAdapter
    from healthbridgeai.infrastructure.storage import (
        FirestoreConversationStore, FirestoreUserStore,
    )
    from healthbridgeai.infrastructure.vector_store import PineconeAdapter

    llm = OpenRouterClient()
    vector_store = PineconeAdapter()
    web_search = TavilyAdapter()
    user_store = FirestoreUserStore()
    conv_store = FirestoreConversationStore()
    cache = SemanticCache()
    registry = get_disease_registry()

    pipeline = MessagePipeline(
        language_svc=LanguageService(llm),
        router_svc=RouterService(llm, registry),
        rag_svc=RAGService(llm, vector_store, web_search, registry),
        generator=ResponseGenerator(llm),
        llm=llm,
        user_store=user_store,
        conv_store=conv_store,
        cache=cache,
    )

    warmed = 0
    for i, question in enumerate(questions):
        msg_id = hashlib.sha256(f"{disease_id}:{question}".encode()).hexdigest()[:12]
        log.info(
            "warm_cache.running",
            disease=disease_id,
            i=i + 1,
            total=len(questions),
            question=question[:70],
        )
        try:
            message = InboundMessage(
                message_id=f"warmup-{msg_id}",
                from_number=_WARMUP_PHONE,
                type=MessageType.TEXT,
                text=question,
                timestamp=0,
            )
            bot = await pipeline.process(message)
            status = "cache_hit" if bot.cache_hit else "cached_new"
            log.info("warm_cache.done", disease=disease_id, status=status, confidence=bot.confidence)
            warmed += 1
        except Exception as exc:
            log.error("warm_cache.error", disease=disease_id, question=question[:60], error=str(exc))

    return warmed


async def run(disease_filter: Optional[str], n: int, dry_run: bool) -> None:
    from healthbridgeai.config import get_disease_registry

    registry = get_disease_registry()
    diseases = [d for d in registry.all_diseases() if d.enabled]
    if disease_filter:
        diseases = [d for d in diseases if d.id == disease_filter]
        if not diseases:
            log.error("warm_cache.disease_not_found", filter=disease_filter)
            sys.exit(1)

    log.info("warm_cache.start", diseases=[d.id for d in diseases], n=n, dry_run=dry_run)

    grand_total = 0
    for disease in diseases:
        questions = _load_questions(disease.id, n)
        if not questions:
            log.warning("warm_cache.no_questions", disease=disease.id,
                        hint=f"Create data/cache_warmup/{disease.id}.yaml")
            continue
        log.info("warm_cache.disease_start", disease=disease.id, count=len(questions))
        warmed = await _warm_disease(disease.id, questions, dry_run)
        grand_total += warmed
        log.info("warm_cache.disease_done", disease=disease.id, warmed=warmed)

    log.info("warm_cache.complete", grand_total=grand_total, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-populate the HealthBridgeAI semantic response cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python scripts/warm_cache.py
              python scripts/warm_cache.py --disease tb --n 20
              python scripts/warm_cache.py --dry-run
        """),
    )
    parser.add_argument("--disease", help="Disease ID to warm (default: all enabled)")
    parser.add_argument("--n", type=int, default=50, help="Questions per disease (default 50)")
    parser.add_argument("--dry-run", action="store_true", help="Print questions without running pipeline")
    args = parser.parse_args()

    _configure_logging()
    asyncio.run(run(args.disease, args.n, args.dry_run))


if __name__ == "__main__":
    main()
