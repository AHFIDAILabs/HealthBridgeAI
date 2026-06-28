"""
setup_pinecone.py — Create (or verify) the HealthBridgeAI Pinecone index.

Run this ONCE before populate_kb.py.  Safe to re-run — it skips creation
if the index already exists.

Usage:
    python scripts/setup_pinecone.py

    # Tear down and recreate (WARNING: deletes all vectors)
    python scripts/setup_pinecone.py --recreate

Environment: reads the same .env as the application (PINECONE_API_KEY,
PINECONE_INDEX_NAME, PINECONE_REGION, PINECONE_INDEX_DIMENSION).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import structlog

log = structlog.get_logger(__name__)

# ── Logging ───────────────────────────────────────────────────────────────────

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


# ── Index spec ────────────────────────────────────────────────────────────────
#
# BAAI/bge-m3 produces 1024-dim dense vectors.
# Pinecone dotproduct metric is required for hybrid search (dense + sparse).
# Namespaces used (one per disease + semantic cache):
#   tb              — TB knowledge base
#   hiv             — HIV/AIDS knowledge base
#   malaria         — Malaria knowledge base
#   response-cache  — SemanticCache
#
# The index must be created with dotproduct + dimension=1024 for hybrid
# search to work.  cosine similarity cannot be used with sparse vectors.

INDEX_METRIC = "dotproduct"
INDEX_CLOUD = "aws"


# ── Main ──────────────────────────────────────────────────────────────────────

def main(recreate: bool = False) -> None:
    from pinecone import Pinecone, ServerlessSpec

    from healthbridgeai.config.settings import settings

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index_name = settings.PINECONE_INDEX_NAME
    dimension = settings.PINECONE_INDEX_DIMENSION
    region = settings.PINECONE_REGION

    existing = [idx.name for idx in pc.list_indexes()]

    # ── Teardown (--recreate only) ────────────────────────────────────────────
    if recreate and index_name in existing:
        log.warning(
            "setup_pinecone.deleting_index",
            index=index_name,
            hint="all vectors will be lost",
        )
        confirm = input(f"Type the index name to confirm deletion [{index_name}]: ").strip()
        if confirm != index_name:
            log.info("setup_pinecone.delete_cancelled")
            sys.exit(0)
        pc.delete_index(index_name)
        log.info("setup_pinecone.deleted", index=index_name)
        existing = []

    # ── Create ────────────────────────────────────────────────────────────────
    if index_name not in existing:
        log.info(
            "setup_pinecone.creating",
            index=index_name,
            dimension=dimension,
            metric=INDEX_METRIC,
            cloud=INDEX_CLOUD,
            region=region,
        )
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=INDEX_METRIC,
            spec=ServerlessSpec(cloud=INDEX_CLOUD, region=region),
        )

        # Wait for index to be ready
        log.info("setup_pinecone.waiting_for_ready")
        for attempt in range(60):
            status = pc.describe_index(index_name).status
            if status.get("ready"):
                break
            log.info("setup_pinecone.not_ready_yet", attempt=attempt + 1)
            time.sleep(5)
        else:
            log.error("setup_pinecone.timeout_waiting_for_ready")
            sys.exit(1)

        log.info("setup_pinecone.created_ok", index=index_name)
    else:
        log.info("setup_pinecone.already_exists", index=index_name)

    # ── Describe ──────────────────────────────────────────────────────────────
    info = pc.describe_index(index_name)
    log.info(
        "setup_pinecone.index_info",
        name=info.name,
        dimension=info.dimension,
        metric=info.metric,
        status=info.status,
    )

    log.info(
        "setup_pinecone.done",
        next_step="Run  python scripts/populate_kb.py  to index disease knowledge",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the HealthBridgeAI Pinecone index"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the index (WARNING: destroys all vectors — prompts for confirmation)",
    )
    args = parser.parse_args()

    _configure_logging()
    main(recreate=args.recreate)
