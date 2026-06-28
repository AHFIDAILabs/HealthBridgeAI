"""
populate_kb.py — Populate HealthBridgeAI Pinecone knowledge base.

For each enabled disease in diseases.yaml, reads the source corpus from GCS
(or a local ZIP override), parses HTML/text/PDF content, chunks semantically
with LangChain SemanticChunker, embeds with BGE-M3 via OpenRouterClient, and
upserts to the correct Pinecone namespace.

Usage:
    # From GCS (production — reads GCS_BUCKET_NAME/corpora/<disease_id>.zip)
    python scripts/populate_kb.py

    # Local ZIP override for a single disease
    python scripts/populate_kb.py --disease tb --zip /path/to/tb_corpus.zip

    # Force re-index even if namespace already has vectors
    python scripts/populate_kb.py --force

    # Dry-run (parse + chunk + embed but don't upsert)
    python scripts/populate_kb.py --dry-run

Environment: expects the same .env as the application (settings.py).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import logging
import re
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import structlog
from langchain.text_splitter import RecursiveCharacterTextSplitter

log = structlog.get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800          # characters per chunk (≈200 tokens for bge-m3)
CHUNK_OVERLAP = 150
UPSERT_BATCH_SIZE = 50    # Pinecone upsert batch (each item has 1024-dim dense + sparse)
MIN_CHUNK_CHARS = 120     # discard very short fragments

# Supported source file extensions inside the ZIP
_PARSEABLE = {".txt", ".md", ".html", ".htm", ".pdf", ".docx"}


# ── Logging ───────────────────────────────────────────────────────────────────
def _configure_logging() -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_text_from_html(raw: bytes) -> str:
    """Strip HTML tags, collapse whitespace."""
    from html.parser import HTMLParser

    class _Extractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._buf: list[str] = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in {"script", "style", "nav", "footer", "header"}:
                self._skip = True

        def handle_endtag(self, tag):
            if tag in {"script", "style", "nav", "footer", "header"}:
                self._skip = False
                self._buf.append("\n")

        def handle_data(self, data):
            if not self._skip:
                self._buf.append(data)

    p = _Extractor()
    try:
        p.feed(raw.decode("utf-8", errors="replace"))
    except Exception:
        pass
    text = " ".join("".join(p._buf).split())
    return text


def _extract_text_from_pdf(raw: bytes) -> str:
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(raw))
        parts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(parts)
    except Exception as exc:
        log.warning("populate_kb.pdf_parse_failed", error=str(exc))
        return ""


def _extract_text_from_docx(raw: bytes) -> str:
    try:
        import docx

        doc = docx.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as exc:
        log.warning("populate_kb.docx_parse_failed", error=str(exc))
        return ""


def _extract_text(suffix: str, raw: bytes) -> str:
    if suffix in {".html", ".htm"}:
        return _extract_text_from_html(raw)
    if suffix == ".pdf":
        return _extract_text_from_pdf(raw)
    if suffix == ".docx":
        return _extract_text_from_docx(raw)
    # .txt / .md
    return raw.decode("utf-8", errors="replace")


# ── ZIP reading ───────────────────────────────────────────────────────────────

def _read_zip(zip_bytes: bytes) -> list[tuple[str, str]]:
    """Return [(filename, text), ...] for all parseable files in the ZIP."""
    results: list[tuple[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            p = Path(info.filename)
            if info.is_dir() or p.suffix.lower() not in _PARSEABLE:
                continue
            raw = zf.read(info.filename)
            text = _extract_text(p.suffix.lower(), raw)
            text = text.strip()
            if len(text) < MIN_CHUNK_CHARS:
                log.debug("populate_kb.skip_short_file", file=info.filename)
                continue
            results.append((info.filename, text))
    return results


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_documents(docs: list[tuple[str, str]]) -> list[dict]:
    """
    Recursively chunk each document.  Returns list of dicts with keys:
      chunk_id, source_file, text, char_count
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for filename, text in docs:
        for i, chunk_text in enumerate(splitter.split_text(text)):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue
            # Deterministic ID: stable across re-runs for the same source
            chunk_id = hashlib.sha256(f"{filename}:{i}:{chunk_text[:80]}".encode()).hexdigest()[:24]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": filename,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                }
            )
    return chunks


# ── Embedding + upsert ────────────────────────────────────────────────────────

async def _embed_and_upsert(
    chunks: list[dict],
    disease_id: str,
    disease_name: str,
    llm,
    vector_store,
    dry_run: bool,
) -> int:
    """Embed chunks in batches and upsert to Pinecone. Returns total upserted."""
    from healthbridgeai.core.models.retrieval import Chunk, Source

    namespace = disease_id
    total = 0

    for batch_start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + UPSERT_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        log.info(
            "populate_kb.embedding",
            disease=disease_id,
            batch=f"{batch_start // UPSERT_BATCH_SIZE + 1}/{-(-len(chunks) // UPSERT_BATCH_SIZE)}",
            chunk_count=len(batch),
        )

        dense_vecs = await llm.embed(texts)
        sparse_vecs = await llm.embed_sparse(texts)

        pinecone_chunks: list[Chunk] = []
        for c, dense, sparse in zip(batch, dense_vecs, sparse_vecs):
            source = Source(
                title=Path(c["source_file"]).stem,
                url=f"gcs://{disease_id}/{c['source_file']}",
                publication_date=None,
                publisher=disease_name,
                is_web_result=False,
            )
            pinecone_chunks.append(
                Chunk(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    source=source,
                    disease_id=disease_id,
                    chunk_type="knowledge",
                    embedding=dense,
                    sparse_embedding=sparse,
                    score=0.0,
                )
            )

        if dry_run:
            log.info("populate_kb.dry_run_skip_upsert", count=len(pinecone_chunks))
            total += len(pinecone_chunks)
        else:
            upserted = await vector_store.upsert_chunks(pinecone_chunks, namespace)
            total += upserted
            log.info("populate_kb.upserted", disease=disease_id, count=upserted, cumulative=total)

    return total


# ── GCS fetch ─────────────────────────────────────────────────────────────────

async def _fetch_zip_from_gcs(disease_id: str, gcs: object) -> Optional[bytes]:
    """Download <disease_id>.zip from GCS corpus folder. Returns None if not found."""
    gcs_path = f"corpora/{disease_id}.zip"
    try:
        data = await gcs.download_bytes(gcs_path)
        log.info("populate_kb.gcs_download_ok", path=gcs_path, size_mb=round(len(data) / 1e6, 1))
        return data
    except Exception as exc:
        log.warning("populate_kb.gcs_download_failed", path=gcs_path, error=str(exc))
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(disease_filter: Optional[str], local_zip: Optional[str], force: bool, dry_run: bool) -> None:
    from healthbridgeai.config import get_disease_registry
    from healthbridgeai.infrastructure.llm import OpenRouterClient
    from healthbridgeai.infrastructure.storage import GCSStorage
    from healthbridgeai.infrastructure.vector_store import PineconeAdapter

    registry = get_disease_registry()
    llm = OpenRouterClient()
    vector_store = PineconeAdapter()
    gcs = GCSStorage()

    diseases = [d for d in registry.all_diseases() if d.enabled]
    if disease_filter:
        diseases = [d for d in diseases if d.id == disease_filter]
        if not diseases:
            log.error("populate_kb.disease_not_found", filter=disease_filter)
            sys.exit(1)

    log.info("populate_kb.start", diseases=[d.id for d in diseases], dry_run=dry_run)

    grand_total = 0
    for disease in diseases:
        log.info("populate_kb.disease_start", disease=disease.id, name=disease.name)

        # ── Check existing index ───────────────────────────────────────────────
        if not force and not dry_run:
            try:
                stats = await vector_store.describe_index()
                ns_stats = stats.get("namespaces", {}).get(disease.id, {})
                existing = ns_stats.get("vector_count", 0)
                if existing > 0:
                    log.info(
                        "populate_kb.already_indexed",
                        disease=disease.id,
                        existing_vectors=existing,
                        hint="use --force to re-index",
                    )
                    continue
            except Exception as exc:
                log.warning("populate_kb.stats_check_failed", error=str(exc))

        # ── Fetch ZIP ──────────────────────────────────────────────────────────
        zip_bytes: Optional[bytes] = None

        if local_zip and disease_filter:
            # Local override only applies when a single disease is targeted
            p = Path(local_zip)
            if not p.exists():
                log.error("populate_kb.local_zip_not_found", path=local_zip)
                sys.exit(1)
            zip_bytes = p.read_bytes()
            log.info("populate_kb.using_local_zip", path=local_zip)
        else:
            zip_bytes = await _fetch_zip_from_gcs(disease.id, gcs)

        if not zip_bytes:
            log.error("populate_kb.no_source_zip", disease=disease.id)
            continue

        # ── Parse ──────────────────────────────────────────────────────────────
        log.info("populate_kb.parsing", disease=disease.id)
        docs = _read_zip(zip_bytes)
        log.info("populate_kb.parsed", disease=disease.id, document_count=len(docs))

        if not docs:
            log.warning("populate_kb.no_parseable_docs", disease=disease.id)
            continue

        # ── Chunk ──────────────────────────────────────────────────────────────
        chunks = _chunk_documents(docs)
        log.info("populate_kb.chunked", disease=disease.id, chunk_count=len(chunks))

        # ── Embed + upsert ─────────────────────────────────────────────────────
        n = await _embed_and_upsert(chunks, disease.id, disease.name, llm, vector_store, dry_run)
        grand_total += n
        log.info("populate_kb.disease_done", disease=disease.id, total_upserted=n)

    log.info("populate_kb.complete", grand_total=grand_total, dry_run=dry_run)


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(
        description="Populate HealthBridgeAI Pinecone knowledge base from GCS or local ZIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              # Index all enabled diseases from GCS
              python scripts/populate_kb.py

              # Re-index TB from a local ZIP
              python scripts/populate_kb.py --disease tb --zip /data/tb_corpus.zip --force

              # Dry-run (no writes)
              python scripts/populate_kb.py --dry-run
            """
        ),
    )
    parser.add_argument("--disease", help="Disease ID to index (default: all enabled)")
    parser.add_argument("--zip", dest="zip_path", help="Local ZIP file path (requires --disease)")
    parser.add_argument("--force", action="store_true", help="Re-index even if namespace already has vectors")
    parser.add_argument("--dry-run", action="store_true", help="Parse + chunk + embed but do not upsert")
    args = parser.parse_args()

    if args.zip_path and not args.disease:
        parser.error("--zip requires --disease to identify which namespace to populate")

    asyncio.run(run(args.disease, args.zip_path, args.force, args.dry_run))


if __name__ == "__main__":
    main()
