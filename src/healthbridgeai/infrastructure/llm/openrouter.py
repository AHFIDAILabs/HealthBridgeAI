"""OpenRouterClient — ILLMClient implementation using BGE-M3 locally + OpenRouter API."""
from __future__ import annotations

import asyncio
import threading
from typing import Type, TypeVar

import httpx
import instructor
import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from healthbridgeai.config.settings import settings
from healthbridgeai.core.exceptions import LLMError

log = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_SITE_URL = "https://healthbridgeai.ahfid.org"
_SITE_NAME = "HealthBridgeAI"

# BGE-M3 is shared across all instances — loaded once, 2 GB on disk
_model_lock = threading.Lock()
_bge_model = None


def _load_bge_model():
    global _bge_model
    if _bge_model is None:
        with _model_lock:
            if _bge_model is None:
                log.info("llm.bge_m3.loading")
                from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415
                _bge_model = BGEM3FlagModel(
                    settings.EMBEDDING_MODEL,
                    use_fp16=True,
                )
                log.info("llm.bge_m3.ready")
    return _bge_model


class OpenRouterClient:
    """
    Implements ILLMClient:
    - structured() / complete() → OpenRouter (OpenAI-compatible API)
    - embed() / embed_sparse()  → BGE-M3 (local, CPU/GPU)
    """

    def __init__(self) -> None:
        headers = {
            "HTTP-Referer": _SITE_URL,
            "X-Title": _SITE_NAME,
        }
        self._raw = AsyncOpenAI(
            base_url=_OPENROUTER_BASE,
            api_key=settings.OPENROUTER_API_KEY,
            default_headers=headers,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        self._instructor = instructor.from_openai(
            AsyncOpenAI(
                base_url=_OPENROUTER_BASE,
                api_key=settings.OPENROUTER_API_KEY,
                default_headers=headers,
                timeout=settings.LLM_TIMEOUT_SECONDS,
            )
        )

    # ── Structured output (instructor) ────────────────────────────────────────

    async def structured(
        self,
        system: str,
        user: str,
        response_model: Type[T],
        model: str | None = None,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> T:
        _model = model or settings.LLM_PRIMARY_MODEL
        try:
            return await self._instructor.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_model=response_model,
                max_retries=max_retries,
                temperature=temperature,
            )
        except Exception as exc:
            log.error("llm.structured.failed", model=_model, error=str(exc))
            raise LLMError(f"structured() failed: {exc}") from exc

    # ── Plain text completion ─────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        _model = model or settings.LLM_PRIMARY_MODEL
        try:
            resp = await self._raw.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            log.error("llm.complete.failed", model=_model, error=str(exc))
            raise LLMError(f"complete() failed: {exc}") from exc

    # ── Embeddings (local BGE-M3) ─────────────────────────────────────────────

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate 1024-dim dense embeddings via BGE-M3 (runs in thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = _load_bge_model()
        output = model.encode(
            texts,
            batch_size=12,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return [vec.tolist() for vec in output["dense_vecs"]]

    async def embed_sparse(self, texts: list[str]) -> list[dict[str, list]]:
        """Generate BM25-style sparse vectors via BGE-M3 lexical weights."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sparse_sync, texts)

    def _embed_sparse_sync(self, texts: list[str]) -> list[dict[str, list]]:
        model = _load_bge_model()
        output = model.encode(
            texts,
            batch_size=12,
            max_length=512,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        result = []
        for lw in output["lexical_weights"]:
            result.append({
                "indices": [int(k) for k in lw.keys()],
                "values": [float(v) for v in lw.values()],
            })
        return result
