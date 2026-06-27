"""ILLMClient — contract for all LLM interactions (routing, translation, generation)."""
from __future__ import annotations

from typing import Any, Protocol, Type, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class ILLMClient(Protocol):
    async def structured(
        self,
        system: str,
        user: str,
        response_model: Type[T],
        model: str | None = None,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> T:
        """
        Call the LLM and parse the response into response_model via instructor.
        Retries up to max_retries on malformed JSON or validation failure.
        model=None → use configured LLM_PRIMARY_MODEL.
        """
        ...

    async def complete(
        self,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Plain text completion — used for translation and HyDE generation."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate BGE-M3 dense embeddings (1024-dim) for a batch of texts."""
        ...

    async def embed_sparse(self, texts: list[str]) -> list[dict[str, list]]:
        """Generate BM25 sparse vectors for hybrid search."""
        ...
