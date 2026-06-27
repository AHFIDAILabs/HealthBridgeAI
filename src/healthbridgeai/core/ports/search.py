"""IWebSearch — contract for trusted-domain web search fallback (Tavily)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models.retrieval import WebResult


@runtime_checkable
class IWebSearch(Protocol):
    async def search(
        self,
        query: str,
        allowed_domains: list[str],
        max_results: int = 5,
    ) -> list[WebResult]:
        """
        Search the web restricted to allowed_domains only.
        Results from outside the domain list must be discarded.
        Used only when KB retrieval score falls below threshold after HyDE.
        """
        ...
