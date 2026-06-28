"""TavilyAdapter — IWebSearch implementation using Tavily trusted-domain search."""
from __future__ import annotations

import structlog
from tavily import AsyncTavilyClient

from healthbridgeai.config.settings import settings
from healthbridgeai.core.models.retrieval import WebResult

log = structlog.get_logger(__name__)


class TavilyAdapter:
    """
    Wraps the Tavily API to enforce trusted-domain restrictions.
    Results outside allowed_domains are discarded before returning.
    """

    def __init__(self) -> None:
        self._client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)

    async def search(
        self,
        query: str,
        allowed_domains: list[str],
        max_results: int = 5,
    ) -> list[WebResult]:
        if not allowed_domains:
            return []

        try:
            # Tavily's include_domains restricts results server-side
            response = await self._client.search(
                query=query,
                search_depth="basic",
                max_results=max_results * 2,  # over-fetch in case some domains are filtered
                include_domains=allowed_domains,
                include_answer=False,
                include_raw_content=False,
            )
        except Exception as exc:
            log.warning("tavily.search.failed", error=str(exc))
            return []

        results: list[WebResult] = []
        for r in response.get("results", []):
            url: str = r.get("url", "")
            domain = _extract_domain(url)
            if not _is_allowed(domain, allowed_domains):
                log.debug("tavily.domain_filtered", url=url)
                continue
            results.append(
                WebResult(
                    title=r.get("title", ""),
                    url=url,
                    content=r.get("content", ""),
                    score=float(r.get("score", 0.0)),
                    domain=domain,
                )
            )
            if len(results) >= max_results:
                break

        log.debug("tavily.search.done", returned=len(results), query=query[:60])
        return results


def _extract_domain(url: str) -> str:
    """Extract bare domain (without www.) from a URL string."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
        return host.removeprefix("www.")
    except Exception:
        return ""


def _is_allowed(domain: str, allowed: list[str]) -> bool:
    """Allow if domain matches or is a subdomain of any allowed entry."""
    for allowed_domain in allowed:
        d = allowed_domain.lower().removeprefix("www.")
        if domain == d or domain.endswith("." + d):
            return True
    return False
