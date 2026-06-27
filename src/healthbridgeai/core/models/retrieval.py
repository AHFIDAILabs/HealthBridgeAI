from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Citation metadata for one source document shown to the user."""

    index: int            # 1-based — matches [Doc N] in LLM prompt
    title: str
    url: str
    domain: str
    source_type: str      # guideline | fact_sheet | patient_education | surveillance_report
    section: Optional[str] = None
    page_number: Optional[int] = None

    @property
    def short_citation(self) -> str:
        """'WHO Consolidated TB Guidelines (2022) — who.int'"""
        return f"{self.title} — {self.domain}"


class Chunk(BaseModel):
    """A single text chunk retrieved from Pinecone with its provenance."""

    text: str
    score: float
    disease: str
    doc_id: str
    source: Source
    chunk_type: str       # symptoms | treatment | prevention | complications | ...
    chunk_index: int = 0
    language: str = "en"


class RetrievalResult(BaseModel):
    """Aggregated output of the RAG retrieval step."""

    chunks: list[Chunk] = Field(default_factory=list)
    best_score: float = 0.0
    used_hyde: bool = False
    used_web_fallback: bool = False
    web_results: list["WebResult"] = Field(default_factory=list)

    @property
    def all_sources(self) -> list[Source]:
        kb_sources = [c.source for c in self.chunks]
        web_sources = [w.as_source(i + len(kb_sources) + 1) for i, w in enumerate(self.web_results)]
        return kb_sources + web_sources


class WebResult(BaseModel):
    """A single result from Tavily web search."""

    title: str
    url: str
    content: str
    score: float
    domain: str

    def as_source(self, index: int) -> Source:
        return Source(
            index=index,
            title=f"Web: {self.title}",
            url=self.url,
            domain=self.domain,
            source_type="web",
        )
