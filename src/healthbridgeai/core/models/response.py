from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .retrieval import Source


class LLMResponse(BaseModel):
    """Structured output from the LLM — validated by instructor before reaching the pipeline."""

    answer: str
    sources_used: list[int]                      # 1-based indices into the context document list
    confidence: Literal["high", "medium", "low"]
    needs_professional: bool                      # Explicitly advise consulting a doctor
    caveat: Optional[str] = None                 # Critical qualification (e.g. drug interaction warning)


class BotResponse(BaseModel):
    """Final response ready to be sent to WhatsApp, in the user's language."""

    text: str                                    # Fully formatted WhatsApp message (citations included)
    language_code: str
    sources: list[Source] = Field(default_factory=list)
    confidence: str
    needs_professional: bool = False
    is_emergency: bool = False
    audio_gcs_uri: Optional[str] = None          # Set if user prefers audio mode
    cache_hit: bool = False


class CachedResponse(BaseModel):
    """Cache entry stored in Pinecone response-cache namespace."""

    english_query: str
    disease_ids: str                             # Comma-joined sorted disease IDs
    query_intent: str
    english_response: str
    sources_json: str                            # JSON-serialised list[Source]
    confidence: str

    # Unix epoch timestamps
    created_at: int
    expires_at: int
    hit_count: int = 0

    def sources(self) -> list[Source]:
        raw = json.loads(self.sources_json)
        return [Source(**s) for s in raw]

    @staticmethod
    def sources_to_json(sources: list[Source]) -> str:
        return json.dumps([s.model_dump() for s in sources], ensure_ascii=False)


MEDICAL_DISCLAIMER = (
    "⚠️ _This information is for educational purposes only. "
    "Always consult a qualified healthcare professional for medical decisions._"
)

CITATION_DIVIDER = "─────────────────────"
