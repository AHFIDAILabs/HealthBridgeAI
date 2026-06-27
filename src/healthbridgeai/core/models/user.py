from __future__ import annotations

import hashlib
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class ConversationTurn(BaseModel):
    role: str                              # "user" | "assistant"
    content: str
    timestamp: int
    disease_ids: list[str] = Field(default_factory=list)
    language_code: str = "en"


class User(BaseModel):
    phone_number: str                      # E.164 format — never logged in plain text
    language_code: str = "en"
    audio_mode: bool = False               # User prefers voice responses
    created_at: int
    last_seen_at: int
    message_count: int = 0

    @computed_field                        # type: ignore[misc]
    @property
    def phone_hash(self) -> str:
        """Truncated SHA-256 — safe to include in logs and monitoring."""
        return hashlib.sha256(self.phone_number.encode()).hexdigest()[:12]


class RateLimit(BaseModel):
    phone_hash: str
    window_start: int                      # Unix epoch: start of current 60-second window
    message_count: int = 0
    is_blocked: bool = False

    def is_exceeded(self, limit: int) -> bool:
        return self.message_count >= limit
