"""IMessagingProvider — contract for any WhatsApp delivery backend (WhatChimp, Twilio, etc.)."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from ..models.message import InboundMessage


@runtime_checkable
class IMessagingProvider(Protocol):
    async def parse_webhook(
        self, payload: dict, headers: dict[str, str]
    ) -> Optional[InboundMessage]:
        """
        Parse a raw webhook POST into an InboundMessage.
        Return None for status/delivery-receipt events that need no response.
        """
        ...

    async def validate_webhook(self, raw_body: bytes, headers: dict[str, str]) -> bool:
        """Verify webhook signature. Raise WebhookAuthError on failure."""
        ...

    async def send_text(self, to: str, text: str) -> str:
        """Send a text message. Return provider message ID."""
        ...

    async def send_audio(self, to: str, audio_url: str, caption: str = "") -> str:
        """Send an audio message via public GCS URL. Return provider message ID."""
        ...

    async def download_media(self, media_id: str) -> bytes:
        """Download audio/image bytes using the provider's media endpoint."""
        ...
