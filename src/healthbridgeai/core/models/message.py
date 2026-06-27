from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    DOCUMENT = "document"
    INTERACTIVE = "interactive"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class MediaInfo(BaseModel):
    media_id: str
    mime_type: str
    url: Optional[str] = None
    filename: Optional[str] = None


class InboundMessage(BaseModel):
    """Raw message as parsed from the WhatChimp webhook payload."""

    message_id: str
    from_number: str          # E.164 format, e.g. "+2348012345678"
    to_number: str            # Bot's WhatsApp number
    type: MessageType
    timestamp: int            # Unix epoch seconds
    text: Optional[str] = None
    media: Optional[MediaInfo] = None
    raw_payload: dict = Field(default_factory=dict, exclude=True)


class ParsedMessage(BaseModel):
    """Message after language detection, translation, and command extraction."""

    original: InboundMessage
    language_code: str = "en"         # BCP-47, e.g. "yo", "ha", "ig", "pcm"
    english_text: str = ""            # Translated to English for processing
    is_command: bool = False
    command: Optional[str] = None     # "help" | "language" | "audio" | "about" | "feedback"
    command_args: list[str] = Field(default_factory=list)


class UserCommand(str, Enum):
    HELP = "help"
    LANGUAGE = "language"
    AUDIO = "audio"
    ABOUT = "about"
    FEEDBACK = "feedback"
