"""Domain exception hierarchy. Catch these at layer boundaries; never swallow silently."""
from __future__ import annotations


class HealthBridgeError(Exception):
    """Root for all domain errors."""


class ValidationError(HealthBridgeError):
    """Input violated a business rule (distinct from Pydantic schema errors)."""


class RateLimitError(HealthBridgeError):
    def __init__(self, phone_hash: str, count: int, limit: int) -> None:
        super().__init__(f"Rate limit: {count}/{limit} messages in window")
        self.phone_hash = phone_hash
        self.count = count
        self.limit = limit


class EmergencyDetectedError(HealthBridgeError):
    """Query matched emergency keywords — abort pipeline and send emergency response immediately."""

    def __init__(self, disease_ids: list[str], keywords: list[str]) -> None:
        super().__init__(f"Emergency in {disease_ids}: {keywords}")
        self.disease_ids = disease_ids
        self.keywords = keywords


class LanguageNotSupportedError(HealthBridgeError):
    def __init__(self, code: str, supported: list[str]) -> None:
        super().__init__(f"Language '{code}' unsupported. Supported: {supported}")
        self.code = code


class ExternalServiceError(HealthBridgeError):
    """External API failed after all retries."""

    def __init__(self, service: str, detail: str) -> None:
        super().__init__(f"{service}: {detail}")
        self.service = service


class RetrievalError(HealthBridgeError):
    """Vector store or re-ranking failure."""


class LLMError(HealthBridgeError):
    """LLM call failed or structured output validation failed after retries."""


class AudioError(HealthBridgeError):
    """Transcription or synthesis failure."""


class WebhookAuthError(HealthBridgeError):
    """Incoming webhook signature is invalid — reject with 401."""


class DiseaseNotEnabledError(HealthBridgeError):
    def __init__(self, disease_id: str) -> None:
        super().__init__(f"Disease '{disease_id}' is disabled in diseases.yaml")
        self.disease_id = disease_id


class CacheError(HealthBridgeError):
    """Semantic cache read/write failure (non-fatal — always fall through to pipeline)."""
