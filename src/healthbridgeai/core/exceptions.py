# modules/exceptions.py: Custom exceptions for better error handling
"""
Custom exception classes for Lighthouse HealthConnect.
Provides specific exception types for different error scenarios.
"""


class HealthConnectException(Exception):
    """Base exception for all Lighthouse HealthConnect errors"""
    pass


# Audio-related exceptions
class AudioProcessingError(HealthConnectException):
    """Base exception for audio-related errors"""
    pass


class TranscriptionError(AudioProcessingError):
    """Speech-to-text transcription failed"""
    pass


class TextToSpeechError(AudioProcessingError):
    """Text-to-speech conversion failed"""
    pass


class AudioFileError(AudioProcessingError):
    """Audio file not found or corrupted"""
    pass


class NATLASError(AudioProcessingError):
    """N-ATLAS model error"""
    pass


class WhisperError(AudioProcessingError):
    """Whisper model error"""
    pass


# Language-related exceptions
class LanguageError(HealthConnectException):
    """Base exception for language-related errors"""
    pass


class LanguageDetectionError(LanguageError):
    """Failed to detect language"""
    pass


class TranslationError(LanguageError):
    """Translation failed"""
    pass


class UnsupportedLanguageError(LanguageError):
    """Language is not supported"""
    pass


# Knowledge base exceptions
class KnowledgeBaseError(HealthConnectException):
    """Base exception for KB-related errors"""
    pass


class VectorStoreError(KnowledgeBaseError):
    """Vector store connection or operation failed"""
    pass


class DocumentLoadError(KnowledgeBaseError):
    """Failed to load documents"""
    pass


class EmbeddingError(KnowledgeBaseError):
    """Failed to generate embeddings"""
    pass


# LLM exceptions
class LLMError(HealthConnectException):
    """Base exception for LLM-related errors"""
    pass


class LLMResponseError(LLMError):
    """LLM failed to generate a response"""
    pass


class LLMTimeoutError(LLMError):
    """LLM request timed out"""
    pass


# Search exceptions
class SearchError(HealthConnectException):
    """Base exception for search-related errors"""
    pass


class WebSearchError(SearchError):
    """Web search failed"""
    pass


class KBSearchError(SearchError):
    """Knowledge base search failed"""
    pass


# Configuration exceptions
class ConfigurationError(HealthConnectException):
    """Configuration is invalid or missing"""
    pass


class APIKeyError(ConfigurationError):
    """API key is missing or invalid"""
    pass


# Session exceptions
class SessionError(HealthConnectException):
    """Session state error"""
    pass
