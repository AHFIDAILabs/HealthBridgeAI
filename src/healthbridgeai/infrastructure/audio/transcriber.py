# modules/audio_transcriber.py: Refactored class-based audio transcription
"""
AudioTranscriber: Handles multilingual speech-to-text with fallback chain.
Priority: N-ATLAS (Nigerian languages) -> Whisper -> Google Speech Recognition
"""

import os
import logging
from typing import Optional, Tuple
import speech_recognition as sr
from dotenv import load_dotenv

from modules.exceptions import (
    TranscriptionError, AudioFileError, NATLASError, WhisperError
)
from modules.config_manager import get_config

logger = logging.getLogger(__name__)


class NATLASTranscriber:
    """Handles N-ATLAS model transcription for Nigerian languages"""
    
    def __init__(self):
        self.pipelines = {}
        self.enabled = False
        self._load_models()
    
    def _load_models(self):
        """Lazy load N-ATLAS models"""
        try:
            from transformers import pipeline
            import librosa
            
            load_dotenv()
            hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
            
            if not hf_token:
                logger.warning("HuggingFace token not found, N-ATLAS unavailable")
                return
            
            os.environ["HF_TOKEN"] = hf_token
            
            config = get_config()
            natlas_langs = config.get_natlas_languages()
            
            model_map = {
                "ha": "Hausa-ASR",
                "ig": "Igbo-ASR",
                "yo": "Yoruba-ASR"
            }
            
            for lang_code in natlas_langs:
                if lang_code not in model_map:
                    continue
                
                model_name = model_map[lang_code]
                try:
                    logger.info(f"Loading N-ATLAS {lang_code.upper()} model...")
                    self.pipelines[lang_code] = pipeline(
                        "automatic-speech-recognition",
                        model=f'NCAIR1/{model_name}',
                        token=hf_token
                    )
                    logger.info(f"✓ N-ATLAS {lang_code.upper()} loaded")
                except Exception as e:
                    logger.warning(f"✗ N-ATLAS {lang_code.upper()} failed: {e}")
            
            self.enabled = len(self.pipelines) > 0
            if self.enabled:
                logger.info(f"N-ATLAS enabled for: {list(self.pipelines.keys())}")
            
        except ImportError as e:
            logger.warning(f"N-ATLAS dependencies not available: {e}")
            self.enabled = False
    
    def can_transcribe(self, lang: str) -> bool:
        """Check if N-ATLAS can transcribe this language"""
        return self.enabled and lang in self.pipelines
    
    def transcribe(self, audio_path: str, lang: str) -> str:
        """Transcribe using N-ATLAS"""
        if not self.can_transcribe(lang):
            raise NATLASError(f"N-ATLAS not available for language: {lang}")
        
        try:
            import librosa
            
            logger.info(f"N-ATLAS transcribing {lang}...")
            
            # Load audio at 16kHz
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Transcribe
            result = self.pipelines[lang](audio_data)
            text = result.get("text", "").strip()
            
            if text and len(text) > 2:
                logger.info(f"N-ATLAS success: '{text[:50]}...'")
                return text
            else:
                raise NATLASError("N-ATLAS returned empty transcription")
                
        except Exception as e:
            logger.error(f"N-ATLAS transcription failed: {e}")
            raise NATLASError(f"N-ATLAS error: {e}")


class WhisperTranscriber:
    """Handles Whisper model transcription"""
    
    def __init__(self):
        self.model = None
        self.enabled = False
    
    def _load_model(self):
        """Lazy load Whisper model"""
        if self.model is not None:
            return
        
        try:
            import whisper
            config = get_config()
            model_size = config.audio.whisper_model_size
            
            logger.info(f"Loading Whisper {model_size} model...")
            self.model = whisper.load_model(model_size)
            self.enabled = True
            logger.info("✓ Whisper model loaded")
            
        except Exception as e:
            logger.warning(f"✗ Whisper not available: {e}")
            self.enabled = False
    
    def transcribe(self, audio_path: str, lang: Optional[str] = None) -> Tuple[str, str]:
        """
        Transcribe using Whisper
        Returns: (text, detected_language)
        """
        self._load_model()
        
        if not self.enabled:
            raise WhisperError("Whisper model not available")
        
        try:
            logger.info("Whisper transcribing...")
            
            result = self.model.transcribe(
                audio_path,
                fp16=False,
                verbose=False,
                language=lang if lang != "auto" else None
            )
            
            text = result.get("text", "").strip()
            detected_lang = result.get("language", "en")
            
            logger.info(f"Whisper success: lang={detected_lang}, text='{text[:50]}...'")
            return text, detected_lang
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise WhisperError(f"Whisper error: {e}")


class GoogleTranscriber:
    """Handles Google Speech Recognition"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
    
    def transcribe(self, audio_path: str, lang: str = "en") -> str:
        """Transcribe using Google Speech Recognition"""
        try:
            logger.info(f"Google SR transcribing ({lang})...")
            
            with sr.AudioFile(audio_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.record(source)
            
            text = self.recognizer.recognize_google(audio, language=lang)
            logger.info(f"Google SR success: '{text[:50]}...'")
            return text
            
        except sr.UnknownValueError:
            raise TranscriptionError("Google could not understand the audio")
        except sr.RequestError as e:
            raise TranscriptionError(f"Google SR request error: {e}")
        except Exception as e:
            raise TranscriptionError(f"Google SR error: {e}")


class AudioTranscriber:
    """
    Main audio transcription class with intelligent fallback chain.
    Priority: N-ATLAS -> Whisper -> Google Speech Recognition
    """
    
    def __init__(self):
        self.natlas = NATLASTranscriber()
        self.whisper = WhisperTranscriber()
        self.google = GoogleTranscriber()
        logger.info("AudioTranscriber initialized")
    
    def transcribe(self, audio_path: str, lang: str = "auto") -> Tuple[str, str]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            lang: Language code or "auto" for auto-detection
        
        Returns:
            Tuple of (transcribed_text, detected_language)
        
        Raises:
            AudioFileError: If audio file doesn't exist
            TranscriptionError: If all transcription methods fail
        """
        if not os.path.isfile(audio_path):
            raise AudioFileError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing audio: {audio_path}, lang={lang}")
        
        # Auto-detect language using Whisper
        if lang == "auto":
            return self._transcribe_with_detection(audio_path)
        
        # Try specific language transcription
        return self._transcribe_with_lang(audio_path, lang)
    
    def _transcribe_with_detection(self, audio_path: str) -> Tuple[str, str]:
        """Transcribe with automatic language detection"""
        try:
            # Use Whisper for detection and transcription
            text, detected_lang = self.whisper.transcribe(audio_path)
            
            # Validate detected language
            config = get_config()
            supported_langs = config.get_supported_languages()
            
            if detected_lang not in supported_langs:
                logger.warning(f"Detected unsupported language: {detected_lang}, using English")
                detected_lang = "en"
            
            return text, detected_lang
            
        except WhisperError as e:
            logger.warning(f"Whisper detection failed: {e}, trying Google SR")
            # Fallback to Google with English
            try:
                text = self.google.transcribe(audio_path, "en")
                return text, "en"
            except Exception as e:
                raise TranscriptionError(f"All transcription methods failed: {e}")
    
    def _transcribe_with_lang(self, audio_path: str, lang: str) -> Tuple[str, str]:
        """Transcribe with specified language"""
        errors = []
        
        # Try N-ATLAS for Nigerian languages
        if self.natlas.can_transcribe(lang):
            try:
                text = self.natlas.transcribe(audio_path, lang)
                return text, lang
            except NATLASError as e:
                errors.append(f"N-ATLAS: {e}")
                logger.warning(f"N-ATLAS failed, trying Whisper: {e}")
        
        # Try Whisper
        try:
            text, detected_lang = self.whisper.transcribe(audio_path, lang)
            return text, lang  # Use requested lang, not detected
        except WhisperError as e:
            errors.append(f"Whisper: {e}")
            logger.warning(f"Whisper failed, trying Google SR: {e}")
        
        # Try Google Speech Recognition
        try:
            text = self.google.transcribe(audio_path, lang)
            return text, lang
        except TranscriptionError as e:
            errors.append(f"Google: {e}")
        
        # All methods failed
        raise TranscriptionError(
            f"All transcription methods failed for {lang}: {'; '.join(errors)}"
        )


# Singleton instance
_transcriber_instance: Optional[AudioTranscriber] = None


def get_transcriber() -> AudioTranscriber:
    """Get the global AudioTranscriber instance"""
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = AudioTranscriber()
    return _transcriber_instance
