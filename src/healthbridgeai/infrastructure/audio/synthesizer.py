# modules/audio_synthesizer.py: Text-to-speech synthesis for multilingual output
"""
AudioSynthesizer: Handles text-to-speech generation with Nigerian language support.
Uses Yarn GPT TTS (priority), MMS-TTS for native support, and gTTS for fallback.
"""

import os
import uuid
import tempfile
import logging
import requests
from typing import Optional
from dotenv import load_dotenv
import torch
import torchaudio
from gtts import gTTS

from modules.exceptions import TextToSpeechError
from modules.config_manager import get_config

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class YarnGPTSynthesizer:
    """Yarn GPT TTS API for Nigerian languages (Hausa, Yoruba, Igbo, English)"""
    
    def __init__(self):
        self.api_url = "https://yarngpt.ai/api/v1/tts"
        self.api_key = os.getenv("YARNGPT_API_KEY")  # Matches the .env file variable name
        self.timeout = 120  # 2 minutes
        
        # All voices support all languages according to user
        self.default_voice = "Idera"  # Default voice
        
        if self.api_key:
            logger.info("✓ Yarn GPT TTS initialized with API key")
        else:
            logger.warning("⚠️ YARNGPT_API_KEY not found in environment - Yarn GPT TTS disabled")
    
    def can_synthesize(self, lang: str) -> bool:
        """Check if Yarn GPT can synthesize this language"""
        # Supports ha, yo, ig, en
        return self.api_key is not None and lang in ["ha", "yo", "ig", "en"]
    
    def synthesize(self, text: str, lang: str, voice: Optional[str] = None) -> str:
        ogg_path = "test.ogg"
        mp3_path = "test.mp3"
        """
        Synthesize speech using Yarn GPT TTS API
        
        Args:
            text: Text to synthesize
            lang: Language code (ha, yo, ig, en)
            voice: Optional voice name (defaults to "Idera")
        
        Returns:
            Path to generated OGG audio file
        
        Raises:
            TextToSpeechError: If synthesis fails
        """
        if not self.api_key:
            raise TextToSpeechError("Yarn GPT API key not configured")
        
        if not self.can_synthesize(lang):
            raise TextToSpeechError(f"Yarn GPT does not support language: {lang}")
        
        try:
            logger.info(f"Yarn GPT synthesizing {lang} with voice {voice or self.default_voice}...")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "text": text,
                "voice": voice or self.default_voice
            }
            
            # Make API request with streaming (no timeout - following official API example)
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=payload, 
                stream=True
            )
            # Check for errors
            if response.status_code != 200:
                error_msg = f"Yarn GPT API error ({response.status_code})"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data}"
                    logger.error(f"❌ {error_msg}")
                except:
                    logger.error(f"❌ {error_msg}: {response.text[:200]}")
                raise TextToSpeechError(error_msg)
            
            # Save streamed audio to temporary WAV file (Yarn GPT returns WAV, not MP3!)
            temp_dir = tempfile.gettempdir()
            wav_path = os.path.join(temp_dir, f"tts_yarngpt_{uuid.uuid4().hex}.wav")
            ogg_path = os.path.join(temp_dir, f"tts_yarngpt_{uuid.uuid4().hex}.ogg")
            
            # Stream audio to file
            with open(wav_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"✓ Yarn GPT audio downloaded (WAV): {wav_path}")
            
            # Convert WAV to OGG Opus for WhatsApp compatibility
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(wav_path)
                audio.export(
                    ogg_path,
                    format='ogg',
                    codec='libopus',
                    parameters=["-strict", "-2"]
                )
                os.remove(wav_path)  # Remove temporary WAV file
                logger.info(f"✓ Yarn GPT synthesis complete (OGG): {ogg_path}")
                return ogg_path
                
            except Exception as conv_error:
                # If conversion fails, use WAV
                logger.warning(f"⚠️ OGG conversion failed, using WAV: {conv_error}")
                logger.info(f"✓ Yarn GPT synthesis complete (WAV): {wav_path}")
                return wav_path
                logger.info(f"✓ Yarn GPT synthesis complete (MP3): {mp3_path}")
                return mp3_path
            
        except requests.exceptions.Timeout:
            error_msg = f"Yarn GPT API timeout after {self.timeout}s"
            logger.error(f"❌ {error_msg}")
            raise TextToSpeechError(error_msg)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Yarn GPT API network error: {e}"
            logger.error(f"❌ {error_msg}")
            raise TextToSpeechError(error_msg)
            
        except Exception as e:
            error_msg = f"Yarn GPT synthesis failed: {e}"
            logger.error(f"❌ {error_msg}")
            raise TextToSpeechError(error_msg)


class MMSynthesizer:
    """Meta Multilingual Speech (MMS) TTS for Yoruba and Hausa"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.enabled_langs = set()
        logger.info("MMS Synthesizer initialized")
    
    def can_synthesize(self, lang: str) -> bool:
        """Check if MMS supports this language"""
        return lang in ["yo", "ha"]
    
    def _load_model(self, lang: str):
        """Lazy load MMS model for a specific language"""
        if lang in self.models:
            return
        
        try:
            from transformers import VitsModel, AutoTokenizer
            
            mms_lang_map = {
                "yo": "yor",  # Yoruba
                "ha": "hau"   # Hausa
            }
            
            if lang not in mms_lang_map:
                raise TextToSpeechError(f"MMS not supported for {lang}")
            
            mms_code = mms_lang_map[lang]
            model_name = f"facebook/mms-tts-{mms_code}"
            
            logger.info(f"Loading MMS model for {lang}...")
            self.models[lang] = VitsModel.from_pretrained(model_name)
            self.tokenizers[lang] = AutoTokenizer.from_pretrained(model_name)
            self.enabled_langs.add(lang)
            logger.info(f"✓ MMS {lang.upper()} model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load MMS model for {lang}: {e}")
            raise TextToSpeechError(f"MMS model load failed: {e}")
    
    def synthesize(self, text: str, lang: str) -> str:
        """
        Synthesize speech using MMS
        Returns: Path to generated audio file
        """
        if not self.can_synthesize(lang):
            raise TextToSpeechError(f"MMS cannot synthesize {lang}")
        
        try:
            self._load_model(lang)
            
            logger.info(f"MMS synthesizing {lang}...")
            
            # Tokenize input
            inputs = self.tokenizers[lang](text, return_tensors="pt")
            
            # Generate speech
            with torch.no_grad():
                output = self.models[lang](**inputs).waveform
            
            # Save to OGG instead of WAV for WhatsApp compatibility
            temp_dir = tempfile.gettempdir()
            wav_path = os.path.join(temp_dir, f"tts_mms_{uuid.uuid4().hex}.wav")
            ogg_path = os.path.join(temp_dir, f"tts_mms_{uuid.uuid4().hex}.ogg")
            
            # Save as WAV first
            torchaudio.save(wav_path, output, sample_rate=16000)
            
            # Convert to OGG Opus for WhatsApp
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(wav_path)
                audio.export(
                    ogg_path, 
                    format='ogg', 
                    codec='libopus',
                    parameters=["-strict", "-2"]
                )
                os.remove(wav_path)  # Remove temporary WAV
                file_path = ogg_path
                logger.info(f"MMS synthesis complete (OGG): {file_path}")
            except ImportError:
                # If pydub not available, use WAV
                file_path = wav_path
                logger.warning("pydub not available, using WAV format")
                logger.info(f"MMS synthesis complete (WAV): {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"MMS synthesis failed: {e}")
            raise TextToSpeechError(f"MMS error: {e}")


class GTTSynthesizer:
    """Google TTS synthesizer (fallback)"""
    
    def synthesize(self, text: str, lang: str = "en", tld: str = "com.ng") -> str:
        """
        Synthesize speech using gTTS
        Returns: Path to generated audio file (OGG format)
        """
        try:
            logger.info(f"gTTS synthesizing {lang} (tld={tld})...")
            
            tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
            
            temp_dir = tempfile.gettempdir()
            mp3_path = os.path.join(temp_dir, f"tts_gtts_{uuid.uuid4().hex}.mp3")
            ogg_path = os.path.join(temp_dir, f"tts_gtts_{uuid.uuid4().hex}.ogg")
            
            # Save as MP3 first
            tts.save(mp3_path)
            
            # Convert to OGG Opus for WhatsApp
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(
                    ogg_path, 
                    format='ogg', 
                    codec='libopus',
                    parameters=["-strict", "-2"]
                )
                os.remove(mp3_path)  # Remove temporary MP3
                file_path = ogg_path
                logger.info(f"gTTS synthesis complete (OGG): {file_path}")
            except Exception as conv_error:
                # If conversion fails, use MP3
                file_path = mp3_path
                logger.warning(f"OGG conversion failed, using MP3: {conv_error}")
                logger.info(f"gTTS synthesis complete (MP3): {file_path}")
            
            logger.info(f"gTTS synthesis complete: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"gTTS synthesis failed: {e}")
            raise TextToSpeechError(f"gTTS error: {e}")


class AudioSynthesizer:
    """
    Main TTS class with intelligent provider selection.
    Priority: Yarn GPT → MMS (ha/yo only) → gTTS (all languages)
    """

    def __init__(self):
        self.yarngpt = YarnGPTSynthesizer()
        self.mms = MMSynthesizer()
        self.gtts = GTTSynthesizer()
        logger.info("AudioSynthesizer initialized with Yarn GPT → MMS → gTTS cascade")
    
    def synthesize(self, text: str, lang: str = "en") -> str:
        """
        Convert text to speech with cascading fallback
        
        Priority order:
        1. Yarn GPT TTS (all languages: ha, yo, ig, en)
        2. MMS TTS (Hausa, Yoruba only)
        3. gTTS (final fallback for all languages)
        
        Args: 
            text: Text to synthesize
            lang: Language code (ha, yo, ig, en)
        
        Returns:
            Path to generated audio file (OGG format preferred)
        
        Raises:
            TextToSpeechError: If all synthesis methods fail
        """
        text = text.strip()
        if not text:
            raise TextToSpeechError("Empty text provided for TTS")
        
        logger.info(f"🔊 Synthesizing speech for {lang}: '{text[:50]}...'")
        
        config = get_config()
        lang_config = config.get_language_config(lang)
        
        if not lang_config:
            logger.warning(f"Unknown language {lang}, using English")
            lang = "en"
            lang_config = config.get_language_config("en")
        
        # === PRIORITY 1: Try Yarn GPT first (supports ha, yo, ig, en) ===
        if self.yarngpt.can_synthesize(lang):
            try:
                logger.info(f"🎯 Trying Yarn GPT TTS for {lang}...")
                result = self.yarngpt.synthesize(text, lang)
                logger.info(f"✅ Yarn GPT TTS succeeded for {lang}")
                return result
            except TextToSpeechError as e:
                logger.warning(f"⚠️ Yarn GPT TTS failed for {lang}: {e}")
                logger.info(f"↩️ Falling back from Yarn GPT...")
        else:
            logger.info(f"⊘ Yarn GPT not available for {lang} (API key missing or unsupported language)")
        
        # === PRIORITY 2: Try MMS for Hausa/Yoruba only ===
        if lang in ["yo", "ha"]:
            try:
                logger.info(f"🎯 Trying MMS TTS for {lang}...")
                result = self.mms.synthesize(text, lang)
                logger.info(f"✅ MMS TTS succeeded for {lang}")
                return result
            except TextToSpeechError as e:
                logger.warning(f"⚠️ MMS TTS failed for {lang}: {e}")
                logger.info(f"↩️ Falling back from MMS to gTTS...")
        
        # === PRIORITY 3: Final fallback to gTTS (works for all languages) ===
        try:
            logger.info(f"🎯 Using gTTS for {lang}...")
            tts_lang = lang_config.tts_lang
            tts_tld = lang_config.tts_tld
            result = self.gtts.synthesize(text, tts_lang, tts_tld)
            logger.info(f"✅ gTTS succeeded for {lang}")
            return result
        except TextToSpeechError as e:
            # Final fallback to English
            logger.error(f"❌ gTTS failed for {lang}, trying English fallback")
            try:
                result = self.gtts.synthesize(text, "en", "com.ng")
                logger.info(f"✅ gTTS English fallback succeeded")
                return result
            except Exception as e2:
                error_msg = f"All TTS methods failed (Yarn GPT → MMS → gTTS → English): {e2}"
                logger.error(f"❌ {error_msg}")
                raise TextToSpeechError(error_msg)
    
    def cleanup_audio_file(self, file_path: str):
        """Clean up temporary audio file"""
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up audio file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup audio file {file_path}: {e}")


# Singleton instance
_synthesizer_instance: Optional[AudioSynthesizer] = None


def get_synthesizer() -> AudioSynthesizer:
    """Get the global AudioSynthesizer instance"""
    global _synthesizer_instance
    if _synthesizer_instance is None:
        _synthesizer_instance = AudioSynthesizer()
    return _synthesizer_instance
