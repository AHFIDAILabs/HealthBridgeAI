# modules/language_service.py: Centralized language detection and translation
"""
LanguageService: Handles all language-related operations including:
- Language detection
- Translation to/from English
- Caching of translations
"""

import logging
from typing import Optional, Dict, Tuple
from langchain_core.messages import HumanMessage

from modules.language_utils import detect_language, is_pidgin, get_language_name
from modules.cache_manager import get_cache
from modules.config_manager import get_config
from modules.exceptions import LanguageDetectionError, TranslationError, UnsupportedLanguageError

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Enhanced language detection with confidence"""
    
    def detect(self, text: str) -> Dict[str, any]:
        """
        Detect language with confidence score
        
        Returns:
            {
                'lang': str,
                'confidence': float,
                'is_pidgin': bool
            }
        """
        if not text or not text.strip():
            raise LanguageDetectionError("Empty text provided for detection")
        
        try:
            detected_lang = detect_language(text)
            is_pidgin_text = is_pidgin(text)
            
            # Adjust confidence based on text length and clarity
            confidence = self._calculate_confidence(text, detected_lang)
            
            result = {
                'lang': detected_lang,
                'confidence': confidence,
                'is_pidgin': is_pidgin_text,
                'language_name': get_language_name(detected_lang)
            }
            
            logger.info(f"Language detected: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            # Default to English
            return {
                'lang': 'en',
                'confidence': 0.5,
                'is_pidgin': False,
                'language_name': 'English'
            }
    
    def _calculate_confidence(self, text: str, lang: str) -> float:
        """Calculate detection confidence based on text characteristics"""
        confidence = 0.7  # Base confidence
        
        # Longer text = higher confidence
        if len(text) > 50:
            confidence += 0.1
        elif len(text) < 10:
            confidence -= 0.2
        
        # Check for language-specific patterns
        if lang in ["yo", "ig", "ha"]:
            # Check for diacritics or special characters
            if any(char in text for char in "àáèéìíòóùúẹọṣụ"):
                confidence += 0.15
        
        return min(1.0, max(0.0, confidence))


class Translator:
    """Handles translation operations with caching"""
    
    def __init__(self, llm):
        self.llm = llm
        self.cache = get_cache()
        self.prompts = self._create_translation_prompts()
        logger.info("Translator initialized")
    
    def _create_translation_prompts(self) -> Dict[str, Dict[str, str]]:
        """Create optimized translation prompts for each language"""
        return {
            "yo": {
                "to_en": """Translate this Yoruba health question to clear English. 
Preserve the exact meaning, especially medical terms.

Yoruba: {text}

English translation:""",
                "from_en": """Translate this English health information to STANDARD Yoruba language.
Use proper Yoruba diacritics (à, á, è, é, ì, í, ò, ó, ù, ú, ẹ, ọ, ṣ).
Make it grammatically correct and easily understandable.
Use common, everyday Yoruba words.

English: {text}

Standard Yoruba translation:"""
            },
            "ig": {
                "to_en": """Translate this Igbo health question to clear English.
Preserve the exact meaning, especially medical terms.

Igbo: {text}

English translation:""",
                "from_en": """Translate this English health information to STANDARD Igbo language.
Use proper Igbo orthography and diacritics where needed.
Make it grammatically correct and easily understandable.
Use common, everyday Igbo words.

English: {text}

Standard Igbo translation:"""
            },
            "ha": {
                "to_en": """Translate this Hausa health question to clear English.
Preserve the exact meaning, especially medical terms.

Hausa: {text}

English translation:""",
                "from_en": """Translate this English health information to STANDARD Hausa language.
Use proper Hausa spelling and orthography.
Make it grammatically correct and easily understandable.
Use common, everyday Hausa words.

English: {text}

Standard Hausa translation:"""
            },
            "pidgin": {
                "to_en": """Translate this Nigerian Pidgin health question to standard English.
Preserve the exact meaning, especially medical terms.

Nigerian Pidgin: {text}

English translation:""",
                "from_en": """Convert this English health information to STANDARD Nigerian Pidgin.
Make it conversational but clear and accurate.
Use proper Nigerian Pidgin widely understood across Nigeria.
Keep medical terms simple and explain them in Pidgin.

English: {text}

Nigerian Pidgin translation:"""
            }
        }
    
    def to_english(self, text: str, source_lang: str) -> str:
        """
        Translate from Nigerian language to English
        
        Args:
            text: Text to translate
            source_lang: Source language code
        
        Returns:
            Translated English text
        """
        if source_lang == "en":
            return text
        
        # Check cache first
        cached = self.cache.get_translation(text, source_lang, "en")
        if cached:
            logger.info(f"Translation cache hit: {source_lang} -> en")
            return cached
        
        try:
            logger.info(f"Translating {source_lang} -> English: '{text[:50]}...'")
            
            prompt_template = self.prompts.get(source_lang, {}).get("to_en")
            if not prompt_template:
                raise TranslationError(f"No translation prompt for {source_lang}")
            
            prompt = prompt_template.format(text=text)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            translation = response.content.strip()
            
            # Cache the translation
            self.cache.set_translation(text, source_lang, "en", translation)
            
            logger.info(f"Translation complete: '{translation[:50]}...'")
            return translation
            
        except Exception as e:
            logger.error(f"Translation {source_lang} -> en failed: {e}")
            raise TranslationError(f"Translation failed: {e}")
    
    def from_english(self, text: str, target_lang: str) -> str:
        """
        Translate from English to Nigerian language
        
        Args:
            text: English text to translate
            target_lang: Target language code
        
        Returns:
            Translated text
        """
        if target_lang == "en":
            return text
        
        # Check cache first
        cached = self.cache.get_translation(text, "en", target_lang)
        if cached:
            logger.info(f"Translation cache hit: en -> {target_lang}")
            return cached
        
        try:
            logger.info(f"Translating English -> {target_lang}: '{text[:50]}...'")
            
            prompt_template = self.prompts.get(target_lang, {}).get("from_en")
            if not prompt_template:
                raise TranslationError(f"No translation prompt for {target_lang}")
            
            prompt = prompt_template.format(text=text)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            translation = response.content.strip()
            
            # Cache the translation
            self.cache.set_translation(text, "en", target_lang, translation)
            
            logger.info(f"Translation complete: '{translation[:50]}...'")
            return translation
            
        except Exception as e:
            logger.error(f"Translation en -> {target_lang} failed: {e}")
            raise TranslationError(f"Translation failed: {e}")


class LanguageService:
    """
    Centralized language service for all language operations
    """
    
    def __init__(self, llm):
        self.detector = LanguageDetector()
        self.translator = Translator(llm)
        self.config = get_config()
        logger.info("LanguageService initialized")
    
    def detect_language(self, text: str) -> Dict[str, any]:
        """Detect language with confidence"""
        return self.detector.detect(text)
    
    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate to English for KB search"""
        return self.translator.to_english(text, source_lang)
    
    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate response back to user's language"""
        return self.translator.from_english(text, target_lang)
    
    def is_language_supported(self, lang_code: str) -> bool:
        """Check if language is supported"""
        return lang_code in self.config.get_supported_languages()
    
    def get_language_name(self, lang_code: str) -> str:
        """Get human-readable language name"""
        lang_config = self.config.get_language_config(lang_code)
        if lang_config:
            return lang_config.name
        return get_language_name(lang_code)
    
    def validate_language(self, lang_code: str) -> str:
        """Validate and return language code, defaulting to English if invalid"""
        if self.is_language_supported(lang_code):
            return lang_code
        
        logger.warning(f"Unsupported language: {lang_code}, defaulting to English")
        return "en"


# Factory function
def create_language_service(llm) -> LanguageService:
    """Create a LanguageService instance"""
    return LanguageService(llm)
