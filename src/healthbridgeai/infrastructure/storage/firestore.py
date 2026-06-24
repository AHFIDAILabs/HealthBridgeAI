# modules/user_preferences.py: User language preference management with Redis
"""
UserPreferenceManager: Manages user language preferences using Redis.
Stores only language preference - no statistics tracking.
"""

import redis
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class UserPreferenceManager:
    """
    Manages user language preferences using Redis.
    Simple storage: wa_id -TheDONOalaji> preferred_lang
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UserPreferenceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        try:
            # Connect to Redis
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True  # Auto-decode bytes to strings
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✓ Redis connected successfully")
            self._initialized = True
            
        except redis.ConnectionError as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.warning("⚠️  Falling back to in-memory storage (will not persist)")
            self.redis_client = None
            self._memory_store = {}  # Fallback to dict
            self._initialized = True
    
    def get_user_preference(self, wa_id: str) -> Optional[str]:
        """
        Get user's preferred language.
        
        Args:
            wa_id: WhatsApp ID (phone number)
            
        Returns:
            Language code ('ha', 'yo', 'ig', 'en') or None if not set
        """
        try:
            if self.redis_client:
                lang = self.redis_client.get(f"user:{wa_id}:lang")
                return lang
            else:
                # Fallback to memory
                return self._memory_store.get(wa_id)
        except Exception as e:
            logger.error(f"Error getting user preference: {e}")
            return None
    
    def set_user_preference(self, wa_id: str, lang: str) -> bool:
        """
        Set user's preferred language.
        
        Args:
            wa_id: WhatsApp ID (phone number)
            lang: Language code ('ha', 'yo', 'ig', 'en')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.redis_client:
                # Simple key-value storage
                self.redis_client.set(f"user:{wa_id}:lang", lang)
                logger.info(f"✓ Saved preference for {wa_id}: {lang}")
                return True
            else:
                # Fallback to memory
                self._memory_store[wa_id] = lang
                logger.info(f"✓ Saved preference (memory) for {wa_id}: {lang}")
                return True
                
        except Exception as e:
            logger.error(f"Error setting user preference: {e}")
            return False
    
    def is_first_time_user(self, wa_id: str) -> bool:
        """
        Check if this is user's first interaction.
        
        Args:
            wa_id: WhatsApp ID (phone number)
            
        Returns:
            True if first-time user (no language preference set), False otherwise
        """
        try:
            if self.redis_client:
                return not self.redis_client.exists(f"user:{wa_id}:lang")
            else:
                return wa_id not in self._memory_store
        except Exception as e:
            logger.error(f"Error checking first-time user: {e}")
            return False


# Singleton instance
_preference_manager = None


def get_preference_manager() -> UserPreferenceManager:
    """Get singleton instance of UserPreferenceManager."""
    global _preference_manager
    if _preference_manager is None:
        _preference_manager = UserPreferenceManager()
    return _preference_manager
