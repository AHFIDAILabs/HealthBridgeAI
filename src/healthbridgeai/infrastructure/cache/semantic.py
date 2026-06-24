# modules/cache_manager.py: Persistent cache for expensive operations
import hashlib
import pickle
import os
import json
import time
from typing import Optional, Any, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages persistent cache for translations and KB results"""
    
    def __init__(self, cache_dir: str = ".cache", ttl: int = 86400):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time-to-live in seconds (default: 24 hours)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = ttl
        
        # Create subdirectories for different cache types
        self.translation_dir = self.cache_dir / "translations"
        self.kb_dir = self.cache_dir / "kb_results"
        self.embeddings_dir = self.cache_dir / "embeddings"
        
        for directory in [self.translation_dir, self.kb_dir, self.embeddings_dir]:
            directory.mkdir(exist_ok=True)
        
        logger.info(f"Cache initialized at {self.cache_dir}")
    
    def _make_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, key: str, cache_type: str = "general") -> Path:
        """Get full path for cache file"""
        if cache_type == "translation":
            return self.translation_dir / f"{key}.pkl"
        elif cache_type == "kb":
            return self.kb_dir / f"{key}.pkl"
        elif cache_type == "embedding":
            return self.embeddings_dir / f"{key}.pkl"
        else:
            return self.cache_dir / f"{key}.pkl"
    
    def _is_expired(self, file_path: Path) -> bool:
        """Check if cache file is expired"""
        if not file_path.exists():
            return True
        
        file_age = time.time() - file_path.stat().st_mtime
        return file_age > self.ttl
    
    def _read_cache(self, key: str, cache_type: str = "general") -> Optional[Any]:
        """Read from cache if available and not expired"""
        cache_path = self._get_cache_path(key, cache_type)
        
        if not cache_path.exists():
            return None
        
        if self._is_expired(cache_path):
            logger.debug(f"Cache expired for key: {key}")
            try:
                cache_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete expired cache: {e}")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                logger.debug(f"Cache hit for key: {key}")
                return data
        except Exception as e:
            logger.error(f"Failed to read cache: {e}")
            return None
    
    def _write_cache(self, key: str, data: Any, cache_type: str = "general"):
        """Write data to cache"""
        cache_path = self._get_cache_path(key, cache_type)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Cache written for key: {key}")
        except Exception as e:
            logger.error(f"Failed to write cache: {e}")
    
    # Translation cache methods
    def get_translation(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Get cached translation"""
        cache_key = self._make_key(f"{source_lang}->{target_lang}:{text}")
        result = self._read_cache(cache_key, "translation")
        
        if result:
            logger.info(f"Translation cache hit: {source_lang} -> {target_lang}")
        
        return result
    
    def set_translation(self, text: str, source_lang: str, target_lang: str, translation: str):
        """Cache a translation"""
        cache_key = self._make_key(f"{source_lang}->{target_lang}:{text}")
        self._write_cache(cache_key, translation, "translation")
        logger.info(f"Translation cached: {source_lang} -> {target_lang}")
    
    # KB result cache methods
    def get_kb_result(self, query: str, lang: str = "en") -> Optional[Dict[str, Any]]:
        """Get cached KB search result"""
        cache_key = self._make_key(f"kb:{lang}:{query}")
        result = self._read_cache(cache_key, "kb")
        
        if result:
            logger.info(f"KB cache hit for query: {query[:50]}...")
        
        return result
    
    def set_kb_result(self, query: str, result: Dict[str, Any], lang: str = "en"):
        """Cache a KB search result"""
        cache_key = self._make_key(f"kb:{lang}:{query}")
        self._write_cache(cache_key, result, "kb")
        logger.info(f"KB result cached for query: {query[:50]}...")
    
    # Embedding cache methods
    def get_embedding(self, text: str) -> Optional[list]:
        """Get cached embedding"""
        cache_key = self._make_key(f"embed:{text}")
        return self._read_cache(cache_key, "embedding")
    
    def set_embedding(self, text: str, embedding: list):
        """Cache an embedding"""
        cache_key = self._make_key(f"embed:{text}")
        self._write_cache(cache_key, embedding, "embedding")
    
    # Cache management methods
    def clear_all(self):
        """Clear all cache files"""
        try:
            for directory in [self.translation_dir, self.kb_dir, self.embeddings_dir]:
                for file in directory.glob("*.pkl"):
                    file.unlink()
            logger.info("All cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def clear_expired(self):
        """Remove all expired cache files"""
        count = 0
        try:
            for directory in [self.translation_dir, self.kb_dir, self.embeddings_dir]:
                for file in directory.glob("*.pkl"):
                    if self._is_expired(file):
                        file.unlink()
                        count += 1
            logger.info(f"Removed {count} expired cache files")
        except Exception as e:
            logger.error(f"Failed to clear expired cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        stats = {
            "translations": len(list(self.translation_dir.glob("*.pkl"))),
            "kb_results": len(list(self.kb_dir.glob("*.pkl"))),
            "embeddings": len(list(self.embeddings_dir.glob("*.pkl"))),
        }
        stats["total"] = sum(stats.values())
        return stats
    
    def get_cache_size(self) -> int:
        """Get total cache size in bytes"""
        total_size = 0
        try:
            for directory in [self.translation_dir, self.kb_dir, self.embeddings_dir]:
                for file in directory.glob("*.pkl"):
                    total_size += file.stat().st_size
        except Exception as e:
            logger.error(f"Failed to calculate cache size: {e}")
        return total_size


# Global cache instance
_cache_instance: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get the global cache instance (singleton)"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance


def clear_cache():
    """Clear all cached data"""
    cache = get_cache()
    cache.clear_all()
