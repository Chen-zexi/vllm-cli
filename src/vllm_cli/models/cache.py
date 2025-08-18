#!/usr/bin/env python3
"""
Model caching utilities for vLLM CLI.

Provides caching functionality for model discovery results to improve
performance when repeatedly accessing model information.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelCache:
    """
    Cache for model discovery results.

    Provides time-based caching of model listings to avoid expensive
    directory scanning operations when model data is accessed frequently.
    """

    def __init__(self, ttl_seconds: float = 30.0):
        """
        Initialize model cache.

        Args:
            ttl_seconds: Time-to-live for cached data in seconds
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
        self._stats = {"hits": 0, "misses": 0, "updates": 0}

    def get_cached_models(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached model list if still valid.

        Returns:
            List of cached models if cache is valid, None if expired or empty
        """
        if self._cache is None:
            self._stats["misses"] += 1
            return None

        cache_time, cached_data = self._cache

        # Check if cache is still valid
        if time.time() - cache_time < self.ttl_seconds:
            self._stats["hits"] += 1
            logger.debug(f"Cache hit: returning {len(cached_data)} cached models")
            return cached_data.copy()  # Return copy to prevent modification
        else:
            # Cache expired
            self._stats["misses"] += 1
            logger.debug("Cache expired")
            self._cache = None
            return None

    def cache_models(self, models: List[Dict[str, Any]]) -> None:
        """
        Cache a list of models.

        Args:
            models: List of model dictionaries to cache
        """
        self._cache = (time.time(), models.copy())
        self._stats["updates"] += 1
        logger.debug(f"Cached {len(models)} models")

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._cache = None
        # Increment misses to reflect that the next access will be a miss
        self._stats["misses"] += 1
        logger.debug("Model cache cleared")

    def is_cached(self) -> bool:
        """
        Check if there is valid cached data.

        Returns:
            True if cache contains valid data, False otherwise
        """
        if self._cache is None:
            return False

        cache_time, _ = self._cache
        return time.time() - cache_time < self.ttl_seconds

    def get_cache_age(self) -> Optional[float]:
        """
        Get the age of cached data in seconds.

        Returns:
            Age in seconds if cache exists, None if no cache
        """
        if self._cache is None:
            return None

        cache_time, _ = self._cache
        return time.time() - cache_time

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache performance statistics
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        )

        stats = {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "updates": self._stats["updates"],
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "is_cached": self.is_cached(),
            "cache_age_seconds": self.get_cache_age(),
            "ttl_seconds": self.ttl_seconds,
        }

        if self._cache:
            _, cached_data = self._cache
            stats["cached_models_count"] = len(cached_data)
        else:
            stats["cached_models_count"] = 0

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """
        Alias for get_cache_stats() for backward compatibility.

        Returns:
            Dictionary with cache performance statistics
        """
        return self.get_cache_stats()

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = {"hits": 0, "misses": 0, "updates": 0}
        logger.debug("Cache statistics reset")

    def set_ttl(self, ttl_seconds: float) -> None:
        """
        Set new TTL for cache.

        Args:
            ttl_seconds: New time-to-live in seconds
        """
        old_ttl = self.ttl_seconds
        self.ttl_seconds = ttl_seconds
        logger.debug(f"Cache TTL changed from {old_ttl}s to {ttl_seconds}s")

    def get_cached_model_count(self) -> int:
        """
        Get the number of models currently cached.

        Returns:
            Number of cached models, 0 if no cache
        """
        if self._cache is None:
            return 0

        _, cached_data = self._cache
        return len(cached_data)


# Global cache instance for backward compatibility
_global_cache = ModelCache()


def get_global_cache() -> ModelCache:
    """Get the global model cache instance."""
    return _global_cache


def cache_models(models: List[Dict[str, Any]]) -> None:
    """Cache models using global cache (convenience function)."""
    _global_cache.cache_models(models)


def get_cached_models() -> Optional[List[Dict[str, Any]]]:
    """Get cached models from global cache (convenience function)."""
    return _global_cache.get_cached_models()


def clear_global_cache() -> None:
    """Clear the global model cache (convenience function)."""
    _global_cache.clear_cache()


def get_global_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics (convenience function)."""
    return _global_cache.get_cache_stats()
