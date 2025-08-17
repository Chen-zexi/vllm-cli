"""Tests for model management and discovery."""

import time

from vllm_cli.models.cache import ModelCache
from vllm_cli.models.manager import ModelManager


class TestModelManager:
    """Test ModelManager functionality."""

    def test_init(self):
        """Test ModelManager initialization."""
        manager = ModelManager()
        assert manager.cache is not None
        assert isinstance(manager.cache, ModelCache)


class TestModelCache:
    """Test ModelCache functionality."""

    def test_cache_models_and_get(self):
        """Test caching and getting models."""
        cache = ModelCache()

        test_models = [{"name": "model1"}, {"name": "model2"}]
        cache.cache_models(test_models)

        retrieved = cache.get_cached_models()
        assert retrieved == test_models

    def test_cache_expiry(self):
        """Test cache expiry based on TTL."""
        cache = ModelCache(ttl_seconds=0.1)  # 100ms TTL

        test_models = [{"name": "model1"}]
        cache.cache_models(test_models)

        # Should be available immediately
        assert cache.get_cached_models() is not None

        # Wait for expiry
        time.sleep(0.15)

        # Should be expired
        assert cache.get_cached_models() is None
