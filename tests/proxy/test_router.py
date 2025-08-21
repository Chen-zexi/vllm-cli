#!/usr/bin/env python3
"""
Unit tests for request router.
"""
import pytest

from vllm_cli.proxy.router import RequestRouter


class TestRequestRouter:
    """Test RequestRouter functionality."""

    @pytest.fixture
    def router(self):
        """Create a RequestRouter instance."""
        return RequestRouter()

    def test_add_backend(self, router):
        """Test adding a backend."""
        router.add_backend(
            "model1",
            "http://localhost:8001",
            {"port": 8001, "gpu_ids": [0], "model_path": "/path/to/model"},
        )

        assert "model1" in router.backends
        assert router.backends["model1"]["url"] == "http://localhost:8001"
        assert router.backends["model1"]["port"] == 8001
        assert router.backends["model1"]["gpu_ids"] == [0]
        assert router.backend_health["model1"] is True

    def test_add_multiple_backends(self, router):
        """Test adding multiple backends."""
        router.add_backend("model1", "http://localhost:8001", {"port": 8001})
        router.add_backend("model2", "http://localhost:8002", {"port": 8002})
        router.add_backend("model3", "http://localhost:8003", {"port": 8003})

        assert len(router.backends) == 3
        assert len(router.backend_health) == 3
        assert all(router.backend_health.values())

    def test_remove_backend(self, router):
        """Test removing a backend."""
        router.add_backend("model1", "http://localhost:8001", {"port": 8001})
        router.add_backend("model2", "http://localhost:8002", {"port": 8002})

        router.remove_backend("model1")

        assert "model1" not in router.backends
        assert "model1" not in router.backend_health
        assert "model2" in router.backends

    def test_remove_nonexistent_backend(self, router):
        """Test removing a backend that doesn't exist."""
        with pytest.raises(KeyError, match="Model 'nonexistent' not found"):
            router.remove_backend("nonexistent")

    def test_route_request_exact_match(self, router):
        """Test routing with exact model name match."""
        router.add_backend("llama-7b", "http://localhost:8001", {"port": 8001})

        url = router.route_request("llama-7b")
        assert url == "http://localhost:8001"

    def test_route_request_with_aliases(self, router):
        """Test routing with model aliases."""
        router.add_backend(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "http://localhost:8001",
            {"port": 8001, "aliases": ["tiny", "tinyllama", "small"]},
        )

        # Test exact match
        url = router.route_request("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        assert url == "http://localhost:8001"

        # Test alias matches
        assert router.route_request("tiny") == "http://localhost:8001"
        assert router.route_request("tinyllama") == "http://localhost:8001"
        assert router.route_request("small") == "http://localhost:8001"

    def test_route_request_no_match(self, router):
        """Test routing when no backend matches."""
        router.add_backend("model1", "http://localhost:8001", {"port": 8001})

        url = router.route_request("nonexistent-model")
        assert url is None

    def test_route_request_unhealthy_backend(self, router):
        """Test routing when backend is unhealthy."""
        router.add_backend("model1", "http://localhost:8001", {"port": 8001})
        router.mark_backend_health("model1", False)

        url = router.route_request("model1")
        assert url is None

    def test_wildcard_backend(self, router):
        """Test wildcard/default backend routing."""
        router.add_backend("model1", "http://localhost:8001", {"port": 8001})
        router.add_backend("*", "http://localhost:9999", {"port": 9999})

        # Specific model should match first
        assert router.route_request("model1") == "http://localhost:8001"

        # Unknown model should match wildcard
        assert router.route_request("unknown-model") == "http://localhost:9999"

    def test_wildcard_backend_unhealthy(self, router):
        """Test wildcard backend when unhealthy."""
        router.add_backend("*", "http://localhost:9999", {"port": 9999})
        router.mark_backend_health("*", False)

        url = router.route_request("any-model")
        assert url is None

    def test_get_active_models(self, router):
        """Test getting list of active models."""
        router.add_backend("model1", "http://localhost:8001", {"port": 8001})
        router.add_backend("model2", "http://localhost:8002", {"port": 8002})
        router.add_backend("model3", "http://localhost:8003", {"port": 8003})

        # All healthy
        active = router.get_active_models()
        assert len(active) == 3
        assert set(active) == {"model1", "model2", "model3"}

        # Mark one unhealthy
        router.mark_backend_health("model2", False)
        active = router.get_active_models()
        assert len(active) == 2
        assert set(active) == {"model1", "model3"}

    def test_get_backends(self, router):
        """Test getting all backend configurations."""
        config1 = {"port": 8001, "gpu_ids": [0]}
        config2 = {"port": 8002, "gpu_ids": [1]}

        router.add_backend("model1", "http://localhost:8001", config1)
        router.add_backend("model2", "http://localhost:8002", config2)

        backends = router.get_backends()
        assert len(backends) == 2
        assert backends["model1"]["port"] == 8001
        assert backends["model2"]["port"] == 8002

        # Note: get_backends returns a shallow copy
        # The outer dict is copied but inner dicts are referenced
        # This is acceptable for the use case
        assert backends is not router.backends  # Different dict objects
        assert backends["model1"] is router.backends["model1"]  # Same inner dict

    def test_mark_backend_health(self, router):
        """Test marking backend health status."""
        router.add_backend("model1", "http://localhost:8001", {"port": 8001})

        # Initially healthy
        assert router.backend_health["model1"] is True

        # Mark unhealthy
        router.mark_backend_health("model1", False)
        assert router.backend_health["model1"] is False

        # Mark healthy again
        router.mark_backend_health("model1", True)
        assert router.backend_health["model1"] is True

    def test_mark_backend_health_nonexistent(self, router):
        """Test marking health for nonexistent backend."""
        # Should not raise error, just log warning
        router.mark_backend_health("nonexistent", True)
        assert "nonexistent" not in router.backend_health

    def test_get_backend_for_gpu(self, router):
        """Test finding model by GPU ID."""
        router.add_backend(
            "model1", "http://localhost:8001", {"port": 8001, "gpu_ids": [0, 1]}
        )
        router.add_backend(
            "model2", "http://localhost:8002", {"port": 8002, "gpu_ids": [2]}
        )
        router.add_backend(
            "model3", "http://localhost:8003", {"port": 8003, "gpu_ids": [3, 4, 5]}
        )

        assert router.get_backend_for_gpu(0) == "model1"
        assert router.get_backend_for_gpu(1) == "model1"
        assert router.get_backend_for_gpu(2) == "model2"
        assert router.get_backend_for_gpu(3) == "model3"
        assert router.get_backend_for_gpu(4) == "model3"
        assert router.get_backend_for_gpu(5) == "model3"
        assert router.get_backend_for_gpu(6) is None

    def test_load_balancing_route(self, router):
        """Test load balancing route (currently same as regular route)."""
        router.add_backend("model1", "http://localhost:8001", {"port": 8001})

        # Currently just delegates to route_request
        url = router.load_balancing_route("model1")
        assert url == "http://localhost:8001"

        # Test with unhealthy backend
        router.mark_backend_health("model1", False)
        url = router.load_balancing_route("model1")
        assert url is None

    def test_complex_routing_scenario(self, router):
        """Test a complex routing scenario with multiple models and aliases."""
        # Add multiple backends with various configurations
        router.add_backend(
            "meta-llama/Llama-2-7b-chat-hf",
            "http://localhost:8001",
            {"port": 8001, "gpu_ids": [0], "aliases": ["llama2-7b", "llama", "chat"]},
        )
        router.add_backend(
            "mistralai/Mistral-7B-Instruct-v0.2",
            "http://localhost:8002",
            {"port": 8002, "gpu_ids": [1], "aliases": ["mistral", "mistral-7b"]},
        )
        router.add_backend(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "http://localhost:8003",
            {"port": 8003, "gpu_ids": [2], "aliases": ["tiny", "fast"]},
        )

        # Test various routing scenarios
        assert router.route_request("llama") == "http://localhost:8001"
        assert router.route_request("mistral") == "http://localhost:8002"
        assert router.route_request("tiny") == "http://localhost:8003"
        assert router.route_request("fast") == "http://localhost:8003"

        # Test exact model names
        assert (
            router.route_request("meta-llama/Llama-2-7b-chat-hf")
            == "http://localhost:8001"
        )
        assert (
            router.route_request("mistralai/Mistral-7B-Instruct-v0.2")
            == "http://localhost:8002"
        )

        # Test GPU assignment
        assert router.get_backend_for_gpu(0) == "meta-llama/Llama-2-7b-chat-hf"
        assert router.get_backend_for_gpu(1) == "mistralai/Mistral-7B-Instruct-v0.2"
        assert router.get_backend_for_gpu(2) == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # Mark one backend unhealthy
        router.mark_backend_health("mistralai/Mistral-7B-Instruct-v0.2", False)
        assert router.route_request("mistral") is None
        assert len(router.get_active_models()) == 2
