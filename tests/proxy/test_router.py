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

    def test_add_multiple_backends(self, router):
        """Test adding multiple backends."""
        router.add_backend("model1", "http://localhost:8001", {"port": 8001})
        router.add_backend("model2", "http://localhost:8002", {"port": 8002})
        router.add_backend("model3", "http://localhost:8003", {"port": 8003})

        assert len(router.backends) == 3

    def test_remove_backend(self, router):
        """Test removing a backend."""
        router.add_backend("model1", "http://localhost:8001", {"port": 8001})
        router.add_backend("model2", "http://localhost:8002", {"port": 8002})

        router.remove_backend("model1")

        assert "model1" not in router.backends
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

    def test_route_request_no_match(self, router):
        """Test routing when no backend matches."""
        router.add_backend("model1", "http://localhost:8001", {"port": 8001})

        url = router.route_request("nonexistent-model")
        assert url is None

    def test_wildcard_backend(self, router):
        """Test wildcard/default backend routing."""
        router.add_backend("model1", "http://localhost:8001", {"port": 8001})
        router.add_backend("*", "http://localhost:9999", {"port": 9999})

        # Specific model should match first
        assert router.route_request("model1") == "http://localhost:8001"

        # Unknown model should match wildcard
        assert router.route_request("unknown-model") == "http://localhost:9999"

    def test_get_active_models(self, router):
        """Test getting list of active models."""
        router.add_backend("model1", "http://localhost:8001", {"port": 8001})
        router.add_backend("model2", "http://localhost:8002", {"port": 8002})
        router.add_backend("model3", "http://localhost:8003", {"port": 8003})

        # All models are active (no health tracking in router anymore)
        active = router.get_active_models()
        assert len(active) == 3
        assert set(active) == {"model1", "model2", "model3"}

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

    def test_simple_routing_scenario(self, router):
        """Test a simple routing scenario with multiple models."""
        # Add multiple backends with various configurations
        router.add_backend(
            "meta-llama/Llama-2-7b-chat-hf",
            "http://localhost:8001",
            {"port": 8001, "gpu_ids": [0]},
        )
        router.add_backend(
            "mistralai/Mistral-7B-Instruct-v0.2",
            "http://localhost:8002",
            {"port": 8002, "gpu_ids": [1]},
        )
        router.add_backend(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "http://localhost:8003",
            {"port": 8003, "gpu_ids": [2]},
        )

        # Test exact model names
        assert (
            router.route_request("meta-llama/Llama-2-7b-chat-hf")
            == "http://localhost:8001"
        )
        assert (
            router.route_request("mistralai/Mistral-7B-Instruct-v0.2")
            == "http://localhost:8002"
        )
        assert (
            router.route_request("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            == "http://localhost:8003"
        )

        # Non-existent model returns None
        assert router.route_request("unknown-model") is None

        # All models should be active
        assert len(router.get_active_models()) == 3
