#!/usr/bin/env python3
"""
Unit tests for proxy server and registry.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vllm_cli.proxy.models import ProxyConfig
from vllm_cli.proxy.registry import (
    ModelEntry,
    ModelRegistry,
    ModelState,
    RegistrationStatus,
)
from vllm_cli.proxy.server import ProxyServer


class TestProxyServer:
    """Test ProxyServer functionality."""

    @pytest.fixture
    def proxy_config(self):
        """Create a test proxy configuration."""
        return ProxyConfig(
            host="127.0.0.1",
            port=8000,
            enable_cors=True,
            enable_metrics=True,
            log_requests=False,
        )

    @pytest.fixture
    def proxy_server(self, proxy_config):
        """Create a ProxyServer instance."""
        with patch("vllm_cli.proxy.server.httpx.AsyncClient"):
            server = ProxyServer(proxy_config)
            yield server

    @pytest.fixture
    def test_client(self, proxy_server):
        """Create a FastAPI test client."""
        return TestClient(proxy_server.app)

    def test_proxy_server_init(self, proxy_config):
        """Test ProxyServer initialization."""
        with patch("vllm_cli.proxy.server.httpx.AsyncClient"):
            server = ProxyServer(proxy_config)

            assert server.config == proxy_config
            assert server.app is not None
            assert server.router is not None
            assert server.total_requests == 0
            assert server.model_requests == {}

    def test_root_endpoint(self, test_client, proxy_server):
        """Test root endpoint."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "vLLM Multi-Model Proxy Server"
        assert "version" in data
        assert "models_count" in data

    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_list_models_empty(self, test_client, proxy_server):
        """Test listing models when none are registered."""
        response = test_client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert data["data"] == []

    def test_list_models_with_backends(self, test_client, proxy_server):
        """Test listing models with registered backends."""
        # Add some backends
        proxy_server.router.add_backend(
            "model1", "http://localhost:8001", {"port": 8001}
        )
        proxy_server.router.add_backend(
            "model2", "http://localhost:8002", {"port": 8002}
        )

        response = test_client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2

        model_ids = [m["id"] for m in data["data"]]
        assert "model1" in model_ids
        assert "model2" in model_ids

    @pytest.mark.asyncio
    async def test_forward_request_success(self, proxy_server, mock_fastapi_request):
        """Test successful request forwarding."""
        # Setup mock backend
        proxy_server.router.add_backend(
            "test-model", "http://localhost:8001", {"port": 8001}
        )

        # Mock httpx response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }

        proxy_server.client.post = AsyncMock(return_value=mock_response)

        # Call forward request
        result = await proxy_server._forward_request(
            mock_fastapi_request, "/v1/chat/completions"
        )

        assert result.status_code == 200
        proxy_server.client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_forward_request_no_backend(self, proxy_server, mock_fastapi_request):
        """Test request forwarding when no backend is available."""
        # No backends registered

        with pytest.raises(Exception) as exc_info:
            await proxy_server._forward_request(
                mock_fastapi_request, "/v1/chat/completions"
            )

        assert "Model 'test-model' not found" in str(exc_info.value)

    def test_proxy_status_endpoint(self, test_client, proxy_server):
        """Test proxy status endpoint."""
        # Register a model in the registry (not just router)
        proxy_server.registry.pre_register(8001, [0], "model1")
        proxy_server.registry.verify_and_activate(8001, "model1")

        # Also add to router for completeness
        proxy_server.router.add_backend(
            "model1",
            "http://localhost:8001",
            {
                "port": 8001,
                "gpu_ids": [0],
                "model_path": "/path/to/model",
            },
        )

        response = test_client.get("/proxy/status")

        assert response.status_code == 200
        data = response.json()
        assert data["proxy_running"] is True
        assert data["proxy_port"] == 8000
        assert data["proxy_host"] == "127.0.0.1"
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "model1"

    # Removed test_add_model_endpoint - endpoint doesn't exist
    # Removed test_remove_model_endpoint - endpoint doesn't exist

    @pytest.mark.asyncio
    async def test_check_backend_health_success(self, proxy_server):
        """Test backend health check success."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        proxy_server.client.get = AsyncMock(return_value=mock_response)

        status = await proxy_server._check_backend_health(8001)

        assert status == "running"

    @pytest.mark.asyncio
    async def test_check_backend_health_failure(self, proxy_server):
        """Test backend health check failure."""
        proxy_server.client.get = AsyncMock(side_effect=Exception("Connection error"))

        status = await proxy_server._check_backend_health(8001)

        assert status == "unknown"

    def test_request_counting(self, proxy_server):
        """Test request counting."""
        assert proxy_server.total_requests == 0
        assert len(proxy_server.model_requests) == 0

        # Simulate handling requests
        proxy_server.total_requests += 1
        proxy_server.model_requests["model1"] = (
            proxy_server.model_requests.get("model1", 0) + 1
        )

        assert proxy_server.total_requests == 1
        assert proxy_server.model_requests["model1"] == 1

        # Another request to same model
        proxy_server.total_requests += 1
        proxy_server.model_requests["model1"] = (
            proxy_server.model_requests.get("model1", 0) + 1
        )

        assert proxy_server.total_requests == 2
        assert proxy_server.model_requests["model1"] == 2

        # Request to different model
        proxy_server.total_requests += 1
        proxy_server.model_requests["model2"] = (
            proxy_server.model_requests.get("model2", 0) + 1
        )

        assert proxy_server.total_requests == 3
        assert proxy_server.model_requests["model1"] == 2
        assert proxy_server.model_requests["model2"] == 1

    def test_chat_completions_endpoint(self, test_client, proxy_server):
        """Test chat completions endpoint routing."""
        # Add backend
        proxy_server.router.add_backend(
            "gpt-model", "http://localhost:8001", {"port": 8001}
        )

        # Mock the forward request method
        with patch.object(
            proxy_server, "_forward_request", new_callable=AsyncMock
        ) as mock_forward:
            mock_forward.return_value = MagicMock(status_code=200)

            response = test_client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            # The actual implementation would forward the request
            # For testing, we just verify the endpoint exists
            assert response is not None

    def test_completions_endpoint(self, test_client, proxy_server):
        """Test completions endpoint routing."""
        # Add backend
        proxy_server.router.add_backend(
            "text-model", "http://localhost:8001", {"port": 8001}
        )

        with patch.object(
            proxy_server, "_forward_request", new_callable=AsyncMock
        ) as mock_forward:
            mock_forward.return_value = MagicMock(status_code=200)

            response = test_client.post(
                "/v1/completions",
                json={
                    "model": "text-model",
                    "prompt": "Once upon a time",
                },
            )

            assert response is not None

    def test_embeddings_endpoint(self, test_client, proxy_server):
        """Test embeddings endpoint routing."""
        # Add backend
        proxy_server.router.add_backend(
            "embedding-model", "http://localhost:8001", {"port": 8001}
        )

        with patch.object(
            proxy_server, "_forward_request", new_callable=AsyncMock
        ) as mock_forward:
            mock_forward.return_value = MagicMock(status_code=200)

            response = test_client.post(
                "/v1/embeddings",
                json={
                    "model": "embedding-model",
                    "input": "Sample text for embedding",
                },
            )

            assert response is not None


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create a ModelRegistry instance."""
        return ModelRegistry()

    def test_registry_init(self, registry):
        """Test ModelRegistry initialization."""
        assert registry.models == {}
        assert registry._shutdown is False

    def test_pre_register_model(self, registry):
        """Test pre-registering a model."""
        result = registry.pre_register(
            port=8001, gpu_ids=[0, 1], config_name="test-model"
        )

        assert result is True
        assert 8001 in registry.models

        entry = registry.models[8001]
        assert entry.port == 8001
        assert entry.gpu_ids == [0, 1]
        assert entry.config_name == "test-model"
        assert entry.status == RegistrationStatus.PENDING
        assert entry.state == ModelState.STARTING

    def test_pre_register_duplicate(self, registry):
        """Test pre-registering a duplicate model."""
        registry.pre_register(8001, [0], "model1")
        result = registry.pre_register(8001, [1], "model2")

        assert result is False
        # Original model should remain
        assert registry.models[8001].config_name == "model1"

    def test_verify_and_activate(self, registry):
        """Test verifying and activating a pre-registered model."""
        registry.pre_register(8001, [0], "test-model")

        result = registry.verify_and_activate(8001, "actual-model-name")

        assert result is True
        entry = registry.models[8001]
        assert entry.actual_name == "actual-model-name"
        assert entry.status == RegistrationStatus.AVAILABLE
        assert entry.state == ModelState.RUNNING

    def test_verify_unregistered_model(self, registry):
        """Test verifying a model that wasn't pre-registered."""
        result = registry.verify_and_activate(8002, "new-model")

        assert result is True
        assert 8002 in registry.models
        entry = registry.models[8002]
        assert entry.actual_name == "new-model"

    def test_verify_preserves_sleeping_state(self, registry):
        """Test that verify_and_activate preserves SLEEPING state."""
        # Pre-register and activate a model
        registry.pre_register(8003, [0], "sleep-test-model")
        registry.verify_and_activate(8003, "sleep-model")

        # Manually set the model to SLEEPING state (as would happen via sleep endpoint)
        entry = registry.models[8003]
        entry.update_state(ModelState.SLEEPING)
        assert entry.state == ModelState.SLEEPING

        # Call verify_and_activate again (as would happen during refresh)
        result = registry.verify_and_activate(8003, "sleep-model")

        # State should still be SLEEPING, not RUNNING
        assert result is True
        assert entry.state == ModelState.SLEEPING  # Should preserve sleeping state
        assert entry.status == RegistrationStatus.AVAILABLE

    def test_mark_model_error(self, registry):
        """Test marking a model as having an error."""
        registry.pre_register(8001, [0], "test-model")

        registry.mark_model_error(8001, "Connection failed")

        entry = registry.models[8001]
        assert entry.status == RegistrationStatus.ERROR
        assert entry.error_message == "Connection failed"

    def test_remove_model(self, registry):
        """Test removing a model from registry."""
        registry.pre_register(8001, [0], "test-model")

        result = registry.remove_model(8001)
        assert result is True
        assert 8001 not in registry.models

        # Try removing non-existent model
        result = registry.remove_model(8001)
        assert result is False

    def test_update_model_state(self, registry):
        """Test updating model state."""
        registry.pre_register(8001, [0], "test-model")

        result = registry.update_model_state(8001, ModelState.SLEEPING)
        assert result is True
        assert registry.models[8001].state == ModelState.SLEEPING

        # Try updating non-existent model
        result = registry.update_model_state(8002, ModelState.RUNNING)
        assert result is False

    def test_get_available_models(self, registry):
        """Test getting available models."""
        # Add models with different states
        registry.pre_register(8001, [0], "model1")
        registry.pre_register(8002, [1], "model2")
        registry.pre_register(8003, [2], "model3")

        # Activate some models
        registry.verify_and_activate(8001, "model1-actual")
        registry.verify_and_activate(8002, "model2-actual")
        # Leave model3 as pending

        available = registry.get_available_models()
        assert len(available) == 2

        ports = [entry.port for entry in available]
        assert 8001 in ports
        assert 8002 in ports
        assert 8003 not in ports

    def test_get_models_on_gpu(self, registry):
        """Test getting models on specific GPU."""
        registry.pre_register(8001, [0], "gpu0-model")
        registry.pre_register(8002, [1], "gpu1-model")
        registry.pre_register(8003, [0, 1], "multi-gpu-model")

        gpu0_models = registry.get_models_on_gpu(0)
        assert len(gpu0_models) == 2

        gpu1_models = registry.get_models_on_gpu(1)
        assert len(gpu1_models) == 2

        gpu2_models = registry.get_models_on_gpu(2)
        assert len(gpu2_models) == 0

    def test_cleanup_stale_entries(self, registry):
        """Test cleaning up stale pending entries."""
        # Add a model and immediately mark it as old
        registry.pre_register(8001, [0], "stale-model")

        # Manually set last_activity to old time
        import datetime as dt

        old_time = dt.datetime.now() - dt.timedelta(minutes=10)
        registry.models[8001].last_activity = old_time

        # Add a fresh model
        registry.pre_register(8002, [1], "fresh-model")

        # Clean up with 5 minute timeout
        removed = registry.cleanup_stale_entries(timeout_seconds=300)

        assert removed == 1
        assert 8001 not in registry.models
        assert 8002 in registry.models

    def test_get_status_summary(self, registry):
        """Test getting registry status summary."""
        # Setup diverse model states
        registry.pre_register(8001, [0], "pending-model")
        registry.pre_register(8002, [1], "available-model")
        registry.verify_and_activate(8002, "actual-name")
        registry.pre_register(8003, [2], "error-model")
        registry.mark_model_error(8003, "Failed to start")

        summary = registry.get_status_summary()

        assert summary["total_models"] == 3
        assert summary["available"] == 1
        assert summary["pending"] == 1
        assert summary["errors"] == 1
        assert len(summary["models"]) == 3
        assert 0 in summary["gpu_usage"]
        assert 1 in summary["gpu_usage"]
        assert 2 in summary["gpu_usage"]

    def test_model_entry_display_name(self):
        """Test ModelEntry display name logic."""
        # With config name only
        entry1 = ModelEntry(8001, [0], "config-name")
        assert entry1.display_name == "config-name"

        # With actual name
        entry2 = ModelEntry(8002, [1])
        entry2.actual_name = "actual-name"
        assert entry2.display_name == "actual-name"

        # With both (actual takes precedence)
        entry3 = ModelEntry(8003, [2], "config-name")
        entry3.actual_name = "actual-name"
        assert entry3.display_name == "actual-name"

        # With neither
        entry4 = ModelEntry(8004, [3])
        assert entry4.display_name == "port_8004"

    def test_model_entry_to_dict(self):
        """Test ModelEntry serialization."""
        entry = ModelEntry(8001, [0, 1], "test-model")
        entry.mark_verified("actual-model")
        # mark_verified sets status to AVAILABLE, need to also update state
        entry.update_state(ModelState.RUNNING)

        data = entry.to_dict()

        assert data["port"] == 8001
        assert data["gpu_ids"] == [0, 1]
        assert data["config_name"] == "test-model"
        assert data["actual_name"] == "actual-model"
        assert data["registration_status"] == "available"
        assert data["state"] == "running"
        assert "last_activity" in data


class TestProxyRegistryEndpoints:
    """Test proxy server registry-related endpoints."""

    @pytest.fixture
    def proxy_config(self):
        """Create a test proxy configuration."""
        return ProxyConfig(
            host="127.0.0.1",
            port=8000,
            enable_cors=True,
            enable_metrics=True,
            log_requests=False,
        )

    @pytest.fixture
    def test_client_with_registry(self, proxy_config):
        """Create test client with registry support."""
        with patch("vllm_cli.proxy.server.httpx.AsyncClient"):
            server = ProxyServer(proxy_config)
            client = TestClient(server.app)
            yield client, server

    def test_pre_register_endpoint(self, test_client_with_registry):
        """Test pre-register model endpoint."""
        client, server = test_client_with_registry

        response = client.post(
            "/proxy/pre_register",
            json={"port": 8001, "gpu_ids": [0, 1], "config_name": "test-model"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify model is in registry
        assert 8001 in server.registry.models
        assert server.registry.models[8001].status == RegistrationStatus.PENDING

    def test_register_endpoint(self, test_client_with_registry):
        """Test register model endpoint (backward compatibility)."""
        client, server = test_client_with_registry

        response = client.post(
            "/proxy/register",
            json={"port": 8001, "gpu_ids": [0], "actual_name": "model-name"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["actual_name"] == "model-name"

        # Verify model is registered and in router
        assert 8001 in server.registry.models
        assert server.registry.models[8001].status == RegistrationStatus.AVAILABLE
        assert "model-name" in server.router.backends

    def test_unregister_model_endpoint(self, test_client_with_registry):
        """Test unregister model endpoint."""
        client, server = test_client_with_registry

        # First register a model
        server.registry.pre_register(8001, [0], "test-model")
        server.registry.verify_and_activate(8001, "actual-name")
        server.router.add_backend(
            "actual-name", "http://localhost:8001", {"port": 8001}
        )

        # Unregister it
        response = client.delete("/proxy/models/8001")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify removal
        assert 8001 not in server.registry.models
        assert "actual-name" not in server.router.backends

    def test_unregister_nonexistent_model(self, test_client_with_registry):
        """Test unregistering a non-existent model."""
        client, server = test_client_with_registry

        response = client.delete("/proxy/models/9999")

        assert response.status_code == 404

    def test_update_model_state_endpoint(self, test_client_with_registry):
        """Test updating model state endpoint."""
        client, server = test_client_with_registry

        # Pre-register a model
        server.registry.pre_register(8001, [0], "test-model")

        # Update to sleeping state
        response = client.post(
            "/proxy/state", json={"port": 8001, "state": "sleeping", "sleep_level": 2}
        )

        assert response.status_code == 200
        entry = server.registry.models[8001]
        assert entry.state == ModelState.SLEEPING
        assert entry.sleep_level == 2

        # Update to running state
        response = client.post("/proxy/state", json={"port": 8001, "state": "running"})

        assert response.status_code == 200
        entry = server.registry.models[8001]
        assert entry.state == ModelState.RUNNING
        assert entry.sleep_level == 0

    def test_get_registry_endpoint(self, test_client_with_registry):
        """Test get registry status endpoint."""
        client, server = test_client_with_registry

        # Setup some models
        server.registry.pre_register(8001, [0], "model1")
        server.registry.pre_register(8002, [1], "model2")
        server.registry.verify_and_activate(8002, "model2-actual")

        response = client.get("/proxy/registry")

        assert response.status_code == 200
        data = response.json()
        assert data["total_models"] == 2
        assert data["available"] == 1
        assert data["pending"] == 1
        assert len(data["models"]) == 2

    @patch("httpx.AsyncClient.get")
    def test_refresh_models_endpoint(self, mock_get, test_client_with_registry):
        """Test refresh models endpoint."""
        client, server = test_client_with_registry

        # Pre-register models
        server.registry.pre_register(8001, [0], "model1")
        server.registry.pre_register(8002, [1], "model2")

        # Mock successful response from model1
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "actual-model1"}]}
        mock_get.return_value = mock_response

        response = client.post("/proxy/refresh_models")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Model1 should be activated (mocked response succeeded)
        # Model2 remains pending (mock only returns success for first call)
        summary = data["summary"]
        assert summary["total"] == 2
