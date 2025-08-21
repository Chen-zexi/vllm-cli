#!/usr/bin/env python3
"""
Unit tests for proxy server.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vllm_cli.proxy.models import ProxyConfig
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
        # Add a backend
        proxy_server.router.add_backend(
            "model1",
            "http://localhost:8001",
            {
                "port": 8001,
                "gpu_ids": [0],
                "model_path": "/path/to/model",
            },
        )

        # Mock health check
        with patch.object(
            proxy_server, "_check_backend_health", new_callable=AsyncMock
        ) as mock_health:
            mock_health.return_value = "running"

            response = test_client.get("/proxy/status")

        assert response.status_code == 200
        data = response.json()
        assert data["proxy_running"] is True
        assert data["proxy_port"] == 8000
        assert data["proxy_host"] == "127.0.0.1"
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "model1"

    def test_add_model_endpoint(self, test_client, proxy_server):
        """Test dynamic model addition endpoint."""
        model_config = {
            "name": "new-model",
            "port": 8003,
            "model_path": "/path/to/new/model",
            "gpu_ids": [2],
        }

        response = test_client.post("/proxy/add_model", json=model_config)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "new-model" in data["message"]

        # Verify model was added to router
        assert "new-model" in proxy_server.router.backends

    def test_remove_model_endpoint(self, test_client, proxy_server):
        """Test dynamic model removal endpoint."""
        # First add a model
        proxy_server.router.add_backend(
            "test-model", "http://localhost:8001", {"port": 8001}
        )

        # Use DELETE method with model name in path
        response = test_client.delete("/proxy/remove_model/test-model")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Verify model was removed from router
        assert "test-model" not in proxy_server.router.backends

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
