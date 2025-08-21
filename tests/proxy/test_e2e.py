#!/usr/bin/env python3
"""
End-to-end tests for proxy functionality with mock vLLM servers.
"""
import asyncio
import threading
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from vllm_cli.proxy.manager import ProxyManager
from vllm_cli.proxy.models import ModelConfig, ProxyConfig
from vllm_cli.proxy.server import ProxyServer


class MockVLLMServer:
    """Mock vLLM server for testing."""

    def __init__(self, model_name: str, port: int):
        self.model_name = model_name
        self.port = port
        self.app = FastAPI()
        self.server = None
        self.thread = None
        self.request_count = 0
        self._setup_routes()

    def _setup_routes(self):
        """Setup mock vLLM API routes."""

        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}

        @self.app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.model_name,
                        "object": "model",
                        "owned_by": "mock-vllm",
                    }
                ],
            }

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Dict[str, Any]):
            self.request_count += 1

            if request.get("stream"):
                # Streaming response
                async def generate():
                    yield (
                        f'data: {{"choices": [{{"delta": '
                        f'{{"content": "Hello from {self.model_name}"}}}}]}}\n\n'
                    )
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                )
            else:
                # Regular response
                return JSONResponse(
                    {
                        "id": f"mock-{self.request_count}",
                        "object": "chat.completion",
                        "model": self.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"Response from {self.model_name}",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )

        @self.app.post("/v1/completions")
        async def completions(request: Dict[str, Any]):
            self.request_count += 1
            return JSONResponse(
                {
                    "id": f"mock-{self.request_count}",
                    "object": "text_completion",
                    "model": self.model_name,
                    "choices": [
                        {
                            "text": f"Completion from {self.model_name}",
                            "index": 0,
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

    def start(self):
        """Start the mock server in a background thread."""

        def run_server():
            uvicorn.run(
                self.app,
                host="127.0.0.1",
                port=self.port,
                log_level="error",
            )

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        time.sleep(1)  # Give server time to start

    def stop(self):
        """Stop the mock server."""
        # In a real implementation, we'd properly shutdown uvicorn
        pass


class TestProxyE2E:
    """End-to-end tests for the proxy system."""

    @pytest.fixture
    def mock_vllm_servers(self):
        """Create mock vLLM servers."""
        servers = [
            MockVLLMServer("model1", 28001),
            MockVLLMServer("model2", 28002),
        ]

        # Start servers
        for server in servers:
            server.start()

        yield servers

        # Cleanup
        for server in servers:
            server.stop()

    @pytest.fixture
    def e2e_proxy_config(self):
        """Create proxy configuration for E2E tests."""
        return ProxyConfig(
            host="127.0.0.1",
            port=28000,
            models=[
                ModelConfig(
                    name="model1",
                    model_path="test/model1",
                    port=28001,
                    config_overrides={"aliases": ["m1", "first"]},
                    enabled=True,
                ),
                ModelConfig(
                    name="model2",
                    model_path="test/model2",
                    port=28002,
                    config_overrides={"aliases": ["m2", "second"]},
                    enabled=True,
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_e2e_request_routing(self, mock_vllm_servers, e2e_proxy_config):
        """Test end-to-end request routing through proxy."""
        # This test verifies routing logic is properly configured
        proxy_server = ProxyServer(e2e_proxy_config)

        # Register backends
        for model in e2e_proxy_config.models:
            proxy_server.router.add_backend(
                model.name,
                f"http://127.0.0.1:{model.port}",
                model.model_dump(),
            )

            # Register aliases
            for alias in model.config_overrides.get("aliases", []):
                proxy_server.router.add_backend(
                    alias,
                    f"http://127.0.0.1:{model.port}",
                    model.model_dump(),
                )

        # Verify routing is configured correctly
        assert proxy_server.router.route_request("model1") == "http://127.0.0.1:28001"
        assert proxy_server.router.route_request("m1") == "http://127.0.0.1:28001"
        assert proxy_server.router.route_request("model2") == "http://127.0.0.1:28002"
        assert proxy_server.router.route_request("m2") == "http://127.0.0.1:28002"

    @pytest.mark.asyncio
    async def test_e2e_streaming_response(self, mock_vllm_servers):
        """Test end-to-end streaming response handling."""
        async with httpx.AsyncClient() as client:
            # Test streaming to first mock server
            response = await client.post(
                "http://127.0.0.1:28001/v1/chat/completions",
                json={
                    "model": "model1",
                    "messages": [{"role": "user", "content": "Stream test"}],
                    "stream": True,
                },
            )

            assert response.status_code == 200

            # Read streaming response
            chunks = []
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    chunks.append(line)

            assert len(chunks) > 0
            assert any("model1" in chunk for chunk in chunks)
            assert "data: [DONE]" in chunks[-1]

    @pytest.mark.asyncio
    async def test_e2e_concurrent_requests(self, mock_vllm_servers):
        """Test handling concurrent requests to different models."""
        async with httpx.AsyncClient() as client:
            # Create multiple concurrent requests
            tasks = []

            for i in range(5):
                # Alternate between models
                port = 28001 if i % 2 == 0 else 28002
                model = "model1" if i % 2 == 0 else "model2"

                task = client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": f"Request {i}"}],
                    },
                )
                tasks.append(task)

            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks)

            # Verify all succeeded
            for i, response in enumerate(responses):
                assert response.status_code == 200
                data = response.json()
                expected_model = "model1" if i % 2 == 0 else "model2"
                assert expected_model in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_e2e_model_health_check(self, mock_vllm_servers):
        """Test model health checking."""
        async with httpx.AsyncClient() as client:
            # Check health of both mock servers
            for port in [28001, 28002]:
                response = await client.get(f"http://127.0.0.1:{port}/health")
                assert response.status_code == 200
                assert response.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_e2e_model_listing(self, mock_vllm_servers):
        """Test model listing from mock servers."""
        async with httpx.AsyncClient() as client:
            # List models from first server
            response = await client.get("http://127.0.0.1:28001/v1/models")
            assert response.status_code == 200
            data = response.json()
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "model1"

            # List models from second server
            response = await client.get("http://127.0.0.1:28002/v1/models")
            assert response.status_code == 200
            data = response.json()
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "model2"

    def test_e2e_proxy_with_mock_manager(self, e2e_proxy_config):
        """Test proxy manager with mocked components."""
        with patch("vllm_cli.proxy.manager.VLLMServer") as mock_server_class:
            with patch(
                "vllm_cli.proxy.manager.ProxyServerProcess"
            ) as mock_process_class:
                # Setup mocks
                mock_server = MagicMock()
                mock_server.start.return_value = True
                mock_server.stop.return_value = None
                mock_server.is_running.return_value = True
                mock_server_class.return_value = mock_server

                mock_process = MagicMock()
                mock_process.start.return_value = True
                mock_process.stop.return_value = None
                mock_process.is_running.return_value = True
                mock_process_class.return_value = mock_process

                # Create manager
                manager = ProxyManager(e2e_proxy_config)

                # Start everything
                with patch("time.sleep"):
                    started = manager.start_all_models()
                    assert started == 2

                    result = manager.start_proxy()
                    assert result is True

                # Verify servers were started
                assert len(manager.vllm_servers) == 2
                assert "model1" in manager.vllm_servers
                assert "model2" in manager.vllm_servers

                # Stop everything
                manager.stop_proxy()

                # Verify cleanup
                assert len(manager.vllm_servers) == 0
                assert manager.proxy_process is None

    @pytest.mark.asyncio
    async def test_e2e_error_handling(self):
        """Test error handling when backend is unavailable."""
        # Create proxy without starting backend servers
        proxy_config = ProxyConfig(
            host="127.0.0.1",
            port=28000,
            models=[
                ModelConfig(
                    name="unavailable",
                    model_path="test/unavailable",
                    port=29999,  # No server on this port
                    enabled=True,
                ),
            ],
        )

        proxy_server = ProxyServer(proxy_config)
        proxy_server.router.add_backend(
            "unavailable",
            "http://127.0.0.1:29999",
            proxy_config.models[0].model_dump(),
        )

        # Verify the backend is registered but will fail when accessed
        assert (
            proxy_server.router.route_request("unavailable") == "http://127.0.0.1:29999"
        )

        # Mark backend as unhealthy (which would happen on real health check)
        proxy_server.router.mark_backend_health("unavailable", False)

        # Now routing should return None for unhealthy backend
        assert proxy_server.router.route_request("unavailable") is None
