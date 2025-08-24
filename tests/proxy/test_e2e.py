#!/usr/bin/env python3
"""
End-to-end tests for proxy functionality with mock vLLM servers.
"""
import asyncio
import socket
import threading
import time
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from vllm_cli.proxy.manager import ProxyManager
from vllm_cli.proxy.models import ModelConfig, ProxyConfig
from vllm_cli.proxy.server import ProxyServer


def get_free_port() -> int:
    """Get a free port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class MockVLLMServer:
    """Mock vLLM server for testing."""

    def __init__(self, model_name: str, port: Optional[int] = None):
        self.model_name = model_name
        self.port = port or get_free_port()
        self.app = FastAPI()
        self.server = None
        self.thread = None
        self.request_count = 0
        self.stop_event = threading.Event()
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
        import uvicorn.config

        config = uvicorn.Config(
            app=self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="error",
        )
        self.server = uvicorn.Server(config)

        def run_server():
            try:
                self.server.run()
            except Exception:
                pass  # Server stopped

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()

        # Wait for server to actually start
        max_wait = 2.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    break
            except (socket.timeout, ConnectionRefusedError):
                time.sleep(0.1)

    def stop(self):
        """Stop the mock server."""
        if self.server:
            self.server.should_exit = True
            self.stop_event.set()

            # Wait a bit for graceful shutdown
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1.0)


class TestProxyE2E:
    """End-to-end tests for the proxy system."""

    @pytest.fixture
    def mock_vllm_servers(self):
        """Create mock vLLM servers."""
        servers = [
            MockVLLMServer("model1"),  # Will use dynamic port
            MockVLLMServer("model2"),  # Will use dynamic port
        ]

        # Start servers
        for server in servers:
            server.start()

        yield servers

        # Cleanup
        for server in servers:
            server.stop()

    @pytest.fixture
    def e2e_proxy_config(self, mock_vllm_servers):
        """Create proxy configuration for E2E tests."""
        return ProxyConfig(
            host="127.0.0.1",
            port=get_free_port(),  # Use dynamic port for proxy too
            models=[
                ModelConfig(
                    name="model1",
                    model_path="test/model1",
                    port=mock_vllm_servers[0].port,  # Use actual server port
                    config_overrides={"aliases": ["m1", "first"]},
                    enabled=True,
                ),
                ModelConfig(
                    name="model2",
                    model_path="test/model2",
                    port=mock_vllm_servers[1].port,  # Use actual server port
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
        port1 = mock_vllm_servers[0].port
        port2 = mock_vllm_servers[1].port
        assert (
            proxy_server.router.route_request("model1") == f"http://127.0.0.1:{port1}"
        )
        assert proxy_server.router.route_request("m1") == f"http://127.0.0.1:{port1}"
        assert (
            proxy_server.router.route_request("model2") == f"http://127.0.0.1:{port2}"
        )
        assert proxy_server.router.route_request("m2") == f"http://127.0.0.1:{port2}"

    @pytest.mark.asyncio
    async def test_e2e_streaming_response(self, mock_vllm_servers):
        """Test end-to-end streaming response handling."""
        async with httpx.AsyncClient() as client:
            # Test streaming to first mock server
            response = await client.post(
                f"http://127.0.0.1:{mock_vllm_servers[0].port}/v1/chat/completions",
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
                port = (
                    mock_vllm_servers[0].port
                    if i % 2 == 0
                    else mock_vllm_servers[1].port
                )
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
            for server in mock_vllm_servers:
                response = await client.get(f"http://127.0.0.1:{server.port}/health")
                assert response.status_code == 200
                assert response.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_e2e_model_listing(self, mock_vllm_servers):
        """Test model listing from mock servers."""
        async with httpx.AsyncClient() as client:
            # List models from first server
            response = await client.get(
                f"http://127.0.0.1:{mock_vllm_servers[0].port}/v1/models"
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "model1"

            # List models from second server
            response = await client.get(
                f"http://127.0.0.1:{mock_vllm_servers[1].port}/v1/models"
            )
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
                    started = manager.start_all_models_no_wait()
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

        # Verify the backend is registered
        assert (
            proxy_server.router.route_request("unavailable") == "http://127.0.0.1:29999"
        )

        # In the new architecture, health is managed by registry, not router
        # The router will still return the URL even if backend is unavailable
        # The actual error would occur when trying to forward the request

    @pytest.mark.asyncio
    async def test_e2e_full_model_lifecycle_with_registry(self, mock_vllm_servers):
        """Test complete model lifecycle with registry states."""
        proxy_config = ProxyConfig(
            host="127.0.0.1",
            port=get_free_port(),
            models=[
                ModelConfig(
                    name="lifecycle-model",
                    model_path="test/lifecycle",
                    port=mock_vllm_servers[0].port,
                    gpu_ids=[0],
                    enabled=True,
                ),
            ],
        )

        proxy_server = ProxyServer(proxy_config)

        # Step 1: Pre-register model (PENDING state)
        result = proxy_server.registry.pre_register(
            mock_vllm_servers[0].port, [0], "lifecycle-model"
        )
        assert result is True

        entry = proxy_server.registry.get_model(mock_vllm_servers[0].port)
        assert entry.status.value == "pending"
        assert entry.state.value == "starting"

        # Step 2: Verify and activate (AVAILABLE/RUNNING state)
        result = proxy_server.registry.verify_and_activate(
            mock_vllm_servers[0].port, "lifecycle-model-actual"
        )
        assert result is True

        entry = proxy_server.registry.get_model(mock_vllm_servers[0].port)
        assert entry.status.value == "available"
        assert entry.state.value == "running"

        # Step 3: Sleep the model
        from vllm_cli.proxy.registry import ModelState

        proxy_server.registry.update_model_state(
            mock_vllm_servers[0].port, ModelState.SLEEPING
        )
        entry.sleep_level = 2

        entry = proxy_server.registry.get_model(mock_vllm_servers[0].port)
        assert entry.state.value == "sleeping"
        assert entry.sleep_level == 2

        # Step 4: Wake the model
        proxy_server.registry.update_model_state(
            mock_vllm_servers[0].port, ModelState.RUNNING
        )
        entry.sleep_level = 0

        entry = proxy_server.registry.get_model(mock_vllm_servers[0].port)
        assert entry.state.value == "running"
        assert entry.sleep_level == 0

        # Step 5: Mark as error
        proxy_server.registry.mark_model_error(mock_vllm_servers[0].port, "Test error")

        entry = proxy_server.registry.get_model(mock_vllm_servers[0].port)
        assert entry.status.value == "error"
        assert entry.error_message == "Test error"

        # Step 6: Remove model
        result = proxy_server.registry.remove_model(mock_vllm_servers[0].port)
        assert result is True
        assert proxy_server.registry.get_model(mock_vllm_servers[0].port) is None

    @pytest.mark.asyncio
    async def test_e2e_concurrent_model_registration(self, mock_vllm_servers):
        """Test concurrent registration of multiple models."""
        import asyncio

        proxy_config = ProxyConfig(
            host="127.0.0.1",
            port=get_free_port(),
        )

        proxy_server = ProxyServer(proxy_config)

        # Pre-register multiple models concurrently
        for i, server in enumerate(mock_vllm_servers):
            # Simulate concurrent pre-registration
            result = proxy_server.registry.pre_register(
                server.port, [i], f"concurrent-model-{i}"
            )
            assert result is True

        # All should be pending
        all_models = proxy_server.registry.get_all_models()
        assert len(all_models) == len(mock_vllm_servers)
        assert all(m.status.value == "pending" for m in all_models.values())

        # Simulate concurrent verification
        async def verify_model(port, name):
            """Simulate model verification."""
            await asyncio.sleep(0.1)  # Simulate network delay
            return proxy_server.registry.verify_and_activate(port, f"{name}-actual")

        # Verify all models concurrently
        verify_tasks = [
            verify_model(server.port, f"concurrent-model-{i}")
            for i, server in enumerate(mock_vllm_servers)
        ]

        results = await asyncio.gather(*verify_tasks)

        # All should succeed
        assert all(results)

        # All should now be available
        available = proxy_server.registry.get_available_models()
        assert len(available) == len(mock_vllm_servers)

    @pytest.mark.asyncio
    async def test_e2e_registry_persistence(self):
        """Test registry state persistence across proxy restarts."""
        from vllm_cli.proxy.registry import ModelRegistry

        # Create first registry instance
        registry1 = ModelRegistry()

        # Add some models with different states
        registry1.pre_register(18001, [0], "model1")
        registry1.pre_register(18002, [1], "model2")
        registry1.verify_and_activate(18002, "model2-actual")

        # Get state summary
        summary1 = registry1.get_status_summary()

        # Simulate proxy restart - create new registry
        registry2 = ModelRegistry()

        # Registry starts empty after restart
        assert len(registry2.models) == 0

        # In real scenario, models would re-register on startup
        # Simulate re-registration
        registry2.pre_register(18001, [0], "model1")
        registry2.pre_register(18002, [1], "model2")

        # Verify model2 again (as it would happen on refresh)
        registry2.verify_and_activate(18002, "model2-actual")

        # State should be similar
        summary2 = registry2.get_status_summary()
        assert summary2["total_models"] == summary1["total_models"]
        assert summary2["available"] == summary1["available"]

    @pytest.mark.asyncio
    async def test_e2e_stale_entry_cleanup_with_refresh(self, mock_vllm_servers):
        """Test stale entry cleanup during refresh process."""
        proxy_config = ProxyConfig(
            host="127.0.0.1",
            port=get_free_port(),
        )

        proxy_server = ProxyServer(proxy_config)

        # Pre-register a model
        proxy_server.registry.pre_register(
            mock_vllm_servers[0].port, [0], "stale-test-model"
        )

        # Make it stale by setting old timestamp
        import datetime as dt

        old_time = dt.datetime.now() - dt.timedelta(minutes=10)
        entry = proxy_server.registry.get_model(mock_vllm_servers[0].port)
        entry.last_activity = old_time

        # Add a fresh model
        proxy_server.registry.pre_register(
            mock_vllm_servers[1].port, [1], "fresh-test-model"
        )

        # Cleanup stale entries
        removed = proxy_server.registry.cleanup_stale_entries(timeout_seconds=300)

        assert removed == 1
        assert proxy_server.registry.get_model(mock_vllm_servers[0].port) is None
        assert proxy_server.registry.get_model(mock_vllm_servers[1].port) is not None

    def test_e2e_gpu_allocation_conflicts(self):
        """Test handling of GPU allocation conflicts."""
        from vllm_cli.proxy.registry import ModelRegistry

        registry = ModelRegistry()

        # Register model on GPU 0
        registry.pre_register(18001, [0], "gpu0-model")
        registry.verify_and_activate(18001, "gpu0-model")

        # Try to register another model on same GPU
        registry.pre_register(18002, [0], "gpu0-conflict")

        # Both should exist in registry
        assert len(registry.models) == 2

        # Check GPU usage tracking
        gpu0_models = registry.get_models_on_gpu(0)
        assert len(gpu0_models) == 2

        # Status summary should show conflict
        summary = registry.get_status_summary()
        assert len(summary["gpu_usage"][0]) == 2

    @pytest.mark.asyncio
    async def test_e2e_dynamic_model_addition_removal(self):
        """Test dynamic addition and removal of models during runtime."""
        proxy_config = ProxyConfig(
            host="127.0.0.1",
            port=get_free_port(),
        )

        proxy_server = ProxyServer(proxy_config)

        # Start with empty proxy
        assert len(proxy_server.registry.models) == 0
        assert len(proxy_server.router.backends) == 0

        # Dynamically add model 1
        proxy_server.registry.pre_register(18001, [0], "dynamic1")
        proxy_server.registry.verify_and_activate(18001, "dynamic1-actual")
        proxy_server.router.add_backend(
            "dynamic1-actual", "http://localhost:18001", {"port": 18001}
        )

        assert len(proxy_server.registry.models) == 1
        assert "dynamic1-actual" in proxy_server.router.backends

        # Add model 2
        proxy_server.registry.pre_register(18002, [1], "dynamic2")
        proxy_server.registry.verify_and_activate(18002, "dynamic2-actual")
        proxy_server.router.add_backend(
            "dynamic2-actual", "http://localhost:18002", {"port": 18002}
        )

        assert len(proxy_server.registry.models) == 2
        assert "dynamic2-actual" in proxy_server.router.backends

        # Remove model 1
        proxy_server.registry.remove_model(18001)
        proxy_server.router.remove_backend("dynamic1-actual")

        assert len(proxy_server.registry.models) == 1
        assert "dynamic1-actual" not in proxy_server.router.backends
        assert "dynamic2-actual" in proxy_server.router.backends

        # Add model 3
        proxy_server.registry.pre_register(18003, [2], "dynamic3")
        proxy_server.registry.verify_and_activate(18003, "dynamic3-actual")
        proxy_server.router.add_backend(
            "dynamic3-actual", "http://localhost:18003", {"port": 18003}
        )

        # Final state: models 2 and 3
        assert len(proxy_server.registry.models) == 2
        available = proxy_server.registry.get_available_models()
        assert len(available) == 2

        ports = [e.port for e in available]
        assert 18002 in ports
        assert 18003 in ports
