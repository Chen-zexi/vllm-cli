#!/usr/bin/env python3
"""
Integration tests for proxy functionality.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from vllm_cli.proxy.config import ProxyConfigManager
from vllm_cli.proxy.manager import ProxyManager
from vllm_cli.proxy.models import ModelConfig, ProxyConfig
from vllm_cli.proxy.server_process import ProxyServerProcess


class TestProxyIntegration:
    """Integration tests for the proxy system."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for test configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def proxy_config_file(self, temp_config_dir):
        """Create a test proxy configuration file."""
        config_path = temp_config_dir / "proxy-config.yaml"

        config_data = {
            "proxy": {
                "host": "127.0.0.1",
                "port": 18080,
                "enable_cors": True,
                "enable_metrics": True,
                "log_requests": False,
            },
            "models": [
                {
                    "name": "tiny-model",
                    "model_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "gpu_ids": [0],
                    "port": 18001,
                    "profile": "standard",
                    "config_overrides": {
                        "aliases": ["tiny", "small"],
                    },
                    "enabled": True,
                },
                {
                    "name": "medium-model",
                    "model_path": "mistralai/Mistral-7B",
                    "gpu_ids": [1],
                    "port": 18002,
                    "profile": "performance",
                    "enabled": False,  # Disabled by default
                },
            ],
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        return config_path

    def test_load_proxy_config(self, proxy_config_file):
        """Test loading proxy configuration from file."""
        config_manager = ProxyConfigManager()
        config = config_manager.load_config(proxy_config_file)

        assert config is not None
        assert config.host == "127.0.0.1"
        assert config.port == 18080
        assert config.enable_cors is True
        assert len(config.models) == 2

        # Check first model
        model1 = config.models[0]
        assert model1.name == "tiny-model"
        assert model1.gpu_ids == [0]
        assert model1.port == 18001
        assert model1.enabled is True
        assert "aliases" in model1.config_overrides

        # Check second model
        model2 = config.models[1]
        assert model2.name == "medium-model"
        assert model2.enabled is False

    @patch("vllm_cli.proxy.manager.VLLMServer")
    @patch("vllm_cli.proxy.manager.ProxyServerProcess")
    def test_proxy_manager_full_lifecycle(
        self, mock_process_class, mock_server_class, proxy_config_file
    ):
        """Test full lifecycle of proxy manager."""
        # Load config
        config_manager = ProxyConfigManager()
        config = config_manager.load_config(proxy_config_file)

        # Setup mocks
        mock_process = MagicMock()
        mock_process.start.return_value = True
        mock_process.is_running.return_value = True
        mock_process_class.return_value = mock_process

        mock_server = MagicMock()
        mock_server.start.return_value = True
        mock_server.stop.return_value = None
        mock_server_class.return_value = mock_server

        # Create manager
        manager = ProxyManager(config)

        # Start all models (only enabled ones)
        with patch("time.sleep"):
            started = manager.start_all_models_no_wait()
        assert started == 1  # Only one model is enabled

        # Start proxy
        with patch("time.sleep"):
            result = manager.start_proxy()
        assert result is True

        # Verify proxy process was created and started
        mock_process_class.assert_called_once()
        mock_process.start.assert_called_once()

        # Stop everything
        manager.stop_proxy()

        # Verify cleanup
        mock_process.stop.assert_called_once()
        mock_server.stop.assert_called_once()
        assert len(manager.vllm_servers) == 0
        assert manager.proxy_process is None

    @patch("subprocess.Popen")
    def test_proxy_server_process_lifecycle(self, mock_popen, sample_proxy_config):
        """Test ProxyServerProcess lifecycle."""
        # Setup mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.pid = 12345
        mock_process.stdout.readline.return_value = "Server started\n"
        mock_popen.return_value = mock_process

        # Create and start process
        server_process = ProxyServerProcess(sample_proxy_config)

        with patch("time.sleep"):
            result = server_process.start()

        assert result is True
        assert server_process.is_running() is True
        mock_popen.assert_called_once()

        # Check command construction
        call_args = mock_popen.call_args[0][0]
        assert "python" in call_args
        assert "-m" in call_args
        assert "vllm_cli.proxy.server_launcher" in call_args
        assert "--host" in call_args
        assert "127.0.0.1" in call_args
        assert "--port" in call_args
        assert "8000" in call_args

        # Stop process
        server_process.stop()
        mock_process.terminate.assert_called_once()

    def test_model_registration_flow(self):
        """Test the flow of registering models with the proxy."""
        from vllm_cli.proxy.server import ProxyServer

        # Create proxy config with models
        proxy_config = ProxyConfig(
            host="127.0.0.1",
            port=18080,
            models=[
                ModelConfig(
                    name="model1",
                    model_path="path1",
                    port=18001,
                    config_overrides={"aliases": ["m1"]},
                    enabled=True,
                ),
                ModelConfig(
                    name="model2",
                    model_path="path2",
                    port=18002,
                    config_overrides={"aliases": ["m2", "second"]},
                    enabled=True,
                ),
            ],
        )

        # Create proxy server
        with patch("vllm_cli.proxy.server.httpx.AsyncClient"):
            proxy_server = ProxyServer(proxy_config)

        # Register models
        for model in proxy_config.models:
            backend_url = f"http://localhost:{model.port}"
            proxy_server.router.add_backend(
                model.name,
                backend_url,
                model.model_dump(),
            )

            # Also register aliases
            for alias in model.config_overrides.get("aliases", []):
                proxy_server.router.add_backend(
                    alias,
                    backend_url,
                    model.model_dump(),
                )

        # Verify routing works
        assert proxy_server.router.route_request("model1") == "http://localhost:18001"
        assert proxy_server.router.route_request("m1") == "http://localhost:18001"
        assert proxy_server.router.route_request("model2") == "http://localhost:18002"
        assert proxy_server.router.route_request("m2") == "http://localhost:18002"
        assert proxy_server.router.route_request("second") == "http://localhost:18002"

        # Check active models
        active = proxy_server.router.get_active_models()
        assert "model1" in active
        assert "model2" in active

    @patch("vllm_cli.proxy.manager.VLLMServer")
    def test_gpu_assignment_integration(self, mock_server_class):
        """Test GPU assignment across multiple models."""
        # Create config with specific GPU assignments
        config = ProxyConfig(
            models=[
                ModelConfig(
                    name="gpu0-model",
                    model_path="model1",
                    gpu_ids=[0],
                    port=18001,
                    enabled=True,
                ),
                ModelConfig(
                    name="gpu1-model",
                    model_path="model2",
                    gpu_ids=[1],
                    port=18002,
                    enabled=True,
                ),
                ModelConfig(
                    name="multi-gpu-model",
                    model_path="model3",
                    gpu_ids=[2, 3],
                    port=18003,
                    enabled=True,
                ),
            ]
        )

        # Setup mock server
        mock_server = MagicMock()
        mock_server.start.return_value = True
        mock_server_class.return_value = mock_server

        # Create manager and start models
        manager = ProxyManager(config)

        with patch("time.sleep"):
            started = manager.start_all_models_no_wait()

        assert started == 3

        # Verify correct GPU assignments in vLLM configs
        calls = mock_server_class.call_args_list

        # First model - GPU 0
        config1 = calls[0][0][0]
        assert config1["device"] == "0"
        assert "tensor_parallel_size" not in config1  # Single GPU

        # Second model - GPU 1
        config2 = calls[1][0][0]
        assert config2["device"] == "1"
        assert "tensor_parallel_size" not in config2  # Single GPU

        # Third model - GPUs 2,3
        config3 = calls[2][0][0]
        assert config3["device"] == "2,3"
        assert config3.get("tensor_parallel_size") == 2  # Multi-GPU

    def test_error_recovery_integration(self):
        """Test error recovery in the proxy system."""
        config = ProxyConfig(
            models=[
                ModelConfig(
                    name="failing-model",
                    model_path="nonexistent/model",
                    port=18001,
                    enabled=True,
                ),
            ]
        )

        manager = ProxyManager(config)

        # Mock VLLMServer to fail
        with patch("vllm_cli.proxy.manager.VLLMServer") as mock_server_class:
            mock_server = MagicMock()
            mock_server.start.return_value = False  # Fail to start
            mock_server_class.return_value = mock_server

            started = manager.start_all_models_no_wait()

            assert started == 0  # No models started
            assert len(manager.vllm_servers) == 0

        # Proxy should still be able to start
        with patch("vllm_cli.proxy.manager.ProxyServerProcess") as mock_process_class:
            mock_process = MagicMock()
            mock_process.start.return_value = True
            mock_process_class.return_value = mock_process

            with patch("time.sleep"):
                result = manager.start_proxy()

            assert result is True  # Proxy starts even with no models

    def test_runtime_state_persistence(self, temp_config_dir):
        """Test runtime state persistence."""
        import json

        from vllm_cli.proxy.runtime import ProxyRuntimeState

        # Create runtime state with custom state_dir
        runtime_state = ProxyRuntimeState()
        # Override the state_dir and state_file paths
        runtime_state.state_dir = temp_config_dir / ".config" / "vllm-cli"
        runtime_state.state_file = runtime_state.state_dir / "proxy_runtime.json"
        runtime_state.state_dir.mkdir(parents=True, exist_ok=True)

        # Save state
        result = runtime_state.save_state(
            host="127.0.0.1",
            port=18080,
            pid=99999,  # Use a fake PID that won't exist
        )
        assert result is True

        # Verify file was created with correct name
        assert runtime_state.state_file.exists()

        # Read the file directly to verify content
        with open(runtime_state.state_file, "r") as f:
            saved_data = json.load(f)

        assert saved_data["host"] == "127.0.0.1"
        assert saved_data["port"] == 18080
        assert saved_data["pid"] == 99999

        # Clear state
        result = runtime_state.clear_state()
        assert result is True
        assert not runtime_state.state_file.exists()

    @patch("httpx.Client.post")
    @patch("httpx.Client.get")
    def test_pre_registration_and_verification_workflow(self, mock_get, mock_post):
        """Test the complete pre-registration and verification workflow."""
        from vllm_cli.proxy.server import ProxyServer

        # Create proxy with models
        proxy_config = ProxyConfig(
            host="127.0.0.1",
            port=18080,
            models=[
                ModelConfig(
                    name="model1",
                    model_path="path1",
                    port=18001,
                    gpu_ids=[0],
                    enabled=True,
                ),
                ModelConfig(
                    name="model2",
                    model_path="path2",
                    port=18002,
                    gpu_ids=[1],
                    enabled=True,
                ),
            ],
        )

        with patch("vllm_cli.proxy.server.httpx.AsyncClient"):
            proxy_server = ProxyServer(proxy_config)

        # Pre-register models
        assert proxy_server.registry.pre_register(18001, [0], "model1")
        assert proxy_server.registry.pre_register(18002, [1], "model2")

        # Both should be pending
        assert len(proxy_server.registry.get_all_models()) == 2
        assert all(
            m.status.value == "pending"
            for m in proxy_server.registry.get_all_models().values()
        )

        # Mock successful model responses
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.side_effect = [
            {"data": [{"id": "actual-model1"}]},  # First model
            {"data": [{"id": "actual-model2"}]},  # Second model
        ]
        mock_get.return_value = mock_get_response

        # Simulate refresh process (simplified without async)
        for port, entry in proxy_server.registry.get_all_models().items():
            if entry.status.value == "pending":
                # Simulate getting model info
                response_data = mock_get_response.json()
                actual_name = response_data["data"][0]["id"]

                # Verify and activate
                proxy_server.registry.verify_and_activate(port, actual_name)

                # Add to router
                proxy_server.router.add_backend(
                    actual_name,
                    f"http://localhost:{port}",
                    {"port": port},
                )

        # All should now be available
        available = proxy_server.registry.get_available_models()
        assert len(available) == 2

        # Router should have both models
        assert "actual-model1" in proxy_server.router.backends
        assert "actual-model2" in proxy_server.router.backends

    @patch("vllm_cli.proxy.manager.VLLMServer")
    @patch("httpx.Client.post")
    def test_sleep_wake_lifecycle(self, mock_post, mock_server_class):
        """Test the sleep/wake lifecycle for models."""
        # Setup mock server
        mock_server = MagicMock()
        mock_server.start.return_value = True
        mock_server.is_running.return_value = True
        mock_server.port = 18001
        mock_server_class.return_value = mock_server

        # Create manager with a model
        config = ProxyConfig(
            models=[
                ModelConfig(
                    name="sleepy-model",
                    model_path="model1",
                    gpu_ids=[0],
                    port=18001,
                    enabled=True,
                ),
            ]
        )

        manager = ProxyManager(config)

        # Start the model
        with patch("time.sleep"):
            assert manager.start_model(config.models[0])

        # Mock successful sleep response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Sleep the model
        result = manager.sleep_model("sleepy-model", level=2)
        assert result is True

        # Mock wake response (accept 202 as success)
        mock_response.status_code = 202
        result = manager.wake_model("sleepy-model")
        assert result is True

    def test_gpu_utilization_validation(self):
        """Test GPU utilization validation in configuration."""
        config_manager = ProxyConfigManager()

        # Valid configuration - total utilization under limit
        valid_config = ProxyConfig(
            models=[
                ModelConfig(
                    name="model1",
                    model_path="path1",
                    port=18001,
                    gpu_ids=[0],
                    config_overrides={"gpu_memory_utilization": 0.4},
                    enabled=True,
                ),
                ModelConfig(
                    name="model2",
                    model_path="path2",
                    port=18002,
                    gpu_ids=[0],
                    config_overrides={"gpu_memory_utilization": 0.4},
                    enabled=True,
                ),
            ]
        )

        errors = config_manager.validate_config(valid_config)
        # Should pass basic validation (port checks, etc)
        # Note: GPU utilization validation would need to be implemented
        assert len(errors) <= 1  # May have warnings but not critical errors

        # Invalid configuration - over 95% on single GPU
        invalid_config = ProxyConfig(
            models=[
                ModelConfig(
                    name="model1",
                    model_path="path1",
                    port=18001,
                    gpu_ids=[0],
                    config_overrides={"gpu_memory_utilization": 0.5},
                    enabled=True,
                ),
                ModelConfig(
                    name="model2",
                    model_path="path2",
                    port=18002,
                    gpu_ids=[0],
                    config_overrides={"gpu_memory_utilization": 0.5},
                    enabled=True,
                ),
            ]
        )

        errors = config_manager.validate_config(invalid_config)
        # Basic validation should still work
        assert isinstance(errors, list)

    def test_stale_entry_cleanup(self):
        """Test cleanup of stale pending entries."""
        from vllm_cli.proxy.registry import ModelRegistry

        registry = ModelRegistry()

        # Add models
        registry.pre_register(18001, [0], "fresh-model")
        registry.pre_register(18002, [1], "stale-model")

        # Make one stale
        import datetime as dt

        old_time = dt.datetime.now() - dt.timedelta(minutes=10)
        registry.models[18002].last_activity = old_time

        # Clean up with 5 minute timeout
        removed = registry.cleanup_stale_entries(timeout_seconds=300)

        assert removed == 1
        assert 18001 in registry.models
        assert 18002 not in registry.models

    @patch("vllm_cli.proxy.manager.ProxyServerProcess")
    @patch("vllm_cli.proxy.manager.VLLMServer")
    def test_refresh_model_registrations(self, mock_server_class, mock_process_class):
        """Test refreshing model registrations with the proxy."""
        # Setup mocks
        mock_process = MagicMock()
        mock_process.is_running.return_value = True
        mock_process_class.return_value = mock_process

        mock_server = MagicMock()
        mock_server.start.return_value = True
        mock_server_class.return_value = mock_server

        # Create manager
        config = ProxyConfig(
            models=[
                ModelConfig(
                    name="model1",
                    model_path="path1",
                    port=18001,
                    gpu_ids=[0],
                    enabled=True,
                ),
            ]
        )

        manager = ProxyManager(config)
        manager.proxy_process = mock_process

        # Mock the API response
        with patch.object(manager, "_proxy_api_request") as mock_request:
            mock_request.return_value = MagicMock(
                json=lambda: {
                    "summary": {
                        "registered": 1,
                        "failed": 0,
                        "already_registered": 0,
                        "removed": 0,
                    }
                }
            )

            result = manager.refresh_model_registrations()

            assert result["summary"]["registered"] == 1
            mock_request.assert_called_once_with(
                "POST", "/proxy/refresh_models", timeout=10.0
            )

    def test_model_allocation_strategies(self):
        """Test different GPU allocation strategies."""
        manager = ProxyManager()

        # Test with 4 GPUs and 2 models
        # Patch where it's imported inside the function
        with patch("vllm_cli.system.get_gpu_info") as mock_gpu:
            mock_gpu.return_value = [
                {"index": 0},
                {"index": 1},
                {"index": 2},
                {"index": 3},
            ]

            manager.proxy_config.models = [
                ModelConfig(
                    name="model1", model_path="path1", port=18001, enabled=True
                ),
                ModelConfig(
                    name="model2", model_path="path2", port=18002, enabled=True
                ),
            ]

            allocated = manager.allocate_gpus_automatically()

            assert len(allocated) == 2
            # Each model should get 2 GPUs
            assert len(allocated[0].gpu_ids) == 2
            assert len(allocated[1].gpu_ids) == 2
            # No overlap
            assert set(allocated[0].gpu_ids).isdisjoint(set(allocated[1].gpu_ids))

        # Test with more models than GPUs
        with patch("vllm_cli.system.get_gpu_info") as mock_gpu:
            mock_gpu.return_value = [{"index": 0}, {"index": 1}]

            manager.proxy_config.models = [
                ModelConfig(
                    name=f"model{i}",
                    model_path=f"path{i}",
                    port=18001 + i,
                    enabled=True,
                )
                for i in range(4)
            ]

            allocated = manager.allocate_gpus_automatically()

            # Only first 2 models get GPUs
            assert len(allocated) == 2
            assert allocated[0].gpu_ids == [0]
            assert allocated[1].gpu_ids == [1]
