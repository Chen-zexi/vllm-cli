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
            started = manager.start_all_models()
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
                model.dict(),
            )

            # Also register aliases
            for alias in model.config_overrides.get("aliases", []):
                proxy_server.router.add_backend(
                    alias,
                    backend_url,
                    model.dict(),
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
            started = manager.start_all_models()

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

            started = manager.start_all_models()

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
