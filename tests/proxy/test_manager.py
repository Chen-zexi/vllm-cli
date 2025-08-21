#!/usr/bin/env python3
"""
Unit tests for proxy manager.
"""
from unittest.mock import MagicMock, patch

import pytest

from vllm_cli.proxy.manager import ProxyManager
from vllm_cli.proxy.models import ModelConfig


class TestProxyManager:
    """Test ProxyManager functionality."""

    @pytest.fixture
    def manager(self, sample_proxy_config, mock_config_manager):
        """Create a ProxyManager instance with mocked dependencies."""
        with patch(
            "vllm_cli.proxy.manager.ConfigManager", return_value=mock_config_manager
        ):
            manager = ProxyManager(sample_proxy_config)
            yield manager

    @pytest.fixture
    def model_config(self):
        """Create a sample model configuration."""
        return ModelConfig(
            name="test-model",
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            gpu_ids=[0],
            port=8001,
            profile="standard",
            config_overrides={"aliases": ["tiny"]},
            enabled=True,
        )

    def test_proxy_manager_init(self, sample_proxy_config):
        """Test ProxyManager initialization."""
        with patch("vllm_cli.proxy.manager.ConfigManager"):
            manager = ProxyManager(sample_proxy_config)

            assert manager.proxy_config == sample_proxy_config
            assert manager.proxy_process is None
            assert manager.vllm_servers == {}
            assert manager.config_manager is not None

    def test_proxy_manager_init_default_config(self):
        """Test ProxyManager with default configuration."""
        with patch("vllm_cli.proxy.manager.ConfigManager"):
            manager = ProxyManager()

            assert manager.proxy_config is not None
            assert manager.proxy_config.host == "0.0.0.0"
            assert manager.proxy_config.port == 8000

    @patch("vllm_cli.proxy.manager.ProxyServerProcess")
    def test_start_proxy_success(self, mock_process_class, manager):
        """Test successful proxy server start."""
        mock_process = MagicMock()
        mock_process.start.return_value = True
        mock_process_class.return_value = mock_process

        with patch("time.sleep"):
            result = manager.start_proxy()

        assert result is True
        assert manager.proxy_process == mock_process
        mock_process.start.assert_called_once()

    @patch("vllm_cli.proxy.manager.ProxyServerProcess")
    def test_start_proxy_failure(self, mock_process_class, manager):
        """Test proxy server start failure."""
        mock_process = MagicMock()
        mock_process.start.return_value = False
        mock_process_class.return_value = mock_process

        result = manager.start_proxy()

        assert result is False
        mock_process.start.assert_called_once()

    @patch("vllm_cli.proxy.manager.ProxyServerProcess")
    def test_start_proxy_exception(self, mock_process_class, manager):
        """Test proxy server start with exception."""
        mock_process_class.side_effect = Exception("Test error")

        result = manager.start_proxy()

        assert result is False

    def test_stop_proxy(self, manager):
        """Test stopping proxy server."""
        # Setup mock proxy process
        mock_process = MagicMock()
        manager.proxy_process = mock_process

        # Add mock vLLM servers
        mock_server1 = MagicMock()
        mock_server2 = MagicMock()
        manager.vllm_servers = {
            "model1": mock_server1,
            "model2": mock_server2,
        }

        # Stop proxy
        manager.stop_proxy()

        # Verify all servers stopped
        mock_server1.stop.assert_called_once()
        mock_server2.stop.assert_called_once()
        mock_process.stop.assert_called_once()
        assert manager.proxy_process is None
        assert len(manager.vllm_servers) == 0

    @patch("vllm_cli.proxy.manager.VLLMServer")
    def test_start_model_success(self, mock_server_class, manager, model_config):
        """Test successful model server start."""
        mock_server = MagicMock()
        mock_server.start.return_value = True
        mock_server_class.return_value = mock_server

        result = manager.start_model(model_config)

        assert result is True
        assert "test-model" in manager.vllm_servers
        assert manager.vllm_servers["test-model"] == mock_server
        mock_server.start.assert_called_once()

    @patch("vllm_cli.proxy.manager.VLLMServer")
    def test_start_model_already_running(
        self, mock_server_class, manager, model_config
    ):
        """Test starting a model that's already running."""
        # Add model to running servers
        manager.vllm_servers[model_config.name] = MagicMock()

        result = manager.start_model(model_config)

        assert result is False
        mock_server_class.assert_not_called()

    @patch("vllm_cli.proxy.manager.VLLMServer")
    def test_start_model_failure(self, mock_server_class, manager, model_config):
        """Test model server start failure."""
        mock_server = MagicMock()
        mock_server.start.return_value = False
        mock_server_class.return_value = mock_server

        result = manager.start_model(model_config)

        assert result is False
        assert model_config.name not in manager.vllm_servers

    def test_stop_model_success(self, manager):
        """Test successful model server stop."""
        mock_server = MagicMock()
        manager.vllm_servers["test-model"] = mock_server

        result = manager.stop_model("test-model")

        assert result is True
        assert "test-model" not in manager.vllm_servers
        mock_server.stop.assert_called_once()

    def test_stop_model_not_running(self, manager):
        """Test stopping a model that's not running."""
        result = manager.stop_model("nonexistent-model")

        assert result is False

    @patch("vllm_cli.proxy.manager.VLLMServer")
    @patch("time.sleep")
    def test_start_all_models(self, mock_sleep, mock_server_class, manager):
        """Test starting all enabled models."""
        # Create mock servers
        mock_server = MagicMock()
        mock_server.start.return_value = True
        mock_server_class.return_value = mock_server

        # Proxy config has 2 enabled models
        started = manager.start_all_models()

        assert started == 2
        assert mock_server_class.call_count == 2
        assert len(manager.vllm_servers) == 2

    @patch("vllm_cli.proxy.manager.VLLMServer")
    def test_start_all_models_some_disabled(self, mock_server_class, manager):
        """Test starting models with some disabled."""
        # Disable one model
        manager.proxy_config.models[1].enabled = False

        mock_server = MagicMock()
        mock_server.start.return_value = True
        mock_server_class.return_value = mock_server

        with patch("time.sleep"):
            started = manager.start_all_models()

        assert started == 1
        assert mock_server_class.call_count == 1

    def test_build_vllm_config_basic(self, manager, model_config):
        """Test building basic vLLM configuration."""
        config = manager._build_vllm_config(model_config)

        assert config["model"] == model_config.model_path
        assert config["port"] == model_config.port
        assert "device" in config
        assert config["device"] == "0"

    def test_build_vllm_config_with_profile(self, manager, model_config):
        """Test building vLLM config with profile."""
        config = manager._build_vllm_config(model_config)

        # Should include profile settings
        assert "max_model_len" in config
        assert config["max_model_len"] == 4096
        assert "gpu_memory_utilization" in config
        assert config["gpu_memory_utilization"] == 0.9

    def test_build_vllm_config_multi_gpu(self, manager):
        """Test building vLLM config for multi-GPU model."""
        model_config = ModelConfig(
            name="large-model",
            model_path="meta-llama/Llama-2-13b",
            gpu_ids=[0, 1, 2, 3],
            port=8002,
            profile="standard",
        )

        config = manager._build_vllm_config(model_config)

        assert config["device"] == "0,1,2,3"
        assert config.get("tensor_parallel_size") == 4

    def test_build_vllm_config_single_gpu_override(self, manager):
        """Test single GPU overriding parallel settings."""
        model_config = ModelConfig(
            name="small-model",
            model_path="gpt2",
            gpu_ids=[0],
            port=8001,
            profile="standard",
            config_overrides={},  # No overrides
        )

        # Add tensor_parallel_size to mock profile
        manager.config_manager.get_profile.return_value = {
            "name": "standard",
            "config": {
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 2,
            },
        }

        config = manager._build_vllm_config(model_config)

        # Should remove parallel settings for single GPU from profile
        assert "tensor_parallel_size" not in config
        assert "pipeline_parallel_size" not in config

    def test_build_vllm_config_with_overrides(self, manager):
        """Test building vLLM config with config overrides."""
        model_config = ModelConfig(
            name="custom-model",
            model_path="model/path",
            gpu_ids=[0],
            port=8001,
            profile="standard",
            config_overrides={
                "max_tokens": 2048,
                "temperature": 0.7,
                "aliases": ["custom", "test"],  # Will be included in vLLM config
            },
        )

        config = manager._build_vllm_config(model_config)

        assert config["max_tokens"] == 2048
        assert config["temperature"] == 0.7
        assert config["aliases"] == [
            "custom",
            "test",
        ]  # Aliases are included with other overrides

    def test_build_vllm_config_no_profile(self, manager):
        """Test building vLLM config without profile."""
        model_config = ModelConfig(
            name="no-profile-model",
            model_path="model/path",
            gpu_ids=[0, 1],
            port=8001,
            profile=None,
        )

        config = manager._build_vllm_config(model_config)

        assert config["model"] == "model/path"
        assert config["port"] == 8001
        assert config["device"] == "0,1"
        assert config.get("tensor_parallel_size") == 2
