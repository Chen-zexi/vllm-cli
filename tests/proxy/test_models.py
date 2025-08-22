#!/usr/bin/env python3
"""
Unit tests for proxy data models.
"""
import pytest
from pydantic import ValidationError

from vllm_cli.proxy.models import ModelConfig, ModelStatus, ProxyConfig, ProxyStatus


class TestModelConfig:
    """Test ModelConfig data model."""

    def test_model_config_creation(self):
        """Test creating a valid ModelConfig."""
        config = ModelConfig(
            name="test-model",
            model_path="/path/to/model",
            gpu_ids=[0, 1],
            port=8001,
            profile="standard",
            config_overrides={"max_tokens": 2048},
            enabled=True,
        )

        assert config.name == "test-model"
        assert config.model_path == "/path/to/model"
        assert config.gpu_ids == [0, 1]
        assert config.port == 8001
        assert config.profile == "standard"
        assert config.config_overrides == {"max_tokens": 2048}
        assert config.enabled is True

    def test_model_config_defaults(self):
        """Test ModelConfig with default values."""
        config = ModelConfig(
            name="minimal",
            model_path="path/to/model",
            port=8001,
        )

        assert config.gpu_ids == []
        assert config.profile is None
        assert config.config_overrides == {}
        assert config.enabled is True

    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        # Missing required fields
        with pytest.raises(ValidationError):
            ModelConfig()

        with pytest.raises(ValidationError):
            ModelConfig(name="test")  # Missing model_path and port

    def test_model_config_with_aliases(self):
        """Test ModelConfig with aliases in config_overrides."""
        config = ModelConfig(
            name="gpt-model",
            model_path="openai/gpt",
            port=8001,
            config_overrides={"aliases": ["gpt", "chatgpt"]},
        )

        assert config.config_overrides["aliases"] == ["gpt", "chatgpt"]

    def test_model_config_serialization(self):
        """Test ModelConfig serialization."""
        config = ModelConfig(
            name="test",
            model_path="path",
            port=8001,
            gpu_ids=[0],
            enabled=False,
        )

        data = config.model_dump()
        assert data["name"] == "test"
        assert data["model_path"] == "path"
        assert data["port"] == 8001
        assert data["gpu_ids"] == [0]
        assert data["enabled"] is False

        # Test JSON serialization
        json_str = config.model_dump_json()
        assert "test" in json_str
        assert "8001" in json_str


class TestProxyConfig:
    """Test ProxyConfig data model."""

    def test_proxy_config_creation(self):
        """Test creating a valid ProxyConfig."""
        model1 = ModelConfig(
            name="model1",
            model_path="path1",
            port=8001,
        )
        model2 = ModelConfig(
            name="model2",
            model_path="path2",
            port=8002,
        )

        config = ProxyConfig(
            host="0.0.0.0",
            port=8000,
            models=[model1, model2],
            enable_cors=True,
            enable_metrics=False,
            log_requests=True,
        )

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert len(config.models) == 2
        assert config.models[0].name == "model1"
        assert config.enable_cors is True
        assert config.enable_metrics is False
        assert config.log_requests is True

    def test_proxy_config_defaults(self):
        """Test ProxyConfig with default values."""
        config = ProxyConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.models == []
        assert config.enable_cors is True
        assert config.enable_metrics is True
        assert config.log_requests is False

    def test_proxy_config_with_empty_models(self):
        """Test ProxyConfig with no models."""
        config = ProxyConfig(
            host="localhost",
            port=9000,
            models=[],
        )

        assert config.host == "localhost"
        assert config.port == 9000
        assert len(config.models) == 0

    def test_proxy_config_serialization(self):
        """Test ProxyConfig serialization."""
        model = ModelConfig(
            name="test",
            model_path="path",
            port=8001,
        )
        config = ProxyConfig(
            host="127.0.0.1",
            port=8080,
            models=[model],
        )

        data = config.model_dump()
        assert data["host"] == "127.0.0.1"
        assert data["port"] == 8080
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "test"


class TestModelStatus:
    """Test ModelStatus data model."""

    def test_model_status_creation(self):
        """Test creating a ModelStatus."""
        status = ModelStatus(
            name="test-model",
            model_path="/path/to/model",
            port=8001,
            gpu_ids=[0, 1],
            status="running",
            uptime=3600.0,
            request_count=100,
            last_request_time="2024-01-01T12:00:00",
        )

        assert status.name == "test-model"
        assert status.model_path == "/path/to/model"
        assert status.port == 8001
        assert status.gpu_ids == [0, 1]
        assert status.status == "running"
        assert status.uptime == 3600.0
        assert status.request_count == 100
        assert status.last_request_time == "2024-01-01T12:00:00"

    def test_model_status_with_error(self):
        """Test ModelStatus with error state."""
        status = ModelStatus(
            name="failed-model",
            model_path="/path",
            port=8001,
            gpu_ids=[],
            status="error",
            error_message="Failed to allocate GPU memory",
        )

        assert status.status == "error"
        assert status.error_message == "Failed to allocate GPU memory"
        assert status.uptime is None
        assert status.request_count == 0

    def test_model_status_defaults(self):
        """Test ModelStatus default values."""
        status = ModelStatus(
            name="test",
            model_path="path",
            port=8001,
            gpu_ids=[],
            status="stopped",
        )

        assert status.uptime is None
        assert status.error_message is None
        assert status.request_count == 0
        assert status.last_request_time is None


class TestProxyStatus:
    """Test ProxyStatus data model."""

    def test_proxy_status_creation(self):
        """Test creating a ProxyStatus."""
        model_status = ModelStatus(
            name="model1",
            model_path="path1",
            port=8001,
            gpu_ids=[0],
            status="running",
        )

        proxy_status = ProxyStatus(
            proxy_running=True,
            proxy_port=8000,
            proxy_host="0.0.0.0",
            models=[model_status],
            total_requests=500,
            start_time="2024-01-01T10:00:00",
        )

        assert proxy_status.proxy_running is True
        assert proxy_status.proxy_port == 8000
        assert proxy_status.proxy_host == "0.0.0.0"
        assert len(proxy_status.models) == 1
        assert proxy_status.models[0].name == "model1"
        assert proxy_status.total_requests == 500
        assert proxy_status.start_time == "2024-01-01T10:00:00"

    def test_proxy_status_stopped(self):
        """Test ProxyStatus when proxy is stopped."""
        proxy_status = ProxyStatus(
            proxy_running=False,
            proxy_port=8000,
            proxy_host="localhost",
            models=[],
        )

        assert proxy_status.proxy_running is False
        assert len(proxy_status.models) == 0
        assert proxy_status.total_requests == 0
        assert proxy_status.start_time is None

    def test_proxy_status_serialization(self):
        """Test ProxyStatus serialization to dict."""
        model_status = ModelStatus(
            name="test",
            model_path="path",
            port=8001,
            gpu_ids=[],
            status="running",
            request_count=10,
        )

        proxy_status = ProxyStatus(
            proxy_running=True,
            proxy_port=8000,
            proxy_host="0.0.0.0",
            models=[model_status],
            total_requests=100,
        )

        data = proxy_status.model_dump()
        assert data["proxy_running"] is True
        assert data["proxy_port"] == 8000
        assert data["total_requests"] == 100
        assert len(data["models"]) == 1
        assert data["models"][0]["request_count"] == 10
