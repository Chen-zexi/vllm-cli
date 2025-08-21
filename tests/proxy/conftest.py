#!/usr/bin/env python3
"""
Test fixtures and utilities for proxy tests.
"""
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import pytest
import yaml

from vllm_cli.proxy.models import ModelConfig, ProxyConfig


@pytest.fixture
def sample_model_config() -> ModelConfig:
    """Create a sample model configuration."""
    return ModelConfig(
        name="test-model",
        model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        gpu_ids=[0],
        port=8001,
        profile="standard",
        config_overrides={"aliases": ["tiny", "test"]},
        enabled=True,
    )


@pytest.fixture
def sample_proxy_config() -> ProxyConfig:
    """Create a sample proxy configuration."""
    return ProxyConfig(
        host="127.0.0.1",
        port=8000,
        models=[
            ModelConfig(
                name="model1",
                model_path="path/to/model1",
                gpu_ids=[0],
                port=8001,
                enabled=True,
            ),
            ModelConfig(
                name="model2",
                model_path="path/to/model2",
                gpu_ids=[1],
                port=8002,
                enabled=True,
            ),
        ],
        enable_cors=True,
        enable_metrics=True,
        log_requests=False,
    )


@pytest.fixture
def temp_config_file(sample_proxy_config) -> Generator[Path, None, None]:
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_dict = sample_proxy_config.dict()
        yaml.dump({"proxy": config_dict["models"][0]}, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_vllm_server():
    """Mock VLLMServer for testing."""
    mock_server = MagicMock()
    mock_server.start.return_value = True
    mock_server.stop.return_value = None
    mock_server.is_running.return_value = True
    mock_server.port = 8001
    mock_server.model = "test-model"
    return mock_server


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for testing process management."""
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.pid = 12345
        mock_process.stdout.readline.return_value = "Server started\n"
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        yield mock_popen


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for API testing."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_response.iter_bytes.return_value = iter([b"data: test\n\n"])
        mock_client.post.return_value = mock_response
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_config_manager():
    """Mock ConfigManager for testing."""
    mock_cm = MagicMock()
    mock_cm.get_profile.return_value = {
        "name": "standard",
        "config": {
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
        },
    }
    return mock_cm


@pytest.fixture
def sample_request_body() -> Dict[str, Any]:
    """Sample request body for OpenAI API."""
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "temperature": 0.7,
        "max_tokens": 100,
    }


@pytest.fixture
def sample_streaming_request() -> Dict[str, Any]:
    """Sample streaming request body."""
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True,
        "temperature": 0.8,
    }


@pytest.fixture
def mock_fastapi_request():
    """Mock FastAPI Request object."""
    mock_request = MagicMock()
    mock_request.url.path = "/v1/chat/completions"
    mock_request.method = "POST"
    mock_request.headers = {"content-type": "application/json"}

    async def mock_json():
        return {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }

    async def mock_body():
        return b'{"model": "test-model"}'

    mock_request.json = mock_json
    mock_request.body = mock_body
    return mock_request


@pytest.fixture
def mock_runtime_state(tmp_path):
    """Mock runtime state with temporary directory."""
    runtime_dir = tmp_path / ".vllm-cli" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    with patch("vllm_cli.proxy.runtime.ProxyRuntimeState") as mock_state:
        instance = MagicMock()
        instance.runtime_dir = runtime_dir
        instance.save_state.return_value = None
        instance.load_state.return_value = {
            "host": "127.0.0.1",
            "port": 8000,
            "pid": 12345,
        }
        instance.clear_state.return_value = None
        mock_state.return_value = instance
        yield instance


def create_mock_backend_response(
    model_name: str = "test-model",
    content: str = "Test response",
    stream: bool = False,
) -> MagicMock:
    """Create a mock backend response."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    if stream:
        # Streaming response
        mock_response.iter_bytes.return_value = iter(
            [
                b'data: {"choices": [{"delta": {"content": "Test"}}]}\n\n',
                b'data: {"choices": [{"delta": {"content": " response"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
    else:
        # Regular response
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    return mock_response


def create_test_config_yaml(
    path: Path,
    num_models: int = 2,
    enable_cors: bool = True,
) -> None:
    """Create a test configuration YAML file."""
    config = {
        "proxy": {
            "host": "0.0.0.0",
            "port": 8080,
            "enable_cors": enable_cors,
            "enable_metrics": True,
            "log_requests": False,
        },
        "models": [],
    }

    for i in range(num_models):
        config["models"].append(
            {
                "name": f"model{i+1}",
                "model_path": f"test/model{i+1}",
                "gpu_ids": [i],
                "port": 8001 + i,
                "profile": "standard",
                "config_overrides": {
                    "aliases": [f"m{i+1}", f"test{i+1}"],
                },
                "enabled": True,
            }
        )

    with open(path, "w") as f:
        yaml.dump(config, f)
