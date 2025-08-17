"""Shared fixtures and test configuration for vLLM CLI tests."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml


@pytest.fixture
def temp_config_dir():
    """Create a temporary config directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config_file(temp_config_dir):
    """Create a mock config file."""
    config_file = temp_config_dir / "config.yaml"
    config_data = {
        "last_model": "test-model",
        "last_config": {"model": "test-model", "port": 8000, "tensor_parallel_size": 1},
        "model_directories": ["/home/test/.cache/huggingface/hub"],
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    return config_file


@pytest.fixture
def mock_profiles():
    """Mock profile data."""
    return {
        "standard": {"description": "Standard configuration"},
        "high_throughput": {
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.95,
            "enable_chunked_prefill": True,
        },
        "low_memory": {
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.70,
            "quantization": "bitsandbytes",
        },
    }


@pytest.fixture
def mock_user_profiles_file(temp_config_dir, mock_profiles):
    """Create a mock user profiles file."""
    profiles_file = temp_config_dir / "user_profiles.json"
    with open(profiles_file, "w") as f:
        json.dump(mock_profiles, f)
    return profiles_file


@pytest.fixture
def mock_models():
    """Mock model data for testing."""
    return [
        {
            "name": "meta-llama/Llama-2-7b-hf",
            "path": "/models/llama-2-7b",
            "size": 13_000_000_000,
            "type": "model",
            "publisher": "meta-llama",
            "display_name": "Llama-2-7b-hf",
        },
        {
            "name": "openai/gpt-3.5-turbo",
            "path": "/models/gpt-3.5",
            "size": 7_000_000_000,
            "type": "model",
            "publisher": "openai",
            "display_name": "gpt-3.5-turbo",
        },
        {
            "name": "custom-model",
            "path": "/models/custom",
            "size": 5_000_000_000,
            "type": "custom_model",
            "publisher": "local",
            "display_name": "custom-model",
        },
    ]


@pytest.fixture
def mock_lora_adapters():
    """Mock LoRA adapter data."""
    return [
        {
            "name": "lora_adapter_1",
            "path": "/adapters/lora1",
            "size": 100_000_000,
            "type": "lora_adapter",
            "rank": 32,
        },
        {
            "name": "lora_adapter_2",
            "path": "/adapters/lora2",
            "size": 150_000_000,
            "type": "lora_adapter",
            "rank": 64,
        },
    ]


@pytest.fixture
def mock_argument_schema():
    """Mock vLLM argument schema."""
    return {
        "model": {
            "type": "string",
            "description": "Model name or path",
            "required": True,
        },
        "port": {
            "type": "integer",
            "description": "Port to serve on",
            "default": 8000,
            "minimum": 1,
            "maximum": 65535,
        },
        "tensor_parallel_size": {
            "type": "integer",
            "description": "Number of GPUs for tensor parallelism",
            "default": 1,
            "minimum": 1,
        },
        "gpu_memory_utilization": {
            "type": "number",
            "description": "GPU memory utilization",
            "default": 0.9,
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "max_model_len": {
            "type": "integer",
            "description": "Maximum model context length",
            "minimum": 1,
        },
        "quantization": {
            "type": "string",
            "description": "Quantization method",
            "enum": ["awq", "bitsandbytes", "gptq", "squeezellm", None],
        },
        "trust_remote_code": {
            "type": "boolean",
            "description": "Trust remote code",
            "default": False,
        },
    }


@pytest.fixture
def mock_gpu_info():
    """Mock GPU information."""
    return {
        "available": True,
        "cuda_version": "12.1",
        "driver_version": "535.129.03",
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 4090",
                "memory_total": 24576,
                "memory_used": 2048,
                "memory_free": 22528,
                "utilization": 15,
            }
        ],
    }


@pytest.fixture
def mock_system_info():
    """Mock system information."""
    return {
        "platform": "Linux",
        "python_version": "3.11.5",
        "vllm_version": "0.5.0",
        "torch_version": "2.3.0+cu121",
        "total_memory": 64_000_000_000,
        "available_memory": 48_000_000_000,
    }


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for server process testing."""
    mock_proc = Mock()
    mock_proc.poll.return_value = None  # Process is running
    mock_proc.pid = 12345
    mock_proc.returncode = None
    return mock_proc


@pytest.fixture
def mock_server_config():
    """Mock server configuration."""
    return {
        "model": "meta-llama/Llama-2-7b-hf",
        "port": 8000,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": False,
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # Import and reset any singletons if needed
    yield
