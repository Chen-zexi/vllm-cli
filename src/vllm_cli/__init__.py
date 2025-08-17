#!/usr/bin/env python3
"""
vLLM CLI Tool: A convenient CLI for serving LLMs with vLLM
"""

# Get version from package metadata (pyproject.toml)
# Python 3.11+ has importlib.metadata built-in
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vllm-cli")
except PackageNotFoundError:
    # Fallback for development/editable installs
    __version__ = "0.1.1-dev"

__author__ = "vLLM CLI Contributors"

from .config import ConfigManager
from .models import get_model_details, list_available_models
from .server import VLLMServer

__all__ = [
    "__version__",
    "__author__",
    "list_available_models",
    "get_model_details",
    "VLLMServer",
    "ConfigManager",
]
