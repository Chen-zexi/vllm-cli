#!/usr/bin/env python3
"""
vLLM CLI Tool: A convenient CLI for serving LLMs with vLLM
"""
import sys

# Get version from package metadata (pyproject.toml)
if sys.version_info >= (3, 8):
    from importlib.metadata import version, PackageNotFoundError
else:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("vllm-cli")
except PackageNotFoundError:
    # Fallback for development/editable installs
    __version__ = "0.1.1-dev"

__author__ = "vLLM CLI Contributors"

from .models import list_available_models, get_model_details
from .server import VLLMServer
from .config import ConfigManager

__all__ = [
    "__version__",
    "__author__",
    "list_available_models",
    "get_model_details",
    "VLLMServer",
    "ConfigManager",
]
