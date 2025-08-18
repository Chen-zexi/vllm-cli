#!/usr/bin/env python3
"""
System Information Package

This package provides comprehensive system information gathering including
GPU detection, memory monitoring, requirements checking, and environment utilities.

Main Components:
- GPU: GPU information and CUDA detection
- Memory: System memory monitoring with caching
- Requirements: System requirements validation for vLLM
- Formatting: Human-readable formatting utilities
- Environment: Environment and package manager integration
"""

from .capabilities import (
    get_attention_backend_capabilities,
    get_gpu_capabilities,
    get_performance_recommendations,
    get_quantization_capabilities,
)

# Dependencies and capabilities
from .dependencies import (
    check_optimization_recommendations,
    get_attention_backend_info,
    get_core_dependencies,
    get_dependency_info,
    get_environment_info,
    get_quantization_info,
)

# Environment utilities
from .environment import get_conda_envs

# Formatting utilities
from .formatting import format_size

# GPU utilities
from .gpu import get_cuda_version, get_gpu_info

# Memory utilities
from .memory import MEMORY_CACHE_TTL, get_memory_info

# Requirements checking
from .requirements import check_system_requirements, check_vllm_installation

__all__ = [
    # GPU
    "get_gpu_info",
    "get_cuda_version",
    # Memory
    "get_memory_info",
    "MEMORY_CACHE_TTL",
    # Requirements
    "check_system_requirements",
    "check_vllm_installation",
    # Formatting
    "format_size",
    # Environment
    "get_conda_envs",
    # Dependencies
    "get_dependency_info",
    "get_attention_backend_info",
    "get_quantization_info",
    "get_core_dependencies",
    "get_environment_info",
    "check_optimization_recommendations",
    # Capabilities
    "get_gpu_capabilities",
    "get_attention_backend_capabilities",
    "get_quantization_capabilities",
    "get_performance_recommendations",
]
