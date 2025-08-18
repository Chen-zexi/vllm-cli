#!/usr/bin/env python3
"""
Model Management Package

This package provides comprehensive model management for vLLM CLI including
discovery, caching, metadata extraction, and requirement analysis.

Main Components:
- Manager: High-level model management with caching
- Discovery: Model detection from various sources (hf-model-tool, directories)
- Cache: Performance optimization through intelligent caching
- Metadata: Configuration and requirement extraction from model files

The package supports both hf-model-tool integration and fallback directory
scanning for maximum compatibility.
"""

# Cache utilities
from .cache import (
    ModelCache,
    cache_models,
    clear_global_cache,
    get_cached_models,
    get_global_cache,
    get_global_cache_stats,
)

# Discovery functions
from .discovery import (
    build_model_dict,
    find_model_by_name,
    scan_for_lora_adapters,
    scan_for_models,
    validate_model_path,
)

# Main model management functions
from .manager import (
    ModelManager,
    get_model_details,
    get_model_manager,
    list_available_models,
    search_models,
)

# Metadata extraction
from .metadata import (
    analyze_model_compatibility,
    extract_model_config,
    get_model_requirements,
)

__all__ = [
    # Main manager
    "ModelManager",
    "get_model_manager",
    "list_available_models",
    "get_model_details",
    "search_models",
    # Discovery
    "scan_for_models",
    "scan_for_lora_adapters",
    "build_model_dict",
    "validate_model_path",
    "find_model_by_name",
    # Cache
    "ModelCache",
    "get_global_cache",
    "cache_models",
    "get_cached_models",
    "clear_global_cache",
    "get_global_cache_stats",
    # Metadata
    "extract_model_config",
    "get_model_requirements",
    "analyze_model_compatibility",
]
