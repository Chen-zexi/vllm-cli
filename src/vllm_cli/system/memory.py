#!/usr/bin/env python3
"""
System memory information utilities.

Provides system memory information gathering with caching support
to avoid excessive system calls.
"""
import logging
import time
from typing import Any, Dict, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)

# Cache for memory information with timestamp
_memory_info_cache: Optional[Tuple[float, Dict[str, Any]]] = None
# Cache TTL in seconds
MEMORY_CACHE_TTL = 1.0


def get_memory_info(use_cache: bool = True) -> Dict[str, Any]:
    """
    Get system memory information with caching.

    Memory information is cached briefly to avoid excessive psutil calls
    when multiple UI components need memory data.

    Args:
        use_cache: Whether to use cached data if available

    Returns:
        Dictionary with memory information containing:
        - total: Total system memory in bytes
        - available: Available memory in bytes
        - used: Used memory in bytes
        - free: Free memory in bytes
        - percent: Memory usage percentage
    """
    global _memory_info_cache

    # Check cache first
    if use_cache and _memory_info_cache is not None:
        cache_time, cached_data = _memory_info_cache
        if time.time() - cache_time < MEMORY_CACHE_TTL:
            return cached_data

    # Fetch fresh data
    memory_data = _fetch_memory_info()

    # Update cache
    _memory_info_cache = (time.time(), memory_data)

    return memory_data


def _fetch_memory_info() -> Dict[str, Any]:
    """
    Fetch current system memory information.

    Returns:
        Dictionary with memory information
    """
    try:
        mem = psutil.virtual_memory()

        return {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "free": mem.free,
            "percent": mem.percent,
        }
    except Exception as e:
        logger.warning(f"Failed to get memory info: {e}")
        # Return fallback values
        return {"total": 0, "available": 0, "used": 0, "free": 0, "percent": 0.0}
