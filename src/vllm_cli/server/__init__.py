#!/usr/bin/env python3
"""
vLLM Server Management Package

This package provides comprehensive vLLM server management including
process lifecycle, monitoring, discovery, and utilities.

Main Components:
- VLLMServer: Core server management class
- Process management: Server registry and lifecycle
- Discovery: External server detection
- Monitoring: Health checks and metrics
- Utils: Port management and cleanup utilities
"""

# Discovery functions
from .discovery import detect_external_servers, detect_running_vllm_servers

# Core server class
from .manager import VLLMServer

# Monitoring functions
from .monitoring import get_server_metrics, monitor_all_servers, perform_health_check

# Process management functions
from .process import (
    add_server,
    cleanup_servers_on_exit,
    find_server_by_model,
    find_server_by_port,
    get_active_servers,
    remove_server,
    stop_all_servers,
)

# Utility functions
from .utils import (
    cleanup_old_logs,
    get_next_available_port,
    is_port_available,
    validate_port_range,
)

__all__ = [
    # Core classes
    "VLLMServer",
    # Process management
    "get_active_servers",
    "add_server",
    "remove_server",
    "stop_all_servers",
    "find_server_by_port",
    "find_server_by_model",
    "cleanup_servers_on_exit",
    # Discovery
    "detect_running_vllm_servers",
    "detect_external_servers",
    # Monitoring
    "perform_health_check",
    "get_server_metrics",
    "monitor_all_servers",
    # Utils
    "get_next_available_port",
    "cleanup_old_logs",
    "is_port_available",
    "validate_port_range",
]
