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

# Core server class
from .manager import VLLMServer

# Process management functions
from .process import (
    get_active_servers,
    add_server,
    remove_server,
    stop_all_servers,
    find_server_by_port,
    find_server_by_model,
    cleanup_servers_on_exit,
)

# Discovery functions
from .discovery import detect_running_vllm_servers, detect_external_servers

# Monitoring functions
from .monitoring import perform_health_check, get_server_metrics, monitor_all_servers

# Utility functions
from .utils import (
    get_next_available_port,
    cleanup_old_logs,
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
