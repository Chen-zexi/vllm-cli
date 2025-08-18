#!/usr/bin/env python3
"""
Server utilities for vLLM CLI.

Provides utility functions for port management, log cleanup, and other
server-related operations.
"""
import logging
import socket
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def get_next_available_port(start_port: int = 8000) -> int:
    """
    Get the next available port for a server.

    Searches for an available port starting from the specified port,
    checking both against active servers and actual system availability.

    Args:
        start_port: Port to start searching from (default: 8000)

    Returns:
        Next available port number

    Raises:
        RuntimeError: If no available ports found in valid range
    """
    from .process import get_active_servers

    active_ports = {s.port for s in get_active_servers()}

    port = start_port
    while port in active_ports:
        port += 1

    # Check system availability
    return _find_system_available_port(port)


def _find_system_available_port(start_port: int) -> int:
    """
    Find the next port available on the system.

    Args:
        start_port: Port to start checking from

    Returns:
        Available port number

    Raises:
        RuntimeError: If no ports available in valid range
    """
    port = start_port
    while port <= 65535:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            port += 1

    raise RuntimeError("No available ports found in valid range")


def cleanup_old_logs(days: int = 7) -> int:
    """
    Clean up old log files to prevent disk space accumulation.

    Removes log files from the vLLM CLI logs directory that are older
    than the specified number of days. This helps maintain system
    cleanliness and prevent log directory bloat.

    Args:
        days: Delete logs older than this many days (default: 7)

    Returns:
        Number of log files successfully deleted
    """
    log_dir = Path.home() / ".vllm-cli" / "logs"

    if not log_dir.exists():
        return 0

    cutoff_time = time.time() - (days * 24 * 60 * 60)
    return _delete_old_log_files(log_dir, cutoff_time)


def _delete_old_log_files(log_dir: Path, cutoff_time: float) -> int:
    """
    Delete log files older than the cutoff time.

    Args:
        log_dir: Directory containing log files
        cutoff_time: Unix timestamp cutoff

    Returns:
        Number of files deleted
    """
    deleted = 0

    for log_file in log_dir.glob("*.log"):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                deleted += 1
                logger.debug(f"Deleted old log file: {log_file}")
            except Exception as e:
                logger.warning(f"Failed to delete old log {log_file}: {e}")

    return deleted


def is_port_available(port: int) -> bool:
    """
    Check if a port is available for binding.

    Args:
        port: Port number to check

    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", port))
            return True
    except OSError:
        return False


def validate_port_range(port: int) -> bool:
    """
    Validate that a port is within the valid range.

    Args:
        port: Port number to validate

    Returns:
        True if port is valid, False otherwise
    """
    return 1 <= port <= 65535
