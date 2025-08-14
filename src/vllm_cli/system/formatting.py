#!/usr/bin/env python3
"""
System utility formatters and helpers.

Provides formatting functions for displaying system information
in a human-readable format.
"""


def format_size(bytes_value: int) -> str:
    """
    Format byte size to human readable string.

    Args:
        bytes_value: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"
