#!/usr/bin/env python3
"""
Common UI utilities and shared components for vLLM CLI.

Provides console setup, panel creation, and shared UI elements.
"""
import logging
import os

from rich import box
from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)

# Determine box style based on environment
# Set VLLM_CLI_ASCII_BOXES=1 to use ASCII boxes for compatibility
use_ascii_boxes = os.environ.get("VLLM_CLI_ASCII_BOXES", "").lower() in (
    "1",
    "true",
    "yes",
)
BOX_STYLE = box.ASCII2 if use_ascii_boxes else box.ROUNDED

# Create console with safe box drawing for better compatibility
console = Console(force_terminal=True, legacy_windows=False)


def create_panel(*args, **kwargs):
    """Create a panel with consistent box style."""
    if "box" not in kwargs:
        kwargs["box"] = BOX_STYLE
    return Panel(*args, **kwargs)
