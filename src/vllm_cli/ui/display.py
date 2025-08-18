#!/usr/bin/env python3
"""
Display utilities for vLLM CLI.

Provides functions for displaying configurations and profiles.
"""
import logging
from typing import Any, Dict, Optional

from rich.table import Table

from ..config import ConfigManager
from .common import console
from .navigation import unified_prompt

logger = logging.getLogger(__name__)


def display_config(config: Dict[str, Any]) -> None:
    """
    Display configuration in a formatted table.
    """
    table = Table(
        title="[bold green]Server Configuration[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    # Skip empty or None values for cleaner display
    for key, value in config.items():
        if value is None or value == "" or key == "extra_args" and not value:
            continue

        if isinstance(value, bool):
            value_str = "✓ Yes" if value else "✗ No"
        elif key == "quantization" and value is None:
            value_str = "None (No quantization)"
        elif key == "extra_args":
            value_str = f'"{value}"'  # Show quotes for clarity
        else:
            value_str = str(value)

        # Format key for display
        display_key = key.replace("_", " ").title()
        if key == "extra_args":
            display_key = "Custom Arguments"
        elif key == "dtype":
            display_key = "Data Type"

        table.add_row(display_key, value_str)

    console.print(table)


def select_profile() -> Optional[str]:
    """
    Select a configuration profile.
    """
    config_manager = ConfigManager()
    all_profiles = config_manager.get_all_profiles()
    profile_choices = list(all_profiles.keys())

    selected = unified_prompt(
        "profile", "Select Configuration Profile", profile_choices, allow_back=True
    )

    if not selected or selected == "BACK":
        return None

    return selected
