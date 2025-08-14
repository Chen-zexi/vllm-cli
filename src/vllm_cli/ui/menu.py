#!/usr/bin/env python3
"""
Main menu module for vLLM CLI.

Handles main menu display and navigation routing.
"""
import logging

from .navigation import unified_prompt
from ..server import get_active_servers

logger = logging.getLogger(__name__)


def show_main_menu() -> str:
    """
    Display the main menu and return the selected action.
    """
    # Import here to avoid circular dependencies
    from .server_control import (
        handle_quick_serve,
        handle_serve_with_profile,
        handle_custom_config,
    )
    from .server_monitor import monitor_active_servers
    from .model_manager import handle_model_management
    from .system_info import show_system_info
    from .settings import handle_settings

    # Check for active servers
    active_servers = get_active_servers()

    menu_options = []
    if active_servers:
        menu_options.append(f"Monitor Active Servers ({len(active_servers)})")

    menu_options.extend(
        [
            "Quick Serve (Last Config)",
            "Serve with Profile",
            "Custom Configuration",
            "Model Management",
            "System Information",
            "Settings",
            "Quit",
        ]
    )

    action = unified_prompt("action", "Main Menu", menu_options, allow_back=False)

    if not action or action == "Quit":
        return "quit"

    # Handle menu selections
    if action == "Quick Serve (Last Config)":
        return handle_quick_serve()
    elif action == "Serve with Profile":
        return handle_serve_with_profile()
    elif action == "Custom Configuration":
        return handle_custom_config()
    elif action == "Model Management":
        return handle_model_management()
    elif action == "System Information":
        return show_system_info()
    elif action == "Settings":
        return handle_settings()
    elif "Monitor Active Servers" in action:
        return monitor_active_servers()

    return "continue"
