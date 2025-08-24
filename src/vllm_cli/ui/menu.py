#!/usr/bin/env python3
"""
Main menu module for vLLM CLI.

Handles main menu display and navigation routing.
"""
import logging

from ..server import get_active_servers
from .navigation import unified_prompt

logger = logging.getLogger(__name__)


def show_main_menu() -> str:
    """
    Display the main menu and return the selected action.
    """
    # Import here to avoid circular dependencies
    from .model_manager import handle_model_management
    from .proxy.menu import (
        get_active_proxy,
        handle_multi_model_proxy,
        manage_running_proxy,
    )
    from .server_control import (
        handle_custom_config,
        handle_quick_serve,
        handle_serve_with_profile,
    )
    from .server_monitor import monitor_active_servers
    from .settings import handle_settings
    from .system_info import show_system_info

    # Check for active proxy and servers
    active_proxy_manager, active_proxy_config = get_active_proxy()
    active_servers = get_active_servers()

    menu_options = []

    # Add proxy monitoring option if proxy is running
    if active_proxy_manager and active_proxy_config:
        try:
            # Check if proxy is actually still running
            if active_proxy_manager.proxy_process:
                menu_options.append("Return to Proxy Monitoring")
        except Exception:
            # If there's any error checking, ignore
            pass

    if active_servers:
        menu_options.append(f"Monitor Active Servers ({len(active_servers)})")

    menu_options.extend(
        [
            "Quick Serve",
            "Serve with Profile",
            "Serve with Custom Config",
            "Multi-Model Proxy (Exp)",
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
    if action == "Return to Proxy Monitoring":
        # Return to proxy monitoring
        if active_proxy_manager and active_proxy_config:
            manage_running_proxy(active_proxy_manager, active_proxy_config)
        return "continue"
    elif action == "Quick Serve":
        return handle_quick_serve()
    elif action == "Serve with Profile":
        return handle_serve_with_profile()
    elif action == "Serve with Custom Config":
        return handle_custom_config()
    elif action == "Multi-Model Proxy (Exp)":
        return handle_multi_model_proxy()
    elif action == "Model Management":
        return handle_model_management()
    elif action == "System Information":
        return show_system_info()
    elif action == "Settings":
        return handle_settings()
    elif "Monitor Active Servers" in action:
        return monitor_active_servers()

    return "continue"
