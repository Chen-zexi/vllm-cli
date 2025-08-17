#!/usr/bin/env python3
"""
User interface module for vLLM CLI.

Provides rich terminal-based user interfaces for server management,
model selection, and configuration.
"""

from .common import console, create_panel
from .display import display_config, select_profile
from .menu import show_main_menu
from .model_manager import handle_model_management, select_model
from .navigation import unified_prompt
from .profiles import create_custom_profile, manage_profiles
from .server_control import (
    handle_custom_config,
    handle_quick_serve,
    handle_serve_with_profile,
    start_server_with_config,
)
from .server_monitor import monitor_active_servers, monitor_server
from .settings import configure_server_defaults, handle_settings
from .system_info import show_system_info

# Import main UI functions for external use
from .welcome import show_welcome_screen

__all__ = [
    # Main functions
    "show_welcome_screen",
    "show_main_menu",
    # Server control
    "handle_quick_serve",
    "handle_serve_with_profile",
    "handle_custom_config",
    "start_server_with_config",
    # Server monitoring
    "monitor_server",
    "monitor_active_servers",
    # Model management
    "select_model",
    "handle_model_management",
    "show_model_details",
    # System info
    "show_system_info",
    # Profile management
    "manage_profiles",
    "create_custom_profile",
    # Settings
    "handle_settings",
    "configure_server_defaults",
    # Display utilities
    "display_config",
    "select_profile",
    # Common utilities
    "console",
    "create_panel",
    # Navigation
    "unified_prompt",
]
