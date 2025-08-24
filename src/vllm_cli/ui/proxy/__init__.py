"""
Proxy UI module for vLLM CLI.

Consolidates all proxy-related UI components into a single module.
"""

# Export proxy components
from .components import (
    create_model_registry_table,
    display_proxy_config_table,
    format_model_state,
    format_registration_status,
)

# Export proxy control functions
from .control import (
    configure_model_for_proxy,
    configure_proxy_interactively,
    edit_proxy_config,
)

# Export main proxy menu functions
from .menu import (
    get_active_proxy,
    handle_multi_model_proxy,
    manage_models_menu,
    manage_running_proxy,
)

# Export proxy monitoring functions
from .monitor import (
    monitor_individual_model_by_name,
    monitor_model_logs_menu,
    monitor_proxy,
    monitor_proxy_logs,
    monitor_startup_progress,
    refresh_model_registry,
)

__all__ = [
    # Menu functions
    "handle_multi_model_proxy",
    "manage_running_proxy",
    "manage_models_menu",
    "get_active_proxy",
    # Control functions
    "configure_proxy_interactively",
    "configure_model_for_proxy",
    "edit_proxy_config",
    # Monitoring functions
    "monitor_startup_progress",
    "monitor_proxy_logs",
    "monitor_model_logs_menu",
    "monitor_individual_model_by_name",
    "refresh_model_registry",
    "monitor_proxy",
    # Component functions
    "create_model_registry_table",
    "display_proxy_config_table",
    "format_registration_status",
    "format_model_state",
]
