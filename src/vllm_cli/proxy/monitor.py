#!/usr/bin/env python3
"""
Proxy monitoring module - wrapper for UI components.

Provides backward compatibility by delegating to UI module.
"""
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import ProxyManager

logger = logging.getLogger(__name__)


def monitor_startup_progress(proxy_manager: "ProxyManager") -> bool:
    """
    Monitor the startup progress of all models with live log display.

    Wrapper function that delegates to UI module.

    Args:
        proxy_manager: The ProxyManager instance

    Returns:
        True if all models started successfully, False otherwise
    """
    from ..ui.proxy_monitor import monitor_startup_progress as ui_monitor_startup

    return ui_monitor_startup(proxy_manager)


def monitor_proxy(proxy_manager: "ProxyManager") -> str:
    """
    Monitor the proxy server and all model engines.

    Wrapper function that delegates to UI module.

    Args:
        proxy_manager: The ProxyManager instance

    Returns:
        Navigation command string
    """
    from ..ui.proxy_monitor import monitor_proxy as ui_monitor_proxy

    return ui_monitor_proxy(proxy_manager)
