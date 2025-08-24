#!/usr/bin/env python3
"""
Proxy-specific UI components for vLLM CLI.

Provides reusable UI components for displaying proxy server information,
model registries, and proxy configurations.
"""
import logging
from typing import Any, Dict, Optional

from rich import box
from rich.table import Table

logger = logging.getLogger(__name__)


def format_registration_status(status: str) -> str:
    """
    Format model registration status with appropriate colors.

    Args:
        status: Registration status string

    Returns:
        Formatted status string with Rich markup
    """
    status_map = {
        "available": "[green]available[/green]",
        "pending": "[yellow]pending[/yellow]",
        "not started": "[dim]not started[/dim]",
        "error": "[red]error[/red]",
    }
    return status_map.get(status, "[red]failed[/red]")


def format_model_state(state: str) -> str:
    """
    Format model state with icons and colors.

    Args:
        state: Model state string

    Returns:
        Formatted state string with icon and Rich markup
    """
    state_map = {
        "running": "[green][●] Running[/green]",
        "sleeping": "[yellow][z] Sleeping[/yellow]",
        "starting": "[yellow]Starting...[/yellow]",
        "stopped": "[dim][○] Stopped[/dim]",
    }
    return state_map.get(state, "[red][×] Error[/red]")


def create_model_registry_table(
    registry_status: Dict[str, Any],
    proxy_manager: Optional[Any] = None,
    include_title: bool = False,
) -> Table:
    """
    Create a formatted table showing the current model registry status.

    Used in both refresh_model_registry view and monitor_proxy_logs view
    to display consistent model information.

    Args:
        registry_status: Registry status from get_proxy_registry_status()
        proxy_manager: Optional ProxyManager for checking configured models
        include_title: Whether to include title in the table

    Returns:
        Rich Table with model registry information
    """
    # Create table with optional title
    if include_title:
        table = Table(
            box=box.ROUNDED, title="[bold cyan]Current Model Registry[/bold cyan]"
        )
    else:
        table = Table(box=box.ROUNDED)
    # Standard columns
    table.add_column("Model", style="cyan")
    table.add_column("Port", style="magenta")
    table.add_column("Registration", style="white")
    table.add_column("State", style="white")
    table.add_column("GPU(s)", style="yellow")
    # Get models from registry
    models_info = registry_status.get("models", [])
    all_models = {}
    # Process registry models
    for model in models_info:
        port = model.get("port")
        all_models[port] = {
            "name": model.get("name", f"port_{port}"),
            "port": port,
            "registration": model.get("registration_status", "available"),
            "state": model.get("state", "unknown"),
            "gpu_ids": model.get("gpu_ids", []),
            "sleep_level": model.get("sleep_level", 0),
        }
    # Merge with configured models if proxy_manager provided
    if proxy_manager and hasattr(proxy_manager, "proxy_config"):
        for config in proxy_manager.proxy_config.models:
            if config.port not in all_models:
                # Check if server is running
                is_running = (
                    config.name in proxy_manager.vllm_servers
                    and proxy_manager.vllm_servers[config.name].is_running()
                )
                all_models[config.port] = {
                    "name": config.name,
                    "port": config.port,
                    "registration": "pending" if is_running else "not started",
                    "state": "starting" if is_running else "stopped",
                    "gpu_ids": config.gpu_ids or [],
                }
            else:
                # Update name from config
                all_models[config.port]["name"] = config.name
    # Sort by port and add to table
    for port in sorted(all_models.keys()):
        model = all_models[port]
        # Format status displays
        reg_display = format_registration_status(model["registration"])
        state_display = format_model_state(model["state"])
        # Format GPU IDs
        gpu_str = (
            ",".join(str(g) for g in model["gpu_ids"]) if model["gpu_ids"] else "N/A"
        )
        table.add_row(
            model["name"], str(model["port"]), reg_display, state_display, gpu_str
        )
    # Add empty row if no models
    if not all_models:
        table.add_row("No models registered", "", "", "", "")
    return table


def display_proxy_config_table(config: Any) -> Table:
    """
    Create a table displaying proxy configuration models.

    Args:
        config: ProxyConfig object with models

    Returns:
        Rich Table with proxy configuration
    """
    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Model Path", style="green")
    table.add_column("GPUs", style="magenta")
    table.add_column("Port", style="yellow")
    table.add_column("Profile", style="blue")
    table.add_column("Status", style="dim")

    if config.models:
        for model in config.models:
            gpu_str = (
                ",".join(str(g) for g in model.gpu_ids) if model.gpu_ids else "Auto"
            )
            status = "Enabled" if model.enabled else "Disabled"
            table.add_row(
                model.name,
                (
                    model.model_path[:40] + "..."
                    if len(model.model_path) > 40
                    else model.model_path
                ),
                gpu_str,
                str(model.port),
                model.profile or "None",
                status,
            )

    return table
