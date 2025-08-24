#!/usr/bin/env python3
"""
Proxy monitoring UI module for vLLM CLI.

Handles UI display for monitoring proxy servers and model engines.
"""
import logging
import re
import threading
import time
from threading import Thread
from typing import TYPE_CHECKING, Dict, List, Optional

from rich import box
from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ...config import ConfigManager
from ...system import get_gpu_info
from ..common import console, create_panel
from ..gpu_utils import calculate_gpu_panel_size, create_gpu_status_panel
from ..navigation import unified_prompt
from .components import create_model_registry_table

if TYPE_CHECKING:
    from ...proxy.manager import ProxyManager
    from ...server import VLLMServer

logger = logging.getLogger(__name__)

# Color palette for distinct model names
MODEL_COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red"]


def parse_sleep_wake_logs(log_line: str) -> Optional[Dict]:
    """
    Parse sleep/wake completion logs for key information.

    Args:
        log_line: Single log line to parse

    Returns:
        Dict with parsed information or None if no pattern matched
    """
    # Check for sleep start
    # Pattern: "POST /sleep?level=1"
    if "POST /sleep" in log_line:
        return {"type": "sleep_start", "message": "Sleep operation initiated"}

    # Check for wake start
    # Pattern: "wake up the engine with tags: None"
    if "wake up the engine" in log_line:
        return {"type": "wake_start", "message": "Wake operation initiated"}

    # Check for sleep completion
    # Pattern: "It took 17.893898 seconds to fall asleep."
    sleep_match = re.search(r"It took ([\d.]+) seconds to fall asleep", log_line)
    if sleep_match:
        return {
            "type": "sleep_complete",
            "time": float(sleep_match.group(1)),
            "message": f"Sleep completed in {sleep_match.group(1)} seconds",
        }

    # Check for wake completion
    # Pattern: "It took 24.368552 seconds to wake up tags {'kv_cache', 'weights'}."
    wake_match = re.search(r"It took ([\d.]+) seconds to wake up", log_line)
    if wake_match:
        tags = None
        tags_match = re.search(r"tags ({[^}]+})", log_line)
        if tags_match:
            tags = tags_match.group(1)
        return {
            "type": "wake_complete",
            "time": float(wake_match.group(1)),
            "tags": tags,
            "message": f"Wake completed in {wake_match.group(1)} seconds",
        }

    # Check for memory info during sleep
    # Pattern: "Sleep mode freed 90.40 GiB memory, 3.30 GiB memory is still in use."
    memory_match = re.search(
        r"Sleep mode freed ([\d.]+) GiB memory, ([\d.]+) GiB memory is still in use",
        log_line,
    )
    if memory_match:
        return {
            "type": "memory_info",
            "freed": float(memory_match.group(1)),
            "remaining": float(memory_match.group(2)),
            "message": f"Freed {memory_match.group(1)} GiB, {memory_match.group(2)} GiB still in use",
        }

    return None


def create_models_log_panel(
    proxy_manager: "ProxyManager",
    startup_status: dict = None,
    total_lines: int = None,
    show_status: bool = True,
    available_height: int = None,
) -> list:
    """
    Create consistent model logs panel for monitoring.

    Args:
        proxy_manager: The ProxyManager instance
        startup_status: Optional dict with startup status for each model
        total_lines: Total number of log lines to display across all models
        show_status: Whether to show status indicators
        available_height: Available terminal height for dynamic calculation

    Returns:
        List of Rich Text objects for display
    """
    models_content = []

    # Get active models
    active_models = [m for m in proxy_manager.proxy_config.models if m.enabled]
    num_models = len(active_models)

    # Calculate lines_per_model dynamically based on available space
    if available_height and num_models > 0:
        # Reserve space for separators and headers
        # Each model after first needs 2 lines for separator
        separator_lines = (num_models - 1) * 2 if num_models > 1 else 0
        # Each model needs 1 line for header
        header_lines = num_models

        # Calculate truly available space for logs
        overhead = separator_lines + header_lines
        available_for_logs = max(0, available_height - overhead)

        if available_for_logs > 0:
            # Distribute available space equally among all models
            # No arbitrary caps - use what's available
            lines_per_model = available_for_logs // num_models

            # Ensure at least 1 line per model if we have any space
            lines_per_model = max(1, lines_per_model)
        else:
            # No space for logs, just show headers
            lines_per_model = 0
    elif total_lines is not None:
        # Use provided total_lines
        lines_per_model = total_lines // num_models if num_models > 0 else total_lines
    else:
        # No height info provided, use conservative fallback
        lines_per_model = 50 // num_models if num_models > 0 else 50

    for idx, model_config in enumerate(active_models):
        model_name = model_config.name
        model_color = MODEL_COLORS[idx % len(MODEL_COLORS)]

        # Add separator if not first model
        if idx > 0:
            models_content.append(Text())  # Empty line
            models_content.append(Rule("─" * 60, style="dim"))

        # Create model header with distinct color
        header = Text()
        header.append(f"[{model_name}]", style=f"bold {model_color}")

        # Add status if provided and requested
        if show_status and startup_status:
            status = startup_status.get(model_name, "unknown")
            header.append(" - ")

            if status == "ready":
                header.append("✓ Ready", style="green")
            elif status == "failed":
                header.append("✗ Failed", style="red")
            elif status == "starting":
                header.append("⠋ Starting...", style="yellow")
            elif status == "pending":
                header.append("Pending", style="dim")
            else:
                # For overview mode, check if server is running
                if model_name in proxy_manager.vllm_servers:
                    server = proxy_manager.vllm_servers[model_name]
                    if server.is_running():
                        header.append("● Running", style="green")
                    else:
                        header.append("○ Stopped", style="red")

        # Add port and GPU info
        header.append(f" (Port: {model_config.port}", style="dim")
        if model_config.gpu_ids:
            gpu_str = ",".join(str(g) for g in model_config.gpu_ids)
            header.append(f", GPU: {gpu_str}", style="dim")
        header.append(")", style="dim")

        models_content.append(header)

        # Get and display logs
        if model_name in proxy_manager.vllm_servers:
            server = proxy_manager.vllm_servers[model_name]
            recent_logs = server.get_recent_logs(lines_per_model)

            if recent_logs:
                for log in recent_logs:
                    # Rich will automatically wrap long lines
                    log_text = Text(f"  {log}", style="dim white")
                    models_content.append(log_text)
            else:
                waiting_text = Text("  Waiting for logs...", style="dim")
                models_content.append(waiting_text)
        else:
            pending_text = Text("  Server not started yet...", style="dim")
            models_content.append(pending_text)

    return models_content


def monitor_startup_progress(proxy_manager: "ProxyManager") -> bool:
    """
    Monitor the startup progress of all models with live log display.

    Shows real-time logs from all models as they start up concurrently,
    and registers them with the proxy as they become ready.

    Args:
        proxy_manager: The ProxyManager instance

    Returns:
        True if all models started successfully, False otherwise
    """
    console.print("\n[bold cyan]Model Startup Monitor[/bold cyan]")
    console.print("[dim]Showing real-time logs from all models...[/dim]\n")

    # Get UI preferences
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()
    monitor_refresh_rate = ui_prefs.get("monitor_refresh_rate", 1.0)

    # Track startup status for each model
    startup_status = {}
    startup_complete = threading.Event()

    def check_startup_completion():
        """Background thread to check startup completion and register models."""
        models_to_register = []

        while not startup_complete.is_set():
            all_ready = True

            for model_config in proxy_manager.proxy_config.models:
                if not model_config.enabled:
                    continue

                model_name = model_config.name

                # Skip if already processed
                if model_name in startup_status and startup_status[model_name] in [
                    "ready",
                    "failed",
                ]:
                    continue

                # Check if server exists
                if model_name not in proxy_manager.vllm_servers:
                    startup_status[model_name] = "pending"
                    all_ready = False
                    continue

                server = proxy_manager.vllm_servers[model_name]

                # Check if server is running
                if not server.is_running():
                    startup_status[model_name] = "failed"
                    logger.error(f"Server {model_name} failed to start")
                    continue

                # Check logs for startup completion
                recent_logs = server.get_recent_logs(20)
                if recent_logs:
                    for log in recent_logs:
                        if "application startup complete" in log.lower():
                            if model_name not in models_to_register:
                                startup_status[model_name] = "ready"
                                models_to_register.append(model_config)
                                logger.info(f"Model {model_name} is ready")
                            break
                    else:
                        startup_status[model_name] = "starting"
                        all_ready = False
                else:
                    startup_status[model_name] = "starting"
                    all_ready = False

            # Register ready models with proxy
            for model_config in models_to_register[:]:
                if (
                    proxy_manager.proxy_process
                    and proxy_manager.proxy_process.is_running()
                ):
                    # Register the model
                    model_data = {
                        "name": model_config.name,
                        "port": model_config.port,
                        "model_path": model_config.model_path,
                        "gpu_ids": model_config.gpu_ids,
                        **model_config.config_overrides,
                    }

                    if proxy_manager._proxy_api_request(
                        "POST", "/proxy/add_model", model_data
                    ):
                        logger.debug(
                            f"Registered model '{model_config.name}' with proxy"
                        )

                        # Also register aliases
                        aliases = model_config.config_overrides.get("aliases", [])
                        for alias in aliases:
                            alias_data = model_data.copy()
                            alias_data["name"] = alias
                            proxy_manager._proxy_api_request(
                                "POST", "/proxy/add_model", alias_data
                            )

                    models_to_register.remove(model_config)

            if all_ready:
                startup_complete.set()
                break

            time.sleep(1)

    # Start background thread to check completion
    check_thread = Thread(target=check_startup_completion, daemon=True)
    check_thread.start()

    # Get GPU info for layout
    gpu_info = get_gpu_info()
    gpu_panel_size = calculate_gpu_panel_size(len(gpu_info) if gpu_info else 0)

    # Create layout for monitoring with GPU panel
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="status", size=4),
        Layout(name="gpu", size=gpu_panel_size),
        Layout(name="divider", size=1),
        Layout(name="models"),
        Layout(name="footer", size=1),
    )

    # Header
    layout["header"].update(
        create_panel(
            "[bold cyan]Starting Model Servers...[/bold cyan]",
            title="Proxy Startup",
            border_style="cyan",
        )
    )

    # Initial GPU panel
    layout["gpu"].update(create_gpu_status_panel())

    # Divider
    layout["divider"].update(Rule("Model Logs", style="cyan"))

    # Footer
    layout["footer"].update(
        Align.center(Text("Press Ctrl+C to cancel", style="dim yellow"))
    )

    # Track if we've shown the final state
    final_state_shown = False

    try:
        with Live(layout, console=console, refresh_per_second=monitor_refresh_rate):
            while not startup_complete.is_set() or not final_state_shown:
                # Recalculate terminal height in case of resize
                current_height = console.height

                # Calculate available space for logs
                # Fixed: header(3) + status(4) + gpu(varies) + divider(1) + footer(1)
                fixed_height = 9 + gpu_panel_size

                # Available height for logs with padding
                available_log_height = max(20, current_height - fixed_height - 2)

                # Update status
                status_table = Table(show_header=False, box=None)
                status_table.add_column("Model", style="cyan")
                status_table.add_column("Status", style="green")

                ready_count = sum(1 for s in startup_status.values() if s == "ready")
                total_count = len(
                    [m for m in proxy_manager.proxy_config.models if m.enabled]
                )

                status_table.add_row(
                    "Progress", f"{ready_count}/{total_count} models ready"
                )

                layout["status"].update(
                    create_panel(
                        status_table,
                        title="Status",
                        border_style=(
                            "green" if ready_count == total_count else "yellow"
                        ),
                    )
                )

                # Update GPU panel periodically
                layout["gpu"].update(create_gpu_status_panel())

                # Use common function to create models log panel with dynamic height
                models_content = create_models_log_panel(
                    proxy_manager,
                    startup_status=startup_status,
                    available_height=available_log_height,  # Use dynamic height
                    show_status=True,
                )

                # Update models panel
                if models_content:
                    layout["models"].update(Padding(Group(*models_content), (1, 2)))
                else:
                    layout["models"].update(
                        Padding(Text("[dim]No models starting...[/dim]"), (1, 2))
                    )

                # Check if all models are ready and mark final state shown
                if startup_complete.is_set() and not final_state_shown:
                    # Give one more iteration to show the final "Ready" state
                    final_state_shown = True
                    time.sleep(1.0)  # Show final state for 1 second
                elif not startup_complete.is_set():
                    time.sleep(0.5)

    except KeyboardInterrupt:
        console.print("\n[yellow]Startup cancelled by user.[/yellow]")
        startup_complete.set()
        return False

    # Wait for check thread to finish
    check_thread.join(timeout=1)

    # Final status
    success_count = sum(1 for s in startup_status.values() if s == "ready")
    fail_count = sum(1 for s in startup_status.values() if s == "failed")

    # Get list of failed models
    failed_models = [
        name for name, status in startup_status.items() if status == "failed"
    ]

    if fail_count > 0:
        console.print(f"\n[red]✗ {fail_count} model(s) failed to start[/red]")

        # Offer to view logs for failed models
        if failed_models:
            handle_failed_models(proxy_manager, failed_models)

        return False
    else:
        console.print(
            f"\n[green]✓ All {success_count} model(s) started successfully[/green]"
        )
        return True


def handle_failed_models(
    proxy_manager: "ProxyManager", failed_models: List[str]
) -> None:
    """
    Handle failed models by offering log viewing options.

    Args:
        proxy_manager: The ProxyManager instance
        failed_models: List of model names that failed to start
    """
    from ..log_viewer import show_log_menu

    for model_name in failed_models:
        if model_name in proxy_manager.vllm_servers:
            server = proxy_manager.vllm_servers[model_name]
            console.print(f"\n[yellow]Model '{model_name}' failed to start[/yellow]")

            # Show last few log lines
            recent_logs = server.get_recent_logs(5)
            if recent_logs:
                console.print("[bold]Last logs:[/bold]")
                for log in recent_logs:
                    console.print(f"  {log}")

            # Offer to view full logs
            view_logs = (
                input(f"\nView full logs for {model_name}? (y/N): ").strip().lower()
            )
            if view_logs in ["y", "yes"]:
                show_log_menu(server)
            else:
                if server.log_path:
                    console.print(f"[dim]Log file: {server.log_path}[/dim]")


def refresh_model_registry(proxy_manager: "ProxyManager"):
    """
    Refresh the model registry by scanning and registering all configured models.

    Args:
        proxy_manager: The ProxyManager instance
    """
    console.print("\n[bold cyan]Refreshing Model Registry[/bold cyan]")
    console.print("[dim]Scanning for models to register...[/dim]\n")

    # Call the refresh method
    with console.status("[cyan]Refreshing model registrations...[/cyan]"):
        result = proxy_manager.refresh_model_registrations()

    if result.get("status") == "error":
        console.print(f"[red]✗ Error: {result.get('message', 'Unknown error')}[/red]")
    else:
        # Display results
        summary = result.get("summary", {})
        details = result.get("details", {})

        # Create a summary table for what changed
        summary_table = Table(box=box.ROUNDED)
        summary_table.add_column("Status", style="cyan")
        summary_table.add_column("Count", style="white")
        summary_table.add_column("Models", style="dim")

        # Only add rows if there were changes
        has_changes = False
        if summary.get("registered", 0) > 0:
            has_changes = True
            registered = details.get("newly_registered", [])
            summary_table.add_row(
                "✓ Newly Registered",
                str(summary.get("registered", 0)),
                ", ".join(registered) if registered else "-",
            )

        if summary.get("failed", 0) > 0:
            has_changes = True
            failed = details.get("newly_failed", [])
            summary_table.add_row(
                "✗ Newly Failed",
                str(summary.get("failed", 0)),
                ", ".join(failed) if failed else "-",
            )

        if summary.get("removed", 0) > 0:
            has_changes = True
            removed = details.get("removed", [])
            summary_table.add_row(
                "[×] Removed (stopped)",
                str(summary.get("removed", 0)),
                ", ".join(removed) if removed else "-",
            )

        # Show the summary table only if there were changes
        if has_changes:
            summary_panel = Panel(
                summary_table,
                title="[bold cyan]Registration Changes[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
            console.print(summary_panel)

        # Show status for already available models
        if summary.get("already_available", 0) > 0:
            available_list = details.get("already_available", [])
            if available_list:
                console.print(
                    f"\n[green]✓ {len(available_list)} model(s) verified as available:[/green] "
                    f"{', '.join(available_list)}"
                )

        # Get full registry status to show current state of all models
        registry_status = proxy_manager.get_proxy_registry_status()
        if registry_status:
            # Use shared component to create table
            status_table = create_model_registry_table(
                registry_status, proxy_manager, include_title=True
            )

            # Display if we have models
            if registry_status.get("models") or (
                proxy_manager and proxy_manager.proxy_config.models
            ):
                console.print()  # Add spacing
                console.print(status_table)

        # Show detailed failure reasons if any
        if details.get("failed"):
            console.print("\n[yellow]Failed Registration Details:[/yellow]")
            for failure in details["failed"]:
                console.print(
                    f"  • {failure.get('name', 'unknown')}: "
                    f"[dim]{failure.get('reason', 'Unknown reason')}[/dim]"
                )

    input("\nPress Enter to continue...")


def monitor_model_logs_menu(proxy_manager: "ProxyManager") -> str:
    """
    Submenu for selecting model monitoring mode.

    Args:
        proxy_manager: The ProxyManager instance

    Returns:
        Navigation command string ('back' or 'stop')
    """
    console.print("[bold cyan]Monitor Model Logs[/bold cyan]\n")

    options = [
        "Overview - Monitor all models",
        "Individual - Monitor specific model",
    ]

    choice = unified_prompt(
        "model_monitoring_mode", "Select monitoring mode", options, allow_back=True
    )

    if choice == "BACK":
        return "back"

    if choice == "Overview - Monitor all models":
        return monitor_proxy_overview(proxy_manager)
    elif choice == "Individual - Monitor specific model":
        return monitor_individual_model(proxy_manager)

    return "back"


def monitor_proxy(proxy_manager: "ProxyManager") -> str:
    """
    Monitor the proxy server and all model engines.

    Provides options to view overall status or individual model logs.

    Args:
        proxy_manager: The ProxyManager instance

    Returns:
        Navigation command string
    """
    while True:
        console.clear()
        console.print("[bold cyan]Multi-Model Proxy Monitor[/bold cyan]")
        console.print("[dim]Press Ctrl+C to exit monitoring[/dim]\n")

        # Get UI preferences
        config_manager = ConfigManager()
        ui_prefs = config_manager.get_ui_preferences()
        ui_prefs.get("show_gpu_in_monitor", True)
        ui_prefs.get("monitor_refresh_rate", 1.0)

        # Show monitoring options
        options = [
            "Overview - Monitor all models",
            "Individual - Monitor specific model logs",
            "Proxy Server Logs - View proxy server logs",
            "Refresh Model Registry - Scan and register models",
            "Status - Show current status",
            "Back to proxy menu",
        ]

        choice = unified_prompt(
            "proxy_monitor", "Select monitoring mode", options, allow_back=True
        )

        if choice == "BACK" or choice == "Back to proxy menu":
            return "back"

        result = None
        if choice == "Overview - Monitor all models":
            result = monitor_proxy_overview(proxy_manager)
        elif choice == "Individual - Monitor specific model logs":
            result = monitor_individual_model(proxy_manager)
        elif choice == "Proxy Server Logs - View proxy server logs":
            result = monitor_proxy_logs(proxy_manager)
        elif choice == "Refresh Model Registry - Scan and register models":
            refresh_model_registry(proxy_manager)
            continue  # Loop back to menu
        elif choice == "Status - Show current status":
            show_proxy_status(proxy_manager)
            input("\nPress Enter to continue...")
            continue  # Loop back to menu

        # Handle navigation results from monitoring functions
        if result == "back":
            return "back"  # Exit to proxy running menu
        elif result == "menu":
            continue  # Loop back to monitoring menu
        # If monitoring function returns another monitoring function,
        # it will handle the transition internally
        elif result:
            # For any other result, assume we should exit
            return result

    return "back"


def monitor_proxy_overview(proxy_manager: "ProxyManager") -> str:
    """
    Monitor overview of all models in the proxy.

    Shows status of all models and aggregated metrics.
    """
    console.print("[bold cyan]Proxy Overview Monitor[/bold cyan]")
    console.print("[dim]Press Ctrl+C for menu options[/dim]\n")

    # Get UI preferences
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()
    show_gpu = ui_prefs.get("show_gpu_in_monitor", True)
    monitor_refresh_rate = ui_prefs.get("monitor_refresh_rate", 1.0)

    try:
        # Get GPU info for panel sizing
        gpu_info = get_gpu_info() if show_gpu else None
        gpu_panel_size = (
            calculate_gpu_panel_size(len(gpu_info) if gpu_info else 0)
            if show_gpu
            else 0
        )

        # Create layout - simplified without models table
        layout = Layout()
        if show_gpu:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="proxy_status", size=4),
                Layout(name="gpu", size=gpu_panel_size),
                Layout(name="log_divider", size=1),
                Layout(name="logs"),  # Takes remaining space
                Layout(name="footer", size=1),
            )
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="proxy_status", size=4),
                Layout(name="log_divider", size=1),
                Layout(name="logs"),  # Takes remaining space
                Layout(name="footer", size=1),
            )

        # Header
        header_text = Text(
            f"Multi-Model Proxy Monitor - {len(proxy_manager.vllm_servers)} Models",
            style="bold cyan",
            justify="center",
        )
        layout["header"].update(Padding(header_text, (1, 0)))

        # Footer
        layout["footer"].update(
            Align.center(
                Text(
                    "Press Ctrl+C for menu options",
                    style="dim cyan",
                )
            )
        )

        with Live(layout, console=console, refresh_per_second=monitor_refresh_rate):
            while True:
                # Recalculate terminal height in case of resize
                current_height = console.height

                # Calculate available space for logs
                # Fixed: header(3) + proxy_status(4) + divider(1) + footer(1) = 9
                fixed_height = 9
                if show_gpu:
                    fixed_height += gpu_panel_size

                # Available height for logs with padding
                available_log_height = max(20, current_height - fixed_height - 2)

                # Update proxy status
                proxy_status = Table(show_header=False, box=None)
                proxy_status.add_column("Key", style="cyan")
                proxy_status.add_column("Value", style="magenta")

                proxy_status.add_row(
                    "Proxy Status",
                    (
                        "[green]Running[/green]"
                        if proxy_manager.proxy_process
                        and proxy_manager.proxy_process.is_running()
                        else "[red]Stopped[/red]"
                    ),
                )
                proxy_status.add_row("Proxy Port", str(proxy_manager.proxy_config.port))
                proxy_status.add_row(
                    "Active Models", str(len(proxy_manager.vllm_servers))
                )

                layout["proxy_status"].update(
                    create_panel(
                        proxy_status,
                        title="Proxy Server",
                        border_style=(
                            "green"
                            if proxy_manager.proxy_process
                            and proxy_manager.proxy_process.is_running()
                            else "red"
                        ),
                    )
                )

                # Update GPU panel if enabled
                if show_gpu:
                    gpu_panel = create_gpu_status_panel()
                    layout["gpu"].update(gpu_panel)

                # Update log divider
                layout["log_divider"].update(Rule("Model Logs", style="cyan"))

                # Use common function to create models log panel with dynamic height
                models_log_content = create_models_log_panel(
                    proxy_manager,
                    startup_status=None,  # No startup status in overview mode
                    available_height=available_log_height,  # Pass dynamic height
                    show_status=True,  # Show status since we removed the table
                )

                if models_log_content:
                    logs_content = Padding(Group(*models_log_content), (1, 2))
                else:
                    logs_content = Padding(
                        Text("Waiting for logs...", style="dim yellow"), (1, 2)
                    )

                layout["logs"].update(logs_content)

                time.sleep(0.5)

    except KeyboardInterrupt:
        pass

    console.print("\n[yellow]Monitoring stopped.[/yellow]")
    console.print("[dim]Press Ctrl+C to stop proxy servers[/dim]")

    try:
        input("\nPress Enter to return to monitoring menu...")
        return "back"
    except KeyboardInterrupt:
        # User wants to stop proxy
        if (
            unified_prompt(
                "confirm_stop_overview",
                "Stop all proxy servers?",
                ["Yes, stop all servers", "No, keep running"],
                allow_back=False,
            )
            == "Yes, stop all servers"
        ):
            return "stop"
        return "back"


def monitor_individual_model_by_name(
    proxy_manager: "ProxyManager", model_name: str
) -> str:
    """
    Monitor logs of a specific model by name.

    Args:
        proxy_manager: The ProxyManager instance
        model_name: Name of the model to monitor

    Returns:
        Navigation command string
    """
    if model_name not in proxy_manager.vllm_servers:
        console.print(f"[red]Model '{model_name}' not found or not running[/red]")
        input("\nPress Enter to continue...")
        return "back"

    server = proxy_manager.vllm_servers[model_name]
    console.print(f"\n[bold cyan]Monitoring: {model_name}[/bold cyan]")
    console.print(f"[dim]Port: {server.port} | Press Ctrl+C to stop monitoring[/dim]\n")

    # Monitor the model logs
    return monitor_model_engine(server, model_name, proxy_manager)


def monitor_individual_model(proxy_manager: "ProxyManager") -> str:
    """
    Monitor logs of a specific model engine.

    Allows user to select a model and view its logs in real-time.
    """
    if not proxy_manager.vllm_servers:
        console.print("[yellow]No models are currently running.[/yellow]")
        input("\nPress Enter to continue...")
        return "back"

    # List available models
    console.print("\n[bold cyan]Select Model to Monitor[/bold cyan]")

    model_options = []
    for model_name, server in proxy_manager.vllm_servers.items():
        status = "Running" if server.is_running() else "Stopped"
        model_options.append(f"{model_name} (Port {server.port}, {status})")

    choice = unified_prompt(
        "select_model", "Select a model to monitor", model_options, allow_back=True
    )

    if choice == "BACK":
        return "back"

    # Extract model name from choice
    model_name = choice.split(" (")[0]

    if model_name not in proxy_manager.vllm_servers:
        console.print(f"[red]Model '{model_name}' not found.[/red]")
        return monitor_individual_model(proxy_manager)

    server = proxy_manager.vllm_servers[model_name]

    # Monitor this specific model (reuse existing monitor function)
    return monitor_model_engine(server, model_name, proxy_manager)


def monitor_model_engine(
    server: "VLLMServer", model_name: str, proxy_manager: "ProxyManager"
) -> str:
    """
    Monitor a specific model engine with its logs.

    Similar to single-model monitoring but within proxy context.
    """
    console.print(f"[bold cyan]Monitoring Model: {model_name}[/bold cyan]")
    console.print("[dim]Press Ctrl+C for menu options[/dim]\n")

    # Get UI preferences
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()
    show_gpu = ui_prefs.get("show_gpu_in_monitor", True)
    monitor_refresh_rate = ui_prefs.get("monitor_refresh_rate", 1.0)

    try:
        # Get GPU info
        gpu_info = get_gpu_info() if show_gpu else None
        gpu_panel_size = (
            calculate_gpu_panel_size(len(gpu_info) if gpu_info else 0)
            if show_gpu
            else 0
        )

        # Create layout
        layout = Layout()
        if show_gpu:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="status", size=6),
                Layout(name="gpu", size=gpu_panel_size),
                Layout(name="logs"),
                Layout(name="footer", size=2),
            )
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="status", size=6),
                Layout(name="logs"),
                Layout(name="footer", size=2),
            )

        # Header
        header_text = Text(
            f"Model Engine Monitor - {model_name}", style="bold cyan", justify="center"
        )
        layout["header"].update(Padding(header_text, (1, 0)))

        # Footer
        footer_text = "Press Ctrl+C for menu options"
        layout["footer"].update(Align.center(Text(footer_text, style="dim cyan")))

        # Track operation state for completion detection
        operation_state = {
            "type": None,  # 'sleeping', 'waking', or None
            "started_this_session": False,  # Track if we saw the start in THIS monitoring session
            "memory_freed": None,
            "memory_remaining": None,
            "completion_time": None,
            "completed": False,
            "notification_shown_until": None,  # Show notification for 5 seconds
            "exit_after": None,  # Track when to auto-exit after completion
        }

        # Flag to track if server failed during monitoring
        server_failed = False

        with Live(layout, console=console, refresh_per_second=monitor_refresh_rate):
            while True:
                # Update status
                status_table = Table(show_header=False, box=None)
                status_table.add_column("Key", style="cyan")
                status_table.add_column("Value", style="magenta")

                status_table.add_row(
                    "Status",
                    (
                        "[green]Running[/green]"
                        if server.is_running()
                        else "[red]Stopped[/red]"
                    ),
                )
                status_table.add_row("Model", model_name)
                status_table.add_row("Port", str(server.port))
                status_table.add_row(
                    "PID", str(server.process.pid) if server.process else "N/A"
                )

                # Add GPU info
                model_config = proxy_manager._get_model_config_by_name(model_name)
                if model_config and model_config.gpu_ids:
                    gpu_str = ",".join(str(g) for g in model_config.gpu_ids)
                    status_table.add_row("GPUs", gpu_str)

                layout["status"].update(
                    create_panel(
                        status_table,
                        title="Engine Status",
                        border_style="green" if server.is_running() else "red",
                    )
                )

                # Update GPU panel
                if show_gpu:
                    gpu_panel = create_gpu_status_panel()
                    layout["gpu"].update(gpu_panel)

                # Update logs dynamically based on available space
                # Calculate available height for logs
                current_height = console.height
                # Fixed: header(3) + status(6) + footer(2) = 11
                fixed_height = 11
                if show_gpu:
                    fixed_height += gpu_panel_size

                # Account for logs panel overhead: Rule(1) + Padding(1) + margin(1)
                logs_panel_overhead = 3

                # Available lines for logs with proper accounting
                available_lines = max(
                    10, current_height - fixed_height - logs_panel_overhead
                )
                recent_logs = server.get_recent_logs(available_lines)

                # Check for sleep/wake operations in logs
                if recent_logs:
                    for log_line in recent_logs:
                        parsed = parse_sleep_wake_logs(log_line)
                        if parsed:
                            # Process start patterns - track if we see them in this session
                            if parsed["type"] == "sleep_start":
                                # Only process if we haven't already started tracking this operation
                                if (
                                    not operation_state.get("started_this_session")
                                    or operation_state["type"] != "sleeping"
                                ):
                                    # New sleep operation in this session
                                    operation_state["type"] = "sleeping"
                                    operation_state["started_this_session"] = True
                                    operation_state["completed"] = False
                                    operation_state["memory_freed"] = None
                                    operation_state["memory_remaining"] = None
                                    operation_state["completion_time"] = None
                                    operation_state["notification_shown_until"] = None
                                    operation_state["exit_after"] = None
                            elif parsed["type"] == "wake_start":
                                # Only process if we haven't already started tracking this operation
                                if (
                                    not operation_state.get("started_this_session")
                                    or operation_state["type"] != "waking"
                                ):
                                    # New wake operation in this session
                                    operation_state["type"] = "waking"
                                    operation_state["started_this_session"] = True
                                    operation_state["completed"] = False
                                    operation_state["completion_time"] = None
                                    operation_state["notification_shown_until"] = None
                                    operation_state["memory_freed"] = None
                                    operation_state["memory_remaining"] = None
                                    operation_state["exit_after"] = None

                            # Always process completion and info patterns
                            if parsed["type"] == "memory_info":
                                # Store memory info for sleep operation
                                operation_state["memory_freed"] = parsed["freed"]
                                operation_state["memory_remaining"] = parsed[
                                    "remaining"
                                ]
                            elif parsed["type"] == "sleep_complete":
                                # Only process completion if we saw the start in this session
                                if (
                                    operation_state.get("started_this_session")
                                    and operation_state["type"] == "sleeping"
                                ):
                                    # Sleep operation completed
                                    operation_state["completion_time"] = parsed["time"]
                                    operation_state["completed"] = True
                                    operation_state["notification_shown_until"] = (
                                        time.time() + 5
                                    )  # Show for 5 seconds
                                    operation_state["exit_after"] = (
                                        time.time() + 8
                                    )  # Exit 3 seconds after notification ends
                                    break  # Exit the log checking loop immediately
                            elif parsed["type"] == "wake_complete":
                                # Only process completion if we saw the start in this session
                                if (
                                    operation_state.get("started_this_session")
                                    and operation_state["type"] == "waking"
                                ):
                                    # Wake operation completed
                                    operation_state["completion_time"] = parsed["time"]
                                    operation_state["completed"] = True
                                    operation_state["notification_shown_until"] = (
                                        time.time() + 5
                                    )
                                    operation_state["exit_after"] = (
                                        time.time() + 8
                                    )  # Exit 3 seconds after notification ends
                                    break  # Exit the log checking loop immediately

                # Build logs display with notification if needed
                if operation_state.get("type") and not operation_state.get("completed"):
                    # Operation in progress
                    if operation_state["type"] == "sleeping":
                        progress_text = (
                            "[yellow]⏳ Sleep operation in progress...[/yellow]"
                        )
                    else:  # waking
                        progress_text = (
                            "[yellow]⏳ Wake operation in progress...[/yellow]"
                        )

                    # Show progress indicator with logs
                    log_text = Text(
                        (
                            "\n".join(recent_logs)
                            if recent_logs
                            else "Waiting for logs..."
                        ),
                        style="dim white",
                    )
                    logs_content = Group(
                        Panel(progress_text, border_style="yellow", box=box.ROUNDED),
                        Rule(f"Engine Logs - {model_name}", style="yellow"),
                        Padding(log_text, (0, 2)),
                    )
                elif operation_state.get("completed") and operation_state.get(
                    "notification_shown_until"
                ):
                    if time.time() < operation_state["notification_shown_until"]:
                        # Check if this notification is still relevant (no new operation started)
                        show_notification = True

                        # Check recent logs for new operations that would invalidate this notification
                        for log_line in (
                            recent_logs[-5:] if len(recent_logs) > 5 else recent_logs
                        ):
                            # Check for new wake operation after sleep completion
                            if (
                                "wake up the engine" in log_line
                                and operation_state["type"] == "sleeping"
                            ):
                                show_notification = False
                                operation_state["notification_shown_until"] = None
                                break
                            # Check for new sleep operation after wake completion
                            elif (
                                "POST /sleep" in log_line
                                and operation_state["type"] == "waking"
                            ):
                                show_notification = False
                                operation_state["notification_shown_until"] = None
                                break

                        if show_notification:
                            # Show completion notification
                            notification_lines = []

                            if operation_state["type"] == "sleeping":
                                notification_lines.append(
                                    "[bold green]✓ SLEEP COMPLETED[/bold green]"
                                )
                                notification_lines.append(
                                    f"[cyan]Time taken: {operation_state['completion_time']:.2f} seconds[/cyan]"
                                )
                                if operation_state.get("memory_freed"):
                                    notification_lines.append(
                                        f"[yellow]Memory freed: {operation_state['memory_freed']:.2f} GiB[/yellow]"
                                    )
                                    notification_lines.append(
                                        f"[dim]Memory in use: {operation_state['memory_remaining']:.2f} GiB[/dim]"
                                    )
                            elif operation_state["type"] == "waking":
                                notification_lines.append(
                                    "[bold green]✓ WAKE COMPLETED[/bold green]"
                                )
                                notification_lines.append(
                                    f"[cyan]Time taken: {operation_state['completion_time']:.2f} seconds[/cyan]"
                                )
                                notification_lines.append(
                                    "[green]Model is now fully operational[/green]"
                                )

                            # Create notification panel
                            notification_panel = Panel(
                                "\n".join(notification_lines),
                                title="[bold yellow]Operation Complete[/bold yellow]",
                                border_style="green",
                                box=box.DOUBLE,
                            )

                            # Show logs below notification
                            log_text = Text(
                                "\n".join(recent_logs[-10:]), style="dim white"
                            )  # Show last 10 lines

                            logs_content = Group(
                                notification_panel,
                                Rule(f"Engine Logs - {model_name}", style="yellow"),
                                Padding(log_text, (0, 2)),
                            )
                        else:
                            # Notification was invalidated by new operation, show normal logs
                            log_text = Text(
                                (
                                    "\n".join(recent_logs)
                                    if recent_logs
                                    else "Waiting for logs..."
                                ),
                                style="dim white",
                            )
                            logs_content = Group(
                                Rule(f"Engine Logs - {model_name}", style="yellow"),
                                Padding(log_text, (0, 2)),
                            )
                    else:
                        # Notification expired, clear the completed flag
                        operation_state["completed"] = False
                        operation_state["notification_shown_until"] = None

                        # Normal log display
                        log_text = Text("\n".join(recent_logs), style="dim white")
                        logs_content = Group(
                            Rule(f"Engine Logs - {model_name}", style="yellow"),
                            Padding(log_text, (0, 2)),
                        )
                else:
                    # Normal log display
                    if recent_logs:
                        log_text = Text("\n".join(recent_logs), style="dim white")
                    else:
                        log_text = Text("Waiting for logs...", style="dim yellow")

                    logs_content = Group(
                        Rule(f"Engine Logs - {model_name}", style="yellow"),
                        Padding(log_text, (0, 2)),
                    )

                layout["logs"].update(logs_content)

                # Auto-exit after completion and notification display
                if operation_state.get("exit_after"):
                    if time.time() > operation_state["exit_after"]:
                        # Time to exit after showing completion
                        break

                # Check if server is still running
                if not server.is_running():
                    # Set flag to handle failure after exiting Live display
                    server_failed = True
                    break

                time.sleep(0.5)

    except KeyboardInterrupt:
        pass

    # Handle server failure after Live display has ended
    if server_failed:
        console.print(f"\n[red]Model engine '{model_name}' has stopped.[/red]")

        # Show last few log lines to help diagnose
        recent_logs = server.get_recent_logs(10)
        if recent_logs:
            console.print("\n[bold]Last logs before failure:[/bold]")
            for log in recent_logs[-5:]:  # Show last 5 lines
                console.print(f"  {log}")

        # Offer to view full logs
        view_logs = input(f"\nView full logs for {model_name}? (y/N): ").strip().lower()
        if view_logs in ["y", "yes"]:
            from ..log_viewer import show_log_menu

            show_log_menu(server)

    # Display operation-specific completion summary
    if operation_state.get("completed"):
        if operation_state["type"] == "sleeping":
            console.print(
                "\n[bold green]✓ Sleep operation completed successfully![/bold green]"
            )
            console.print(
                f"[cyan]Time taken: {operation_state['completion_time']:.2f} seconds[/cyan]"
            )
            if operation_state.get("memory_freed"):
                console.print(
                    f"[yellow]Memory freed: {operation_state['memory_freed']:.2f} GiB[/yellow]"
                )
                console.print(
                    f"[dim]Memory still in use: {operation_state['memory_remaining']:.2f} GiB[/dim]"
                )
            console.print(f"\n[green]Model '{model_name}' is now in sleep mode[/green]")
        elif operation_state["type"] == "waking":
            console.print(
                "\n[bold green]✓ Wake operation completed successfully![/bold green]"
            )
            console.print(
                f"[cyan]Time taken: {operation_state['completion_time']:.2f} seconds[/cyan]"
            )
            console.print(
                f"\n[green]Model '{model_name}' is now fully operational[/green]"
            )

        # Auto-return to menu after brief pause
        console.print("\n[dim]Returning to proxy management menu in 3 seconds...[/dim]")
        time.sleep(3)
        return "back"
    else:
        # Manual exit via Ctrl+C
        console.print("\n[yellow]Monitoring stopped.[/yellow]")
        console.print(f"[green]✓ Model '{model_name}' continues running[/green]")
        console.print("[dim]Press Ctrl+C to stop proxy servers[/dim]")

        try:
            input("\nPress Enter to return to monitoring menu...")
            return "back"
        except KeyboardInterrupt:
            # User wants to stop proxy
            if (
                unified_prompt(
                    "confirm_stop_model",
                    "Stop all proxy servers?",
                    ["Yes, stop all servers", "No, keep running"],
                    allow_back=False,
                )
                == "Yes, stop all servers"
            ):
                return "stop"
            return "back"


def show_proxy_status(proxy_manager: "ProxyManager"):
    """
    Display current status of proxy and all models.
    """
    console.print("\n[bold cyan]Proxy Server Status[/bold cyan]")

    # Proxy status
    proxy_table = Table(show_header=False, box=None)
    proxy_table.add_column("Property", style="cyan")
    proxy_table.add_column("Value", style="magenta")

    proxy_table.add_row(
        "Proxy Server",
        (
            "[green]Running[/green]"
            if proxy_manager.proxy_process and proxy_manager.proxy_process.is_running()
            else "[red]Not Running[/red]"
        ),
    )
    proxy_table.add_row("Host", proxy_manager.proxy_config.host)
    proxy_table.add_row("Port", str(proxy_manager.proxy_config.port))
    proxy_table.add_row(
        "CORS", "Enabled" if proxy_manager.proxy_config.enable_cors else "Disabled"
    )
    proxy_table.add_row(
        "Metrics",
        "Enabled" if proxy_manager.proxy_config.enable_metrics else "Disabled",
    )

    console.print(create_panel(proxy_table, title="Proxy Configuration"))

    # Models status
    if proxy_manager.vllm_servers:
        console.print("\n[bold]Model Engines:[/bold]")

        models_table = Table()
        models_table.add_column("Model", style="cyan")
        models_table.add_column("Port", style="magenta")
        models_table.add_column("Status", style="green")
        models_table.add_column("GPU(s)", style="yellow")
        models_table.add_column("Profile", style="blue")

        for model_name, server in proxy_manager.vllm_servers.items():
            model_config = proxy_manager._get_model_config_by_name(model_name)

            gpu_str = "N/A"
            profile_str = "N/A"

            if model_config:
                if model_config.gpu_ids:
                    gpu_str = ",".join(str(g) for g in model_config.gpu_ids)
                if model_config.profile:
                    profile_str = model_config.profile

            models_table.add_row(
                model_name,
                str(server.port),
                (
                    "[green]Running[/green]"
                    if server.is_running()
                    else "[red]Stopped[/red]"
                ),
                gpu_str,
                profile_str,
            )

        console.print(models_table)
    else:
        console.print("\n[yellow]No models currently running.[/yellow]")


def monitor_proxy_logs(proxy_manager: "ProxyManager") -> str:
    """
    Monitor proxy server logs and statistics.

    Shows proxy server logs, request statistics, and routing info.

    Args:
        proxy_manager: The ProxyManager instance

    Returns:
        Navigation command string
    """
    console.print("[bold cyan]Proxy Server Logs & Statistics[/bold cyan]")
    console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")

    # Get UI preferences
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()
    show_gpu = ui_prefs.get("show_gpu_in_monitor", True)
    monitor_refresh_rate = ui_prefs.get("monitor_refresh_rate", 1.0)

    try:
        # Get GPU info for panel sizing
        gpu_info = get_gpu_info() if show_gpu else None
        gpu_panel_size = (
            calculate_gpu_panel_size(len(gpu_info) if gpu_info else 0)
            if show_gpu
            else 0
        )

        # Calculate dynamic size for backends based on number of models
        model_count = (
            len(proxy_manager.vllm_servers) if proxy_manager.vllm_servers else 1
        )
        # Size: 6 overhead (borders, title, header, separator, padding) + model rows
        # Minimum 7 lines for display, maximum 15 to prevent tall panels
        backends_size = max(7, min(6 + model_count, 15))

        # Create layout
        layout = Layout()
        if show_gpu:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="proxy_info", size=6),
                Layout(name="backends", size=backends_size),
                Layout(name="gpu", size=gpu_panel_size),
                Layout(name="logs"),  # Takes remaining space
                Layout(name="footer", size=1),
            )
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="proxy_info", size=6),
                Layout(
                    name="backends", size=backends_size
                ),  # Same size, calculation already includes overhead
                Layout(name="logs"),  # Takes remaining space
                Layout(name="footer", size=1),
            )

        # Header
        header_text = Text(
            "Proxy Server Logs & Statistics", style="bold cyan", justify="center"
        )
        layout["header"].update(Padding(header_text, (1, 0)))

        # Footer
        layout["footer"].update(
            Align.center(Text("Press Ctrl+C to stop monitoring", style="dim cyan"))
        )

        # Smart registry refresh for pending models
        registry_refresh_interval = 5.0  # Start with 5 seconds
        max_refresh_interval = 30.0  # Max 30 seconds between refreshes
        last_registry_update = time.time()
        pending_model_count = 0
        refresh_attempts = 0
        max_refresh_attempts = 10  # Stop trying after 10 attempts

        # Initial registry fetch
        try:
            initial_registry_status = proxy_manager.get_proxy_registry_status()
            if initial_registry_status:
                # Count pending models
                models = initial_registry_status.get("models", [])
                pending_model_count = sum(
                    1 for m in models if m.get("registration_status") == "pending"
                )
                cached_registry_table = create_model_registry_table(
                    initial_registry_status, proxy_manager, include_title=False
                )

                if pending_model_count > 0:
                    logger.debug(
                        f"Found {pending_model_count} pending models, will refresh registry"
                    )
            else:
                cached_registry_table = Table(box=box.ROUNDED)
                cached_registry_table.add_column("Status", style="yellow")
                cached_registry_table.add_row("Registry status unavailable")
        except Exception as e:
            logger.warning(f"Error fetching initial registry: {e}")
            cached_registry_table = Table(box=box.ROUNDED)
            cached_registry_table.add_column("Error", style="red")
            cached_registry_table.add_row(f"Error: {str(e)[:50]}")

        with Live(layout, console=console, refresh_per_second=monitor_refresh_rate):
            while True:
                # Proxy server information
                if (
                    proxy_manager.proxy_process
                    and proxy_manager.proxy_process.is_running()
                ):
                    proxy_table = Table(show_header=False, box=None)
                    proxy_table.add_column("Key", style="cyan")
                    proxy_table.add_column("Value", style="magenta")

                    proxy_table.add_row(
                        "Status",
                        (
                            "[green]Running[/green]"
                            if proxy_manager.proxy_process
                            and proxy_manager.proxy_process.is_running()
                            else "[red]Stopped[/red]"
                        ),
                    )
                    proxy_table.add_row("Host", proxy_manager.proxy_config.host)
                    proxy_table.add_row("Port", str(proxy_manager.proxy_config.port))
                    proxy_table.add_row(
                        "CORS",
                        (
                            "[green]Enabled[/green]"
                            if proxy_manager.proxy_config.enable_cors
                            else "[yellow]Disabled[/yellow]"
                        ),
                    )
                    proxy_table.add_row(
                        "Metrics",
                        (
                            "[green]Enabled[/green]"
                            if proxy_manager.proxy_config.enable_metrics
                            else "[yellow]Disabled[/yellow]"
                        ),
                    )

                    # Calculate uptime if proxy process has start_time
                    if proxy_manager.proxy_process and hasattr(
                        proxy_manager.proxy_process, "start_time"
                    ):
                        uptime = (
                            time.time()
                            - proxy_manager.proxy_process.start_time.timestamp()
                        )
                        hours, remainder = divmod(int(uptime), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        proxy_table.add_row(
                            "Uptime", f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        )

                    # Request statistics are available via /proxy/status endpoint
                    # but not needed for basic monitoring display
                else:
                    proxy_table = Table(show_header=False, box=None)
                    proxy_table.add_column("", style="red")
                    proxy_table.add_row("Proxy server not running")

                layout["proxy_info"].update(
                    create_panel(
                        proxy_table,
                        title="Proxy Server Status",
                        border_style=(
                            "green"
                            if proxy_manager.proxy_process
                            and proxy_manager.proxy_process.is_running()
                            else "red"
                        ),
                    )
                )

                # Smart refresh: Only update registry if there are pending models
                current_time = time.time()
                should_refresh = False

                if pending_model_count > 0 and refresh_attempts < max_refresh_attempts:
                    # Check if enough time has passed for next refresh
                    if (
                        current_time - last_registry_update
                    ) >= registry_refresh_interval:
                        should_refresh = True
                if should_refresh:
                    try:
                        # Fetch updated registry
                        updated_registry = proxy_manager.get_proxy_registry_status()
                        if updated_registry:
                            # Count pending models in update
                            models = updated_registry.get("models", [])
                            new_pending_count = sum(
                                1
                                for m in models
                                if m.get("registration_status") == "pending"
                            )
                            # Update the table
                            cached_registry_table = create_model_registry_table(
                                updated_registry, proxy_manager, include_title=False
                            )

                            # Check if we made progress
                            if new_pending_count < pending_model_count:
                                # Progress made, reset interval
                                registry_refresh_interval = 5.0
                                refresh_attempts = 0
                                logger.debug(
                                    f"Registry refresh: {pending_model_count} -> {new_pending_count} pending"
                                )
                            else:
                                # No progress, increase interval (exponential backoff)
                                registry_refresh_interval = min(
                                    registry_refresh_interval * 1.5,
                                    max_refresh_interval,
                                )
                                refresh_attempts += 1

                            pending_model_count = new_pending_count
                            last_registry_update = current_time

                            # Stop refreshing if all models are registered
                            if pending_model_count == 0:
                                logger.debug(
                                    "All models registered, stopping registry refresh"
                                )

                    except Exception as e:
                        logger.warning(f"Error refreshing registry: {e}")
                        # On error, increase interval
                        registry_refresh_interval = min(
                            registry_refresh_interval * 2, max_refresh_interval
                        )
                        refresh_attempts += 1

                # Use the cached (possibly updated) registry table
                layout["backends"].update(
                    create_panel(
                        cached_registry_table,
                        title="Current Model Registry",
                        border_style="blue",
                    )
                )

                # GPU panel if enabled
                if show_gpu:
                    gpu_panel = create_gpu_status_panel()
                    layout["gpu"].update(gpu_panel)

                # Display proxy server logs dynamically
                # Calculate available height for logs
                current_height = console.height
                # Calculate fixed elements
                fixed_height = (
                    3 + 6 + backends_size + 1
                )  # header + proxy_info + backends + footer
                if show_gpu:
                    fixed_height += gpu_panel_size

                # Available lines for logs
                available_lines = max(20, current_height - fixed_height - 2)

                # Get recent logs from proxy process if available
                recent_logs = []
                if proxy_manager.proxy_process and hasattr(
                    proxy_manager.proxy_process, "get_recent_logs"
                ):
                    recent_logs = proxy_manager.proxy_process.get_recent_logs(
                        available_lines
                    )

                if recent_logs:
                    log_text = Text("\n".join(recent_logs), style="dim white")
                else:
                    log_text = Text(
                        "Waiting for proxy server logs...\n", style="dim yellow"
                    )
                    log_text.append(
                        "\nNote: Request logs will appear here when the proxy "
                        "receives requests.",
                        style="dim",
                    )

                logs_content = Group(
                    Rule("Proxy Server Logs", style="yellow"), Padding(log_text, (1, 2))
                )
                layout["logs"].update(logs_content)

                time.sleep(0.5)

    except KeyboardInterrupt:
        pass

    console.print("\n[yellow]Monitoring stopped.[/yellow]")
    console.print("[dim]Press Ctrl+C to stop proxy servers[/dim]")

    try:
        input("\nPress Enter to return to monitoring menu...")
        return "back"
    except KeyboardInterrupt:
        # User wants to stop proxy
        if (
            unified_prompt(
                "confirm_stop_proxy",
                "Stop all proxy servers?",
                ["Yes, stop all servers", "No, keep running"],
                allow_back=False,
            )
            == "Yes, stop all servers"
        ):
            return "stop"
        return "back"
