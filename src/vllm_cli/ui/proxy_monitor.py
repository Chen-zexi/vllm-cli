#!/usr/bin/env python3
"""
Proxy monitoring UI module for vLLM CLI.

Handles UI display for monitoring proxy servers and model engines.
"""
import logging
import threading
import time
from threading import Thread
from typing import TYPE_CHECKING

from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ..config import ConfigManager
from ..system import get_gpu_info
from .common import console, create_panel
from .gpu_utils import calculate_gpu_panel_size, create_gpu_status_panel
from .navigation import unified_prompt

if TYPE_CHECKING:
    from ..proxy.manager import ProxyManager
    from ..server import VLLMServer

logger = logging.getLogger(__name__)

# Color palette for distinct model names
MODEL_COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red"]


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

    if fail_count > 0:
        console.print(f"\n[red]✗ {fail_count} model(s) failed to start[/red]")
        return False
    else:
        console.print(
            f"\n[green]✓ All {success_count} model(s) started successfully[/green]"
        )
        return True


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

        # Footer with model list
        other_models = [m for m in proxy_manager.vllm_servers.keys() if m != model_name]
        footer_text = "Press Ctrl+C for menu options"
        if other_models:
            footer_text += f" • Other models: {', '.join(other_models[:3])}"
            if len(other_models) > 3:
                footer_text += f" (+{len(other_models)-3} more)"

        layout["footer"].update(Align.center(Text(footer_text, style="dim cyan")))

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

                # Available lines for logs
                available_lines = max(20, current_height - fixed_height - 2)
                recent_logs = server.get_recent_logs(available_lines)

                if recent_logs:
                    log_text = Text("\n".join(recent_logs), style="dim white")
                else:
                    log_text = Text("Waiting for logs...", style="dim yellow")

                logs_content = Group(
                    Rule(f"Engine Logs - {model_name}", style="yellow"),
                    Padding(log_text, (1, 2)),
                )
                layout["logs"].update(logs_content)

                # Check if server is still running
                if not server.is_running():
                    console.print(
                        f"\n[red]Model engine '{model_name}' has stopped.[/red]"
                    )
                    break

                time.sleep(0.5)

    except KeyboardInterrupt:
        pass

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

                # Backend servers table
                backends_table = Table()
                backends_table.add_column("Model", style="cyan")
                backends_table.add_column("URL", style="magenta")
                backends_table.add_column("Status", style="green")
                backends_table.add_column("Requests", style="yellow")

                # Display backend information from vllm_servers
                if proxy_manager.vllm_servers:
                    try:
                        for model_name, server in proxy_manager.vllm_servers.items():
                            # Build URL from server info
                            url = f"http://localhost:{server.port}"

                            # Server is already from the loop
                            status = (
                                "[green]Running[/green]"
                                if server and server.is_running()
                                else "[red]Stopped[/red]"
                            )

                            # Request count available via /proxy/status
                            request_count = "N/A"

                            backends_table.add_row(
                                (
                                    model_name[:30] + "..."
                                    if len(model_name) > 30
                                    else model_name
                                ),
                                url,
                                status,
                                str(request_count),
                            )
                    except Exception as e:
                        logger.warning(f"Error displaying backends: {e}")
                        backends_table.add_row(
                            "Error loading backends", str(e)[:30], "", ""
                        )
                else:
                    backends_table.add_row("No backends registered", "", "", "")

                layout["backends"].update(
                    create_panel(
                        backends_table, title="Registered Backends", border_style="blue"
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
