#!/usr/bin/env python3
"""
Monitoring module for the proxy server and individual model engines.
"""
import logging
import time

from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ..config import ConfigManager
from ..server import VLLMServer
from ..system import get_gpu_info
from ..ui.common import console, create_panel
from ..ui.gpu_utils import calculate_gpu_panel_size, create_gpu_status_panel
from ..ui.navigation import unified_prompt
from .manager import ProxyManager

logger = logging.getLogger(__name__)


def monitor_model_logs_menu(proxy_manager: ProxyManager) -> str:
    """
    Submenu for selecting model monitoring mode.

    Args:
        proxy_manager: The ProxyManager instance

    Returns:
        Navigation command string ('back' or 'stop')
    """
    from ..ui.common import console

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


def monitor_proxy(proxy_manager: ProxyManager) -> str:
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


def monitor_proxy_overview(proxy_manager: ProxyManager) -> str:
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

        # Create layout
        layout = Layout()
        if show_gpu:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="proxy_status", size=4),
                Layout(name="models", size=10),
                Layout(name="gpu", size=gpu_panel_size),
                Layout(name="logs"),
                Layout(name="footer", size=1),
            )
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="proxy_status", size=4),
                Layout(name="models", size=10),
                Layout(name="logs"),
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
                # Update proxy status
                proxy_status = Table(show_header=False, box=None)
                proxy_status.add_column("Key", style="cyan")
                proxy_status.add_column("Value", style="magenta")

                proxy_status.add_row(
                    "Proxy Status",
                    (
                        "[green]Running[/green]"
                        if proxy_manager.proxy_server
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
                        border_style="green" if proxy_manager.proxy_server else "red",
                    )
                )

                # Update models table
                models_table = Table(title="Model Engines")
                models_table.add_column("#", style="dim", width=3)
                models_table.add_column("Model", style="cyan")
                models_table.add_column("Port", style="magenta")
                models_table.add_column("Status", style="green")
                models_table.add_column("GPU(s)", style="yellow")
                models_table.add_column("Uptime", style="dim")

                for idx, (model_name, server) in enumerate(
                    proxy_manager.vllm_servers.items(), 1
                ):
                    # Get model config for GPU info
                    model_config = proxy_manager._get_model_config_by_name(model_name)
                    gpu_str = (
                        ",".join(str(g) for g in model_config.gpu_ids)
                        if model_config
                        else "?"
                    )

                    # Calculate uptime
                    uptime_str = "N/A"
                    if server.start_time:
                        uptime = time.time() - server.start_time.timestamp()
                        hours, remainder = divmod(int(uptime), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                    models_table.add_row(
                        str(idx),
                        model_name[:20] + "..." if len(model_name) > 20 else model_name,
                        str(server.port),
                        (
                            "[green]Running[/green]"
                            if server.is_running()
                            else "[red]Stopped[/red]"
                        ),
                        gpu_str,
                        uptime_str,
                    )

                layout["models"].update(
                    create_panel(
                        models_table,
                        title="Model Status",
                        border_style="blue",
                    )
                )

                # Update GPU panel if enabled
                if show_gpu:
                    gpu_panel = create_gpu_status_panel()
                    layout["gpu"].update(gpu_panel)

                # Aggregate recent logs from all models
                all_logs = []
                for model_name, server in proxy_manager.vllm_servers.items():
                    recent = server.get_recent_logs(5)  # Get last 5 from each
                    if recent:
                        all_logs.append(f"[{model_name}]")
                        all_logs.extend(recent)

                if all_logs:
                    log_text = Text(
                        "\n".join(all_logs[-20:]), style="dim white"
                    )  # Show last 20 lines total
                else:
                    log_text = Text("Waiting for logs...", style="dim yellow")

                logs_content = Group(
                    Rule("Recent Logs (All Models)", style="yellow"),
                    Padding(log_text, (1, 2)),
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


def monitor_individual_model(proxy_manager: ProxyManager) -> str:
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
    server: VLLMServer, model_name: str, proxy_manager: ProxyManager
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

                # Update logs
                monitor_log_lines = ui_prefs.get("log_lines_monitor", 50)
                recent_logs = server.get_recent_logs(monitor_log_lines)

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


def show_proxy_status(proxy_manager: ProxyManager):
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
            if proxy_manager.proxy_server
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


def monitor_proxy_logs(proxy_manager: ProxyManager) -> str:
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
        # Size calculation: 6 overhead (2 panel borders, 1 title, 1 table header, 1 separator, 1 padding) + model rows
        # Minimum 7 lines to ensure proper display, maximum 15 to prevent too tall panels
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
                if proxy_manager.proxy_server:
                    proxy_table = Table(show_header=False, box=None)
                    proxy_table.add_column("Key", style="cyan")
                    proxy_table.add_column("Value", style="magenta")

                    proxy_table.add_row(
                        "Status",
                        (
                            "[green]Running[/green]"
                            if proxy_manager.proxy_thread
                            and proxy_manager.proxy_thread.is_alive()
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

                    # Calculate uptime if proxy server has start_time
                    if hasattr(proxy_manager.proxy_server, "start_time"):
                        uptime = (
                            time.time()
                            - proxy_manager.proxy_server.start_time.timestamp()
                        )
                        hours, remainder = divmod(int(uptime), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        proxy_table.add_row(
                            "Uptime", f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        )

                    # Show request statistics if available
                    if hasattr(proxy_manager.proxy_server, "total_requests"):
                        proxy_table.add_row(
                            "Total Requests",
                            str(proxy_manager.proxy_server.total_requests),
                        )
                else:
                    proxy_table = Table(show_header=False, box=None)
                    proxy_table.add_column("", style="red")
                    proxy_table.add_row("Proxy server not running")

                layout["proxy_info"].update(
                    create_panel(
                        proxy_table,
                        title="Proxy Server Status",
                        border_style="green" if proxy_manager.proxy_server else "red",
                    )
                )

                # Backend servers table
                backends_table = Table()
                backends_table.add_column("Model", style="cyan")
                backends_table.add_column("URL", style="magenta")
                backends_table.add_column("Status", style="green")
                backends_table.add_column("Requests", style="yellow")

                if proxy_manager.proxy_server and proxy_manager.proxy_server.router:
                    try:
                        router = proxy_manager.proxy_server.router
                        for model_name, backend_config in router.backends.items():
                            # Extract URL from backend config
                            url = (
                                backend_config.get("url", "N/A")
                                if isinstance(backend_config, dict)
                                else str(backend_config)
                            )

                            # Get server status
                            server = proxy_manager.vllm_servers.get(model_name)
                            status = (
                                "[green]Running[/green]"
                                if server and server.is_running()
                                else "[red]Stopped[/red]"
                            )

                            # Get request count if available
                            request_count = (
                                proxy_manager.proxy_server.model_requests.get(
                                    model_name, 0
                                )
                            )

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

                # Display proxy server logs
                monitor_log_lines = ui_prefs.get("log_lines_monitor", 50)

                # Get recent logs from proxy server if available
                recent_logs = []
                if proxy_manager.proxy_server and hasattr(
                    proxy_manager.proxy_server, "get_recent_logs"
                ):
                    recent_logs = proxy_manager.proxy_server.get_recent_logs(
                        monitor_log_lines
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
