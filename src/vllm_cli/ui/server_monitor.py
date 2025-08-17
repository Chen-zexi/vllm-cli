#!/usr/bin/env python3
"""
Server monitoring module for vLLM CLI.

Handles monitoring of running vLLM servers and displaying their status.
"""
import logging
import time

import inquirer
from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ..config import ConfigManager
from ..server import VLLMServer, get_active_servers
from ..system import get_gpu_info
from .common import console, create_panel
from .gpu_utils import calculate_gpu_panel_size, create_gpu_status_panel
from .log_viewer import show_log_menu
from .navigation import unified_prompt

logger = logging.getLogger(__name__)


def monitor_server(server: VLLMServer) -> str:
    """
    Monitor a running vLLM server.
    """
    # Clear console for clean display
    console.clear()
    console.print("[bold cyan]Monitoring Server[/bold cyan]")
    console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")

    # Get UI preferences
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()
    show_gpu = ui_prefs.get("show_gpu_in_monitor", True)
    monitor_refresh_rate = ui_prefs.get("monitor_refresh_rate", 1.0)

    try:
        # Get GPU info to calculate optimal panel size
        gpu_info = get_gpu_info() if show_gpu else None
        gpu_panel_size = (
            calculate_gpu_panel_size(len(gpu_info) if gpu_info else 0)
            if show_gpu
            else 0
        )

        # Create layout for monitoring
        layout = Layout()
        if show_gpu:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="status", size=5),
                Layout(
                    name="gpu", size=gpu_panel_size
                ),  # Dynamic size based on GPU count
                Layout(name="logs"),  # Takes remaining space
                Layout(name="footer", size=1),
            )
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="status", size=6),
                Layout(name="logs"),
                Layout(name="footer", size=1),
            )

        # Header
        layout["header"].update(
            create_panel(
                f"vLLM Server Monitor - {server.model}",
                title="Monitor",
                border_style="cyan",
            )
        )

        # Footer with exit instructions
        layout["footer"].update(
            Align.center(
                Text(
                    "Press Ctrl+C to stop monitoring • Server continues in background",
                    style="dim cyan",
                )
            )
        )

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
                status_table.add_row("Port", str(server.port))
                status_table.add_row(
                    "PID", str(server.process.pid) if server.process else "N/A"
                )
                status_table.add_row(
                    "Model",
                    (
                        server.model[:40] + "..."
                        if len(server.model) > 40
                        else server.model
                    ),
                )

                # Update status
                layout["status"].update(
                    create_panel(
                        status_table,
                        title="Server Status",
                        border_style="green" if server.is_running() else "red",
                    )
                )

                # Update GPU panel if enabled
                if show_gpu:
                    gpu_panel = create_gpu_status_panel()
                    layout["gpu"].update(gpu_panel)

                # Update logs with integrated divider
                monitor_log_lines = ui_prefs.get("log_lines_monitor", 50)
                recent_logs = server.get_recent_logs(monitor_log_lines)
                if recent_logs:
                    log_text = Text("\n".join(recent_logs), style="dim white")
                    logs_content = Group(
                        Rule("Recent Logs", style="yellow"), Padding(log_text, (1, 2))
                    )
                else:
                    # Check if this is an external server
                    if not hasattr(server.process, "poll"):
                        log_text = Text(
                            "This server was started externally.\n"
                            "Logs are not available through vLLM CLI.\n\n"
                            "To view logs, check the terminal where the server was started,\n"
                            "or look for log files in the vLLM working directory.",
                            style="dim yellow",
                        )
                    else:
                        log_text = Text("Waiting for logs...", style="dim yellow")

                    logs_content = Group(
                        Rule("Recent Logs", style="yellow"), Padding(log_text, (1, 2))
                    )

                layout["logs"].update(logs_content)

                # Check if server is still running
                if not server.is_running():
                    console.print("\n[red]Server has stopped unexpectedly.[/red]")
                    break

                # Small sleep to prevent high CPU usage
                time.sleep(0.5)

    except KeyboardInterrupt:
        # Clean exit via Ctrl+C
        pass  # Exit the Live context cleanly

    # After exiting monitoring (via Ctrl+C)
    console.print("\n[yellow]Monitoring stopped.[/yellow]")
    console.print("\n[green]✓ Server continues running in background[/green]")
    console.print(f"[dim]Server endpoint: http://localhost:{server.port}[/dim]")
    console.print("")

    # Show options with navigation
    options = [
        "Resume monitoring",
        "View server logs",
        "Stop server",
        "Return to main menu",
    ]

    choice = unified_prompt(
        "post_monitor", "What would you like to do?", options, allow_back=False
    )

    if choice == "Resume monitoring":
        return monitor_server(server)
    elif choice == "View server logs":
        show_log_menu(server)
        return monitor_server(server)  # Return to monitoring after viewing logs
    elif choice == "Stop server":
        console.print("[yellow]Stopping server...[/yellow]")
        server.stop()
        console.print("[green]✓ Server stopped.[/green]")
        input("\nPress Enter to continue...")
        return "continue"
    else:
        # Return to main menu
        console.print("[dim]Server will continue running.[/dim]")
        console.print("[dim]You can monitor it again from the main menu.[/dim]")
        # Log server status for debugging
        if server.is_running():
            logger.info(
                f"Server {server.model} on port {server.port} is still running after exiting monitor"
            )
        else:
            logger.warning(
                f"Server {server.model} on port {server.port} appears to have stopped"
            )
        input("\nPress Enter to continue...")
        return "continue"


def monitor_active_servers() -> str:
    """
    Monitor all active vLLM servers.
    """
    servers = get_active_servers()
    if not servers:
        console.print("[yellow]No active servers found.[/yellow]")
        input("\nPress Enter to continue...")
        return "continue"

    # Show active servers
    table = Table(
        title="[bold green]Active vLLM Servers[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Model", style="magenta")
    table.add_column("Port", style="yellow")
    table.add_column("PID", style="green")
    table.add_column("Status", style="white")

    for i, server in enumerate(servers, 1):
        status = (
            "[green]Running[/green]" if server.is_running() else "[red]Stopped[/red]"
        )
        table.add_row(
            str(i),
            server.model,
            str(server.port),
            str(server.process.pid) if server.process else "N/A",
            status,
        )

    console.print(table)

    # Select server to monitor or manage
    choices = [f"Monitor Server {i}" for i in range(1, len(servers) + 1)]
    choices.extend([f"View Logs {i}" for i in range(1, len(servers) + 1)])
    choices.append("Stop All Servers")

    action = unified_prompt(
        "server_action", "Server Management", choices, allow_back=True
    )

    if action == "BACK" or not action:
        return "continue"
    elif action == "Stop All Servers":
        confirm = inquirer.confirm(f"Stop all {len(servers)} servers?", default=False)
        if confirm:
            for server in servers:
                server.stop()
            console.print("[green]All servers stopped.[/green]")
            input("\nPress Enter to continue...")
    elif action.startswith("Monitor Server"):
        server_idx = int(action.split()[-1]) - 1
        return monitor_server(servers[server_idx])
    elif action.startswith("View Logs"):
        server_idx = int(action.split()[-1]) - 1
        show_log_menu(servers[server_idx])
        return "continue"

    return "continue"
