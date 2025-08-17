#!/usr/bin/env python3
"""
Settings module for vLLM CLI.

Handles configuration settings and application preferences.
"""
import logging
from rich.table import Table
from rich.panel import Panel

from ..config import ConfigManager
from ..models import list_available_models
from .navigation import unified_prompt
from .common import console, create_panel
from .profiles import manage_profiles
from ..ui.progress_styles import (
    PROGRESS_STYLES,
    get_progress_bar,
    list_available_styles,
)

logger = logging.getLogger(__name__)


def handle_settings() -> str:
    """
    Handle settings and configuration.
    """
    while True:
        settings_options = [
            "Manage Profiles",
            "Model Directories",
            "Server Defaults",
            "UI Preferences",
            "Clear Cache",
        ]

        action = unified_prompt("settings", "Settings", settings_options, allow_back=True)

        if action == "← Back" or action == "BACK" or not action:
            return "continue"
        elif action == "Manage Profiles":
            manage_profiles()
        elif action == "Model Directories":
            manage_model_directories()
        elif action == "Server Defaults":
            configure_server_defaults()
        elif action == "UI Preferences":
            configure_ui_preferences()
        elif action == "Clear Cache":
            config_manager = ConfigManager()
            config_manager.clear_cache()
            console.print("[green]Cache cleared.[/green]")
            input("\nPress Enter to continue...")

    return "continue"


def manage_model_directories() -> str:
    """
    Manage model directories using integrated hf-model-tool API.
    
    This function uses the hf-model-tool API directly to provide
    a seamless directory management experience within vLLM CLI.
    """
    from .model_directories import manage_model_directories as manage_dirs
    return manage_dirs()


def configure_server_defaults() -> str:
    """
    Configure default server settings.
    """
    config_manager = ConfigManager()
    defaults = config_manager.get_server_defaults()

    console.print("\n[bold cyan]Server Defaults[/bold cyan]")
    console.print("Configure default settings for all servers:")

    # Edit defaults
    defaults["default_port"] = int(
        input(f"Default port [{defaults.get('default_port', 8000)}]: ").strip()
        or defaults.get("default_port", 8000)
    )
    defaults["auto_restart"] = input(
        f"Auto-restart on failure (yes/no) [{defaults.get('auto_restart', False)}]: "
    ).strip().lower() in ["yes", "true", "1"]
    defaults["log_level"] = input(
        f"Log level (info/debug/warning/error) [{defaults.get('log_level', 'info')}]: "
    ).strip() or defaults.get("log_level", "info")

    config_manager.save_server_defaults(defaults)
    console.print("[green]Server defaults updated.[/green]")
    input("\nPress Enter to continue...")

    return "continue"


def configure_ui_preferences() -> str:
    """
    Configure UI preferences including progress bar style.
    """
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()

    console.print("\n[bold cyan]UI Preferences[/bold cyan]")

    # Show current settings
    current_style = ui_prefs.get("progress_bar_style", "blocks")
    console.print(f"\nCurrent progress bar style: [yellow]{current_style}[/yellow]")

    # Create preview table
    preview_table = Table(
        title="[bold]Progress Bar Style Preview[/bold]",
        show_header=True,
        header_style="bold cyan",
    )
    preview_table.add_column("#", style="cyan", width=3)
    preview_table.add_column("Style", style="yellow", width=12)
    preview_table.add_column("25%", style="white")
    preview_table.add_column("50%", style="white")
    preview_table.add_column("75%", style="white")
    preview_table.add_column("100%", style="white")

    styles = list_available_styles()
    for i, style_name in enumerate(styles, 1):
        style_obj = PROGRESS_STYLES[style_name]
        preview_table.add_row(
            str(i),
            style_name,
            get_progress_bar(25, style_name, 10),
            get_progress_bar(50, style_name, 10),
            get_progress_bar(75, style_name, 10),
            get_progress_bar(100, style_name, 10),
        )

    console.print(preview_table)

    # Select new style
    console.print("\nSelect a progress bar style:")
    for i, style in enumerate(styles, 1):
        console.print(f"  {i}. {style}")

    choice = input(
        f"\nEnter choice (1-{len(styles)}) [{styles.index(current_style) + 1}]: "
    ).strip()

    if choice.isdigit() and 1 <= int(choice) <= len(styles):
        new_style = styles[int(choice) - 1]
        ui_prefs["progress_bar_style"] = new_style
        console.print(f"\n[green]Progress bar style set to: {new_style}[/green]")
    else:
        console.print("[yellow]No change made to progress bar style[/yellow]")

    # Configure GPU monitoring
    console.print("\n[bold]GPU Monitoring Settings[/bold]")
    show_gpu = ui_prefs.get("show_gpu_in_monitor", True)
    gpu_choice = (
        input(
            f"Show GPU panel in server monitor? (yes/no) [{'yes' if show_gpu else 'no'}]: "
        )
        .strip()
        .lower()
    )

    if gpu_choice in ["yes", "y", "true", "1"]:
        ui_prefs["show_gpu_in_monitor"] = True
        console.print("[green]GPU panel will be shown in server monitor[/green]")
    elif gpu_choice in ["no", "n", "false", "0"]:
        ui_prefs["show_gpu_in_monitor"] = False
        console.print("[yellow]GPU panel will be hidden in server monitor[/yellow]")

    # Configure log display settings
    console.print("\n[bold]Log Display Settings[/bold]")

    # Startup log lines
    current_startup_lines = ui_prefs.get("log_lines_startup", 50)
    console.print(
        f"Current startup log lines: [yellow]{current_startup_lines}[/yellow]"
    )
    startup_choice = input(
        f"Number of log lines during startup (5-50) [{current_startup_lines}]: "
    ).strip()

    if startup_choice.isdigit() and 5 <= int(startup_choice) <= 50:
        ui_prefs["log_lines_startup"] = int(startup_choice)
        console.print(f"[green]Startup log lines set to: {startup_choice}[/green]")
    elif startup_choice:
        console.print("[yellow]Invalid input. Startup log lines unchanged.[/yellow]")

    # Monitor log lines
    current_monitor_lines = ui_prefs.get("log_lines_monitor", 50)
    console.print(
        f"Current monitor log lines: [yellow]{current_monitor_lines}[/yellow]"
    )
    monitor_choice = input(
        f"Number of log lines in server monitor (10-100) [{current_monitor_lines}]: "
    ).strip()

    if monitor_choice.isdigit() and 10 <= int(monitor_choice) <= 100:
        ui_prefs["log_lines_monitor"] = int(monitor_choice)
        console.print(f"[green]Monitor log lines set to: {monitor_choice}[/green]")
    elif monitor_choice:
        console.print("[yellow]Invalid input. Monitor log lines unchanged.[/yellow]")

    # Configure refresh rates
    console.print("\n[bold]Log Refresh Rate Settings[/bold]")
    console.print("[dim]Higher refresh rates provide more responsive logs but use more CPU[/dim]")

    # Startup refresh rate
    current_startup_rate = ui_prefs.get("startup_refresh_rate", 4.0)
    console.print(
        f"Current startup refresh rate: [yellow]{current_startup_rate} Hz[/yellow]"
    )
    startup_rate_choice = input(
        f"Startup log refresh rate (1-10 Hz) [{current_startup_rate}]: "
    ).strip()

    if startup_rate_choice:
        try:
            rate = float(startup_rate_choice)
            if 1.0 <= rate <= 10.0:
                ui_prefs["startup_refresh_rate"] = rate
                console.print(f"[green]Startup refresh rate set to: {rate} Hz[/green]")
            else:
                console.print("[yellow]Invalid range. Startup refresh rate unchanged.[/yellow]")
        except ValueError:
            console.print("[yellow]Invalid input. Startup refresh rate unchanged.[/yellow]")

    # Monitor refresh rate
    current_monitor_rate = ui_prefs.get("monitor_refresh_rate", 1.0)
    console.print(
        f"Current monitor refresh rate: [yellow]{current_monitor_rate} Hz[/yellow]"
    )
    monitor_rate_choice = input(
        f"Monitor log refresh rate (0.5-5 Hz) [{current_monitor_rate}]: "
    ).strip()

    if monitor_rate_choice:
        try:
            rate = float(monitor_rate_choice)
            if 0.5 <= rate <= 5.0:
                ui_prefs["monitor_refresh_rate"] = rate
                console.print(f"[green]Monitor refresh rate set to: {rate} Hz[/green]")
            else:
                console.print("[yellow]Invalid range. Monitor refresh rate unchanged.[/yellow]")
        except ValueError:
            console.print("[yellow]Invalid input. Monitor refresh rate unchanged.[/yellow]")

    # Save preferences
    config_manager.save_ui_preferences(ui_prefs)
    console.print("\n[green]UI preferences saved.[/green]")
    input("\nPress Enter to continue...")

    return "continue"
