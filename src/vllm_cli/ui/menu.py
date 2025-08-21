#!/usr/bin/env python3
"""
Main menu module for vLLM CLI.

Handles main menu display and navigation routing.
"""
import logging

from ..server import get_active_servers
from .navigation import unified_prompt

logger = logging.getLogger(__name__)

# Module-level variables to track active proxy
_active_proxy_manager = None
_active_proxy_config = None


def manage_running_proxy(proxy_manager, proxy_config) -> None:
    """
    Simplified proxy management interface focusing on monitoring.

    Args:
        proxy_manager: The ProxyManager instance
        proxy_config: The proxy configuration
    """
    import time

    from ..proxy.monitor import monitor_model_logs_menu, monitor_proxy_logs
    from .common import console

    # Track the active proxy globally
    global _active_proxy_manager, _active_proxy_config
    _active_proxy_manager = proxy_manager
    _active_proxy_config = proxy_config

    time.sleep(2)

    while True:
        console.print("\n[bold cyan]Proxy Server Running[/bold cyan]")
        console.print(f"Access at: http://{proxy_config.host}:{proxy_config.port}")
        console.print("[dim]Use Ctrl+C in monitoring views to return here[/dim]\n")

        management_options = [
            "Monitor proxy logs",
            "Monitor model logs",
            "Stop all servers",
        ]

        mgmt_choice = unified_prompt(
            "proxy_monitoring",
            "Select action",
            management_options,
            allow_back=True,
        )

        if mgmt_choice == "BACK":
            # Exit to main menu, proxy continues running
            console.print("\n[green]✓ Proxy continues running in background[/green]")
            console.print(f"Access at: http://{proxy_config.host}:{proxy_config.port}")
            console.print(
                "[dim]Return to Multi-Model Proxy menu to manage it later[/dim]"
            )
            time.sleep(2)
            break

        if mgmt_choice == "Monitor proxy logs":
            result = monitor_proxy_logs(proxy_manager)
            if result == "stop":
                # User requested to stop proxy
                console.print(
                    "\n[yellow]Stopping proxy server and all models...[/yellow]"
                )
                proxy_manager.stop_proxy()
                console.print("[green]✓ All servers stopped[/green]")

                # Clear the global tracking variables
                _active_proxy_manager = None
                _active_proxy_config = None
                time.sleep(1)
                break
        elif mgmt_choice == "Monitor model logs":
            result = monitor_model_logs_menu(proxy_manager)
            if result == "stop":
                # User requested to stop proxy
                console.print(
                    "\n[yellow]Stopping proxy server and all models...[/yellow]"
                )
                proxy_manager.stop_proxy()
                console.print("[green]✓ All servers stopped[/green]")

                # Clear the global tracking variables
                _active_proxy_manager = None
                _active_proxy_config = None
                time.sleep(1)
                break
        elif mgmt_choice == "Stop all servers":
            # Confirm before stopping
            if (
                unified_prompt(
                    "confirm_stop_servers",
                    "Stop all proxy servers?",
                    ["Yes, stop all servers", "No, keep running"],
                    allow_back=False,
                )
                == "Yes, stop all servers"
            ):
                console.print(
                    "\n[yellow]Stopping proxy server and all models...[/yellow]"
                )
                proxy_manager.stop_proxy()
                console.print("[green]✓ All servers stopped[/green]")

                # Clear the global tracking variables
                _active_proxy_manager = None
                _active_proxy_config = None

                time.sleep(1)
                break
            # If user cancels, continue to show menu


def handle_multi_model_proxy() -> str:
    """
    Handle multi-model proxy configuration and management.
    """
    from ..proxy import ProxyManager
    from ..proxy.config import ProxyConfigManager
    from .common import console
    from .navigation import unified_prompt
    from .proxy_control import configure_proxy_interactively

    console.print("\n[bold cyan]Multi-Model Proxy Server[/bold cyan]")

    # Check if there are saved configurations
    config_manager = ProxyConfigManager()
    saved_configs = config_manager.list_saved_configs()

    options = []

    # Add saved config option if configs exist
    if saved_configs:
        options.append("Start from saved configuration")

    # Always show configure new proxy option
    options.append("Configure new proxy")

    choice = unified_prompt("proxy_menu", "Select action", options, allow_back=True)

    if choice == "BACK":
        return "continue"

    if choice == "Start from saved configuration":
        # Quick start from saved configuration
        if len(saved_configs) == 1:
            # Only one config, start it directly
            config_name = list(saved_configs.keys())[0]
            console.print(
                f"\n[cyan]Starting saved configuration: '{config_name}'[/cyan]"
            )
            proxy_config = config_manager.load_named_config(config_name)
        else:
            # Multiple configs, let user choose
            console.print("\n[bold]Select configuration to start:[/bold]")
            config_choices = []
            for name, info in saved_configs.items():
                models_str = (
                    f"{info['models']} model{'s' if info['models'] != 1 else ''}"
                )
                preview = ", ".join(info["model_names"][:2])
                if len(info["model_names"]) > 2:
                    preview += ", ..."
                config_choices.append(f"{name} ({models_str}: {preview})")
            config_choices.append("Cancel")

            selected = unified_prompt(
                "select_config",
                "Choose configuration",
                config_choices,
                allow_back=False,
            )

            if selected == "Cancel":
                return "continue"

            config_name = selected.split(" (")[0]
            proxy_config = config_manager.load_named_config(config_name)

        if proxy_config:
            # Display configuration summary
            console.print("\n[bold]Configuration Summary:[/bold]")
            console.print(f"Port: {proxy_config.port}")
            console.print(f"Models: {len(proxy_config.models)}")
            for model in proxy_config.models:
                gpu_str = (
                    ",".join(str(g) for g in model.gpu_ids) if model.gpu_ids else "Auto"
                )
                console.print(f"  • {model.name} (Port {model.port}, GPU {gpu_str})")

            # Confirm start
            if (
                unified_prompt(
                    "confirm_start",
                    "\nStart this proxy configuration?",
                    ["Yes, start proxy", "No, go back"],
                    allow_back=False,
                )
                == "Yes, start proxy"
            ):
                # Start the proxy
                proxy_manager = ProxyManager(proxy_config)
                console.print("\n[cyan]Starting model servers...[/cyan]")
                started = proxy_manager.start_all_models()
                console.print(f"Started {started}/{len(proxy_config.models)} models")

                if proxy_manager.start_proxy():
                    console.print(
                        f"\n[green]✓ Proxy server running at "
                        f"http://{proxy_config.host}:{proxy_config.port}[/green]"
                    )

                    # Enter simplified proxy management
                    manage_running_proxy(proxy_manager, proxy_config)

    elif choice == "Configure new proxy":
        proxy_config = configure_proxy_interactively()
        if not proxy_config:
            return "continue"

        # Validate configuration
        config_manager = ProxyConfigManager()
        errors = config_manager.validate_config(proxy_config)
        if errors:
            console.print("[red]Configuration errors:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            input("\nPress Enter to continue...")
            return "continue"

        # Ask what to do with the configuration
        console.print("\n[bold cyan]Configuration Complete[/bold cyan]")
        action_options = [
            "Save and start now",
            "Save for later use",
            "Start without saving",
            "Cancel",
        ]

        action = unified_prompt(
            "config_action",
            "What would you like to do with this configuration?",
            action_options,
            allow_back=False,
        )

        if action == "Cancel":
            return "continue"

        # Handle saving if requested
        if action in ["Save and start now", "Save for later use"]:
            console.print("\nEnter a name for this configuration:")
            config_name = input("Name (default: 'default'): ").strip() or "default"
            config_manager.save_named_config(proxy_config, config_name)
            console.print(f"[green]✓ Configuration saved as '{config_name}'[/green]")

            if action == "Save for later use":
                console.print(
                    "\n[dim]You can start this configuration later from the proxy menu.[/dim]"
                )
                input("\nPress Enter to continue...")
                return "continue"

        # Start the proxy if requested
        if action in ["Save and start now", "Start without saving"]:
            # Start proxy
            proxy_manager = ProxyManager(proxy_config)
            console.print("\n[cyan]Starting model servers...[/cyan]")
            started = proxy_manager.start_all_models()
            console.print(f"Started {started}/{len(proxy_config.models)} models")

            if proxy_manager.start_proxy():
                console.print(
                    f"\n[green]✓ Proxy server running at "
                    f"http://{proxy_config.host}:{proxy_config.port}[/green]"
                )

                # Enter simplified proxy management
                manage_running_proxy(proxy_manager, proxy_config)

    return "continue"


def show_main_menu() -> str:
    """
    Display the main menu and return the selected action.
    """
    # Import here to avoid circular dependencies
    from .model_manager import handle_model_management
    from .server_control import (
        handle_custom_config,
        handle_quick_serve,
        handle_serve_with_profile,
    )
    from .server_monitor import monitor_active_servers
    from .settings import handle_settings
    from .system_info import show_system_info

    # Check for active proxy and servers
    global _active_proxy_manager, _active_proxy_config
    active_servers = get_active_servers()

    menu_options = []

    # Add proxy monitoring option if proxy is running
    if _active_proxy_manager and _active_proxy_config:
        try:
            # Check if proxy is actually still running
            if _active_proxy_manager.proxy_process:
                menu_options.append("Return to Proxy Monitoring")
        except Exception:
            # If there's any error checking, clear the variables
            _active_proxy_manager = None
            _active_proxy_config = None

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
        if _active_proxy_manager and _active_proxy_config:
            manage_running_proxy(_active_proxy_manager, _active_proxy_config)
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
