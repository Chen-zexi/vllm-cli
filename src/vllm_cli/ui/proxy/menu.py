#!/usr/bin/env python3
"""
Proxy menu module for vLLM CLI.

Handles all proxy-related menu functionality.
"""
import logging
import threading
import time

import inquirer

from ...proxy import ProxyManager
from ...proxy.config import ProxyConfigManager
from ..common import console
from ..navigation import unified_prompt
from .control import configure_model_for_proxy, configure_proxy_interactively
from .monitor import (
    monitor_individual_model_by_name,
    monitor_model_logs_menu,
    monitor_proxy_logs,
    monitor_startup_progress,
    refresh_model_registry,
)

logger = logging.getLogger(__name__)

# Module-level variables to track active proxy
_active_proxy_manager = None
_active_proxy_config = None


def get_active_proxy():
    """Get the currently active proxy manager and config."""
    return _active_proxy_manager, _active_proxy_config


def manage_models_menu(proxy_manager, proxy_config) -> None:
    """
    Model management submenu for adding new models or managing existing ones.

    Args:
        proxy_manager: The ProxyManager instance
        proxy_config: The ProxyConfig instance
    """
    while True:
        console.print("\n[bold cyan]Model Management[/bold cyan]\n")

        options = [
            "Add New Model",
            "Manage Existing Models",
            "← Back to proxy menu",
        ]

        choice = unified_prompt(
            "model_management", "Select action", options, allow_back=False
        )

        if choice == "BACK" or choice == "← Back to proxy menu":
            return

        if choice == "Add New Model":
            add_new_model_to_proxy(proxy_manager, proxy_config)
        elif choice == "Manage Existing Models":
            manage_existing_models(proxy_manager, proxy_config)


def add_new_model_to_proxy(proxy_manager, proxy_config) -> None:
    """
    Add a new model to the running proxy using the unified configuration UI.

    Args:
        proxy_manager: The ProxyManager instance
        proxy_config: The ProxyConfig instance
    """
    # Use the unified function for consistent UI experience
    new_model = configure_model_for_proxy(
        index=len(proxy_config.models),
        existing_models=proxy_config.models,
        is_running_proxy=True,
        proxy_manager=proxy_manager,
    )

    if not new_model:
        return

    # Check if we need to handle port conflicts with stopped models
    # (The unified function only checks for conflicts, doesn't handle reuse)
    for existing_model in proxy_config.models[:]:
        if existing_model.port == new_model.port:
            # The unified function should have prevented this for running models
            # This handles the case of reusing a stopped model's port
            is_running = (
                existing_model.name in proxy_manager.vllm_servers
                and proxy_manager.vllm_servers[existing_model.name].is_running()
            )
            if not is_running:
                console.print(
                    f"[yellow]Removing stopped model '{existing_model.name}' from port {existing_model.port}[/yellow]"
                )
                proxy_config.models.remove(existing_model)

    # Check GPU memory and show warnings
    if new_model.gpu_ids:
        from ..gpu_utils import check_gpu_memory_warnings

        warnings = check_gpu_memory_warnings(new_model.gpu_ids)
        if warnings:
            console.print("\n[yellow]GPU Memory Warnings:[/yellow]")
            for warning in warnings:
                console.print(warning)
            console.print(
                "\n[yellow]Starting additional models may cause Out-Of-Memory errors.[/yellow]"
            )

            # Ask user if they want to continue
            if not inquirer.confirm("Continue anyway?", default=False):
                return

    # Add to proxy config
    proxy_config.models.append(new_model)

    # Start the model
    console.print(
        f"\n[cyan]Starting {new_model.name} on port {new_model.port}...[/cyan]"
    )
    if proxy_manager.start_model(new_model):
        console.print("[green]✓ Model process started[/green]")

        # Start registration in background thread
        registration_thread = threading.Thread(
            target=proxy_manager.wait_and_register_model, args=(new_model,), daemon=True
        )
        registration_thread.start()

        # Immediately show monitoring - user sees logs right away!
        console.print(f"\n[cyan]Monitoring {new_model.name} startup...[/cyan]")
        console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")

        # Monitor shows real-time logs during startup
        result = monitor_individual_model_by_name(proxy_manager, new_model.name)

        # Ensure registration thread completes
        registration_thread.join(timeout=1.0)

        return result
    else:
        console.print(f"[red]✗ Failed to start model {new_model.name}[/red]")

        # Offer to view logs if server was created
        if new_model.name in proxy_manager.vllm_servers:
            server = proxy_manager.vllm_servers[new_model.name]

            # Show last few log lines
            recent_logs = server.get_recent_logs(5)
            if recent_logs:
                console.print("\n[bold]Last logs:[/bold]")
                for log in recent_logs:
                    console.print(f"  {log}")

            # Offer to view full logs
            view_logs = (
                input(f"\nView full logs for {new_model.name}? (y/N): ").strip().lower()
            )
            if view_logs in ["y", "yes"]:
                from ..log_viewer import show_log_menu

                show_log_menu(server)
            else:
                if server.log_path:
                    console.print(f"[dim]Log file: {server.log_path}[/dim]")
                input("\nPress Enter to continue...")

        # Remove from config if failed
        proxy_config.models.remove(new_model)


def manage_existing_models(proxy_manager, proxy_config) -> None:
    """
    Manage existing models with sleep/stop/start options.

    Args:
        proxy_manager: The ProxyManager instance
        proxy_config: The ProxyConfig instance
    """
    console.print("\n[bold cyan]Manage Existing Models[/bold cyan]\n")

    if not proxy_config.models:
        console.print("[yellow]No models configured[/yellow]")
        input("\nPress Enter to continue...")
        return

    # Build current state with sleep status
    model_states = []
    for model in proxy_config.models:
        if model.enabled:
            # Check if model is running
            is_running = (
                model.name in proxy_manager.vllm_servers
                and proxy_manager.vllm_servers[model.name].is_running()
            )
            is_sleeping = False

            # Get sleep state from registry if proxy is running
            if (
                is_running
                and proxy_manager.proxy_process
                and proxy_manager.proxy_process.is_running()
            ):
                registry_status = proxy_manager.get_proxy_registry_status()
                if registry_status:
                    models_info = registry_status.get("models", [])
                    for model_info in models_info:
                        if model_info.get("port") == model.port:
                            is_sleeping = model_info.get("state") == "sleeping"
                            break

            # Build status label with emoji
            if is_running and not is_sleeping:
                status = "[●] Running"
            elif is_sleeping:
                status = "[z] Sleeping"
            else:
                status = "[○] Stopped"

            label = f"{model.name} (Port: {model.port})"
            if model.gpu_ids:
                label += f" GPU: {','.join(map(str, model.gpu_ids))}"
            label += f" - {status}"

            model_states.append(
                {
                    "label": label,
                    "model": model,
                    "running": is_running,
                    "sleeping": is_sleeping,
                }
            )

    if not model_states:
        console.print("[yellow]No enabled models found[/yellow]")
        input("\nPress Enter to continue...")
        return

    # Let user select a model to manage
    model_choices = [state["label"] for state in model_states]
    model_choices.append("← Back")

    selected = unified_prompt(
        "select_model", "Select a model to manage", model_choices, allow_back=False
    )

    if selected == "← Back" or not selected:
        return

    # Find the selected model
    selected_state = None
    for state in model_states:
        if state["label"] == selected:
            selected_state = state
            break

    if not selected_state:
        return

    model = selected_state["model"]
    is_running = selected_state["running"]
    is_sleeping = selected_state["sleeping"]

    # Show action options based on model state
    actions = []
    if is_running and not is_sleeping:
        actions = [
            "[z] Put to sleep (free GPU, keep port)",
            "[■] Stop completely (free GPU and port)",
            "↻ Restart model",
            "← Back",
        ]
    elif is_sleeping:
        actions = ["[!] Wake up model", "[■] Stop completely (free port)", "← Back"]
    else:  # Stopped
        actions = ["► Start model", "[×] Remove model", "← Back"]

    action = unified_prompt(
        "model_action", f"Action for {model.name}", actions, allow_back=False
    )

    if action == "← Back" or not action:
        return manage_existing_models(
            proxy_manager, proxy_config
        )  # Go back to model list

    # Process the selected action
    if "Put to sleep" in action:
        console.print(f"\n[cyan]Putting {model.name} to sleep...[/cyan]")
        console.print("[dim]This may take several minutes for large models[/dim]")

        # Initiate sleep operation (returns immediately)
        if proxy_manager.sleep_model(model.name):
            console.print(
                "[yellow]Sleep command sent. Monitoring progress...[/yellow]\n"
            )
            console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")

            # Monitor the model logs - handles completion detection and notifications
            return monitor_individual_model_by_name(proxy_manager, model.name)
        else:
            console.print(f"[red]✗ Failed to initiate sleep for {model.name}[/red]")

    elif "Wake up" in action:
        console.print(f"\n[cyan]Waking up {model.name}...[/cyan]")
        console.print("[dim]This may take several minutes for large models[/dim]")

        # Initiate wake operation (returns immediately)
        if proxy_manager.wake_model(model.name):
            console.print(
                "[yellow]Wake command sent. Monitoring progress...[/yellow]\n"
            )
            console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")

            # Monitor the model logs - handles completion detection and notifications
            return monitor_individual_model_by_name(proxy_manager, model.name)
        else:
            console.print(f"[red]✗ Failed to initiate wake for {model.name}[/red]")

    elif "Stop completely" in action:
        console.print(f"\n[yellow]Stopping {model.name}...[/yellow]")
        if proxy_manager.stop_model(model.name):
            console.print(
                f"[green]✓ {model.name} stopped (port {model.port} is now free)[/green]"
            )
            # Remove from configuration
            proxy_config.models = [
                m for m in proxy_config.models if m.name != model.name
            ]
        else:
            console.print(f"[red]✗ Failed to stop {model.name}[/red]")

    elif "Start model" in action:
        console.print(f"\n[cyan]Starting {model.name}...[/cyan]")
        if proxy_manager.start_model(model):
            console.print("[green]✓ Model process started[/green]")

            # Background registration
            registration_thread = threading.Thread(
                target=proxy_manager.wait_and_register_model, args=(model,), daemon=True
            )
            registration_thread.start()

            # Immediate monitoring
            console.print(f"\n[cyan]Monitoring {model.name} startup...[/cyan]")
            console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")

            return monitor_individual_model_by_name(proxy_manager, model.name)
        else:
            console.print(f"[red]✗ Failed to start {model.name}[/red]")

            # Offer to view logs if server was created
            if model.name in proxy_manager.vllm_servers:
                server = proxy_manager.vllm_servers[model.name]

                # Show last few log lines
                recent_logs = server.get_recent_logs(5)
                if recent_logs:
                    console.print("\n[bold]Last logs:[/bold]")
                    for log in recent_logs:
                        console.print(f"  {log}")

                # Offer to view full logs
                view_logs = input("\nView full logs? (y/N): ").strip().lower()
                if view_logs in ["y", "yes"]:
                    from ..log_viewer import show_log_menu

                    show_log_menu(server)
                    return manage_existing_models(proxy_manager, proxy_config)

    elif "Restart" in action:
        console.print(f"\n[cyan]Restarting {model.name}...[/cyan]")
        proxy_manager.stop_model(model.name)
        console.print("[dim]Waiting for cleanup...[/dim]")
        time.sleep(2)

        if proxy_manager.start_model(model):
            console.print("[green]✓ Model process restarted[/green]")

            # Background registration
            registration_thread = threading.Thread(
                target=proxy_manager.wait_and_register_model, args=(model,), daemon=True
            )
            registration_thread.start()

            # Immediate monitoring
            console.print(f"\n[cyan]Monitoring {model.name} restart...[/cyan]")
            console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")

            return monitor_individual_model_by_name(proxy_manager, model.name)
        else:
            console.print(f"[red]✗ Failed to restart {model.name}[/red]")

            # Offer to view logs if server was created
            if model.name in proxy_manager.vllm_servers:
                server = proxy_manager.vllm_servers[model.name]

                # Show last few log lines
                recent_logs = server.get_recent_logs(5)
                if recent_logs:
                    console.print("\n[bold]Last logs:[/bold]")
                    for log in recent_logs:
                        console.print(f"  {log}")

                # Offer to view full logs
                view_logs = input("\nView full logs? (y/N): ").strip().lower()
                if view_logs in ["y", "yes"]:
                    from ..log_viewer import show_log_menu

                    show_log_menu(server)
                    return manage_existing_models(proxy_manager, proxy_config)

    elif "Remove model" in action:
        if inquirer.confirm(f"\nRemove {model.name} from proxy?", default=False):
            # Unregister from proxy registry to prevent stale entries
            proxy_manager.unregister_model_from_proxy(model.port)

            # Remove from configuration
            proxy_config.models = [
                m for m in proxy_config.models if m.name != model.name
            ]
            console.print(f"[green]✓ {model.name} removed from proxy[/green]")
            input("\nPress Enter to continue...")
            return

    input("\nPress Enter to continue...")
    # After any action, go back to the model list
    return manage_existing_models(proxy_manager, proxy_config)


def manage_running_proxy(proxy_manager, proxy_config) -> None:
    """
    Simplified proxy management interface focusing on monitoring.

    Args:
        proxy_manager: The ProxyManager instance
        proxy_config: The proxy configuration
    """
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
            "Manage Models",
            "Refresh Model Registry",
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
        elif mgmt_choice == "Manage Models":
            manage_models_menu(proxy_manager, proxy_config)
            # Continue to show menu after managing models
        elif mgmt_choice == "Refresh Model Registry":
            refresh_model_registry(proxy_manager)
            # Continue to show menu after refresh
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

            # Confirm start - use inquirer.confirm for inline prompt
            console.print()  # Add blank line before prompt
            if inquirer.confirm("Start this proxy configuration?", default=True):
                # Start the proxy
                proxy_manager = ProxyManager(proxy_config)

                # Start model servers without waiting, show logs immediately
                console.print("\n[cyan]Launching model servers...[/cyan]")
                launched = proxy_manager.start_all_models_no_wait()

                if launched > 0:
                    # Monitor startup progress with live logs
                    all_started = monitor_startup_progress(proxy_manager)

                    if not all_started:
                        console.print("[yellow]Some models failed to start.[/yellow]")
                        if not inquirer.confirm(
                            "Continue with available models?", default=False
                        ):
                            console.print("[yellow]Stopping all servers...[/yellow]")
                            proxy_manager.stop_proxy()
                            return "continue"

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

            # Start model servers without waiting, show logs immediately
            console.print("\n[cyan]Launching model servers...[/cyan]")
            launched = proxy_manager.start_all_models_no_wait()

            if launched > 0:
                # Monitor startup progress with live logs
                all_started = monitor_startup_progress(proxy_manager)

                if not all_started:
                    console.print("[yellow]Some models failed to start.[/yellow]")
                    if not inquirer.confirm(
                        "Continue with available models?", default=False
                    ):
                        console.print("[yellow]Stopping all servers...[/yellow]")
                        proxy_manager.stop_proxy()
                        return "continue"

            if proxy_manager.start_proxy():
                console.print(
                    f"\n[green]✓ Proxy server running at "
                    f"http://{proxy_config.host}:{proxy_config.port}[/green]"
                )

                # Enter simplified proxy management
                manage_running_proxy(proxy_manager, proxy_config)

    return "continue"
