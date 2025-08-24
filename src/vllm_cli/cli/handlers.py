#!/usr/bin/env python3
"""
Command handlers for vLLM CLI commands.

Implements the actual logic for each CLI command including
serve, info, models, status, and stop operations.
"""
import argparse
import logging
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .. import __version__
from ..config import ConfigManager
from ..models import list_available_models
from ..proxy import ProxyManager
from ..proxy.config import ProxyConfigManager
from ..proxy.models import ModelConfig
from ..server import (
    VLLMServer,
    find_server_by_model,
    find_server_by_port,
    get_active_servers,
    stop_all_servers,
)
from ..system import (
    check_vllm_installation,
    format_size,
    get_cuda_version,
    get_memory_info,
)
from ..ui.gpu_utils import create_gpu_status_panel

logger = logging.getLogger(__name__)
console = Console()


def handle_serve(args: argparse.Namespace) -> bool:
    """
    Handle the 'serve' command to start a vLLM server.

    Processes the serve command arguments, sets up configuration,
    and starts a new vLLM server instance.

    Args:
        args: Parsed command line arguments

    Returns:
        True if server started successfully, False otherwise
    """
    try:
        config_manager = ConfigManager()

        # Validate that either model or shortcut is provided
        if not args.model and not (hasattr(args, "shortcut") and args.shortcut):
            console.print(
                "[red]Error: Either MODEL or --shortcut must be specified.[/red]"
            )
            console.print("Usage: vllm-cli serve MODEL [options]")
            console.print("   or: vllm-cli serve --shortcut SHORTCUT_NAME [options]")
            return False

        # Handle shortcut mode
        if hasattr(args, "shortcut") and args.shortcut:
            shortcut = config_manager.get_shortcut(args.shortcut)
            if not shortcut:
                console.print(f"[red]Shortcut '{args.shortcut}' not found.[/red]")
                console.print("Use 'vllm-cli shortcuts' to list available shortcuts.")
                return False

            # Get profile configuration
            profile = config_manager.get_profile(shortcut["profile"])
            if not profile:
                console.print(
                    f"[red]Profile '{shortcut['profile']}' referenced by shortcut not found.[/red]"
                )
                return False

            # Build config from shortcut
            config = profile.get("config", {}).copy()
            config["model"] = shortcut["model"]

            # Apply any config overrides from shortcut
            if "config_overrides" in shortcut:
                config.update(shortcut["config_overrides"])

            # Override model in args for display purposes
            args.model = shortcut["model"]

            # Update last used timestamp
            config_manager.shortcut_manager.update_last_used(args.shortcut)

            console.print(f"[cyan]Using shortcut: {args.shortcut}[/cyan]")
            console.print(f"  Model: {shortcut['model']}")
            console.print(f"  Profile: {shortcut['profile']}")
        else:
            # Build configuration from arguments
            config = _build_serve_config(args, config_manager)

        # Validate configuration
        is_valid, errors = config_manager.validate_config(config)
        if not is_valid:
            console.print("[red]Configuration validation failed:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            return False

        # Check for compatibility issues
        is_compatible, warnings = config_manager.validate_argument_combination(config)
        if warnings:
            console.print("[yellow]Configuration warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  • {warning}")

        # Save profile if requested
        if args.save_profile:
            profile_data = {
                "name": args.save_profile,
                "description": f"Profile for {args.model}",
                "config": config,
            }
            config_manager.save_user_profile(args.save_profile, profile_data)
            console.print(f"[green]Saved profile: {args.save_profile}[/green]")

        # Save shortcut if requested
        if hasattr(args, "save_shortcut") and args.save_shortcut:
            # Need a profile for the shortcut
            if args.profile:
                profile_name = args.profile
            elif args.save_profile:
                profile_name = args.save_profile
            else:
                console.print(
                    "[yellow]Creating default profile for shortcut...[/yellow]"
                )
                profile_name = f"config_{args.model.replace('/', '_')}"
                profile_data = {
                    "name": profile_name,
                    "description": f"Auto-generated profile for {args.model}",
                    "config": config,
                }
                config_manager.save_user_profile(profile_name, profile_data)

            shortcut_data = {
                "model": args.model,
                "profile": profile_name,
                "description": f"Shortcut for {args.model}",
            }
            if config_manager.save_shortcut(args.save_shortcut, shortcut_data):
                console.print(f"[green]Saved shortcut: {args.save_shortcut}[/green]")
                console.print(
                    f"Use 'vllm-cli serve --shortcut \"{args.save_shortcut}\"' to run it."
                )
            else:
                console.print(
                    f"[red]Failed to save shortcut: {args.save_shortcut}[/red]"
                )

        # Create and start server
        server = VLLMServer(config)

        # Check if this is a remote model
        is_remote_model = "/" in args.model and not args.model.startswith("/")

        if is_remote_model:
            console.print(
                f"[blue]Starting vLLM server for remote model: {args.model}[/blue]"
            )
            console.print(
                "[yellow]Note: Model will be downloaded from HuggingFace Hub if not cached[/yellow]"
            )
        else:
            console.print(f"[blue]Starting vLLM server for model: {args.model}[/blue]")

        console.print(f"Port: {config.get('port', 8000)}")
        console.print(f"Host: {config.get('host', 'localhost')}")

        if server.start():
            console.print("[green]✓ Server started successfully[/green]")
            console.print(
                f"Server URL: http://{config.get('host', 'localhost')}:{config.get('port', 8000)}"
            )

            # Save as last used configuration
            config_manager.save_last_config(config)
            config_manager.add_recent_model(args.model)

            return True
        else:
            console.print("[red]✗ Failed to start server[/red]")
            return False

    except Exception as e:
        logger.exception(f"Error in serve command: {e}")
        console.print(f"[red]Error starting server: {e}[/red]")
        return False


def handle_info() -> bool:
    """
    Handle the 'info' command to show system information.

    Displays comprehensive system information including GPU status,
    memory usage, and software versions.

    Returns:
        True if information was displayed successfully
    """
    try:
        console.print("\n[bold cyan]System Information[/bold cyan]\n")

        # GPU Information
        gpu_panel = create_gpu_status_panel()
        console.print(gpu_panel)

        # System Memory
        memory_info = get_memory_info()
        memory_panel = Panel(
            f"Total: {format_size(memory_info['total'])}\n"
            f"Used: {format_size(memory_info['used'])} ({memory_info['percent']:.1f}%)\n"
            f"Available: {format_size(memory_info['available'])}",
            title="System Memory",
            border_style="blue",
        )
        console.print(memory_panel)

        # Software Information
        cuda_version = get_cuda_version()
        software_info = f"vLLM CLI: {__version__}\n"

        try:
            import torch

            software_info += f"PyTorch: {torch.__version__}\n"
        except ImportError:
            software_info += "PyTorch: Not installed\n"

        if cuda_version:
            software_info += f"CUDA: {cuda_version}"
        else:
            software_info += "CUDA: Not available"

        software_panel = Panel(software_info, title="Software", border_style="green")
        console.print(software_panel)

        # vLLM Installation Check
        if check_vllm_installation():
            console.print("[green]✓ vLLM is properly installed[/green]")
        else:
            console.print("[yellow]⚠ vLLM not found or not properly installed[/yellow]")

        return True

    except Exception as e:
        logger.exception(f"Error in info command: {e}")
        console.print(f"[red]Error getting system information: {e}[/red]")
        return False


def handle_models() -> bool:
    """
    Handle the 'models' command to list available models.

    Displays all available models that can be served with vLLM,
    including their paths and sizes.

    Returns:
        True if models were listed successfully
    """
    try:
        console.print("\n[bold cyan]Available Models[/bold cyan]\n")

        # Get available models
        models = list_available_models()

        if not models:
            console.print("[yellow]No models found.[/yellow]")
            console.print("Use hf-model-tool to download models first.")
            return True

        # Create table
        table = Table(title=f"Found {len(models)} model(s)")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Size", style="magenta")
        table.add_column("Type", style="green")
        table.add_column("Path", style="dim", overflow="fold")

        # Sort models by name
        models.sort(key=lambda x: x["name"])

        for model in models:
            size_str = format_size(model["size"]) if model["size"] > 0 else "Unknown"
            model_type = model.get("type", "model")
            path = model.get("path", "Unknown")

            table.add_row(model["name"], size_str, model_type, path)

        console.print(table)
        return True

    except Exception as e:
        logger.exception(f"Error in models command: {e}")
        console.print(f"[red]Error listing models: {e}[/red]")
        return False


def handle_shortcuts(args) -> bool:
    """
    Handle the 'shortcuts' command to list and manage shortcuts.

    Args:
        args: Parsed command line arguments

    Returns:
        True if operation was successful
    """
    try:
        from pathlib import Path

        config_manager = ConfigManager()

        # Handle delete operation
        if hasattr(args, "delete") and args.delete:
            if config_manager.delete_shortcut(args.delete):
                console.print(f"[green]✓ Shortcut '{args.delete}' deleted.[/green]")
            else:
                console.print(f"[red]Shortcut '{args.delete}' not found.[/red]")
            return True

        # Handle export operation
        if hasattr(args, "export") and args.export:
            shortcut = config_manager.get_shortcut(args.export)
            if not shortcut:
                console.print(f"[red]Shortcut '{args.export}' not found.[/red]")
                return False

            file_path = Path(f"shortcut_{args.export}.json")
            if config_manager.shortcut_manager.export_shortcut(args.export, file_path):
                console.print(f"[green]✓ Shortcut exported to {file_path}[/green]")
            else:
                console.print("[red]Failed to export shortcut.[/red]")
            return True

        # Handle import operation
        if hasattr(args, "import_file") and args.import_file:
            file_path = Path(args.import_file)
            if not file_path.exists():
                console.print(f"[red]File not found: {args.import_file}[/red]")
                return False

            try:
                if config_manager.shortcut_manager.import_shortcut(file_path):
                    console.print("[green]✓ Shortcut imported successfully![/green]")
                else:
                    console.print("[red]Failed to import shortcut.[/red]")
            except Exception as e:
                console.print(f"[red]Error importing shortcut: {e}[/red]")
                return False
            return True

        # Default: List all shortcuts
        shortcuts = config_manager.list_shortcuts()

        if not shortcuts:
            console.print("\n[yellow]No shortcuts configured.[/yellow]")
            console.print(
                "\nCreate shortcuts to quickly launch frequently used configurations:"
            )
            console.print(
                "  • From CLI: vllm-cli serve MODEL --profile PROFILE --save-shortcut NAME"
            )
            console.print("  • From interactive mode: Settings → Manage Shortcuts")
            return True

        # Create table
        table = Table(
            title=f"[bold cyan]Configured Shortcuts ({len(shortcuts)})[/bold cyan]"
        )
        table.add_column("Shortcut", style="cyan", no_wrap=True)
        table.add_column("Model", style="green")
        table.add_column("Profile", style="magenta")
        table.add_column("Description", style="dim")

        for shortcut in shortcuts:
            model = shortcut["model"]
            # Truncate long model paths
            if len(model) > 40:
                model_display = "..." + model[-37:]
            else:
                model_display = model

            table.add_row(
                shortcut["name"],
                model_display,
                shortcut["profile"],
                shortcut.get("description", ""),
            )

        console.print("")
        console.print(table)
        console.print("\n[dim]Usage:[/dim]")
        console.print("  • Launch: vllm-cli serve --shortcut SHORTCUT_NAME")
        console.print("  • Delete: vllm-cli shortcuts --delete SHORTCUT_NAME")
        console.print("  • Export: vllm-cli shortcuts --export SHORTCUT_NAME")

        return True

    except Exception as e:
        logger.exception(f"Error in shortcuts command: {e}")
        console.print(f"[red]Error managing shortcuts: {e}[/red]")
        return False


def handle_status() -> bool:
    """
    Handle the 'status' command to show active servers.

    Displays status information for all currently running
    vLLM servers including PIDs, ports, and uptime.

    Returns:
        True if status was displayed successfully
    """
    try:
        console.print("\n[bold cyan]Active vLLM Servers[/bold cyan]\n")

        # Get active servers
        servers = get_active_servers()

        if not servers:
            console.print("[yellow]No active servers found.[/yellow]")
            return True

        # Create table
        table = Table(title=f"{len(servers)} active server(s)")
        table.add_column("Model", style="cyan")
        table.add_column("Port", style="magenta")
        table.add_column("PID", style="green")
        table.add_column("Status", style="blue")
        table.add_column("Uptime", style="yellow")

        for server in servers:
            status = server.get_status()

            # Determine status
            if status["running"]:
                status_str = "[●] Running"
            else:
                status_str = "[×] Stopped"

            # Format uptime
            uptime_str = status.get("uptime_str", "Unknown")

            table.add_row(
                status["model"],
                str(status["port"]),
                str(status["pid"]) if status["pid"] else "N/A",
                status_str,
                uptime_str,
            )

        console.print(table)
        return True

    except Exception as e:
        logger.exception(f"Error in status command: {e}")
        console.print(f"[red]Error getting server status: {e}[/red]")
        return False


def handle_stop(args: argparse.Namespace) -> bool:
    """
    Handle the 'stop' command to stop vLLM servers.

    Stops one or more vLLM servers based on the provided arguments
    (specific model, port, or all servers).

    Args:
        args: Parsed command line arguments

    Returns:
        True if servers were stopped successfully
    """
    try:
        if args.all:
            # Stop all servers
            console.print("[blue]Stopping all servers...[/blue]")
            stopped_count = stop_all_servers()

            if stopped_count > 0:
                console.print(f"[green]✓ Stopped {stopped_count} server(s)[/green]")
            else:
                console.print("[yellow]No servers were running[/yellow]")

            return True

        elif args.port:
            # Stop server by port
            server = find_server_by_port(args.port)
            if server:
                console.print(f"[blue]Stopping server on port {args.port}...[/blue]")
                if server.stop():
                    console.print(
                        f"[green]✓ Stopped server on port {args.port}[/green]"
                    )
                    return True
                else:
                    console.print(
                        f"[red]✗ Failed to stop server on port {args.port}[/red]"
                    )
                    return False
            else:
                console.print(f"[yellow]No server found on port {args.port}[/yellow]")
                return False

        elif args.model:
            # Stop server by model name or try as port number
            server = None

            # First try as model name
            server = find_server_by_model(args.model)

            # If not found, try as port number
            if not server:
                try:
                    port = int(args.model)
                    server = find_server_by_port(port)
                except ValueError:
                    pass

            if server:
                console.print(f"[blue]Stopping server for {args.model}...[/blue]")
                if server.stop():
                    console.print(f"[green]✓ Stopped server for {args.model}[/green]")
                    return True
                else:
                    console.print(
                        f"[red]✗ Failed to stop server for {args.model}[/red]"
                    )
                    return False
            else:
                console.print(f"[yellow]No server found for {args.model}[/yellow]")
                return False

        return True

    except Exception as e:
        logger.exception(f"Error in stop command: {e}")
        console.print(f"[red]Error stopping server: {e}[/red]")
        return False


def _build_serve_config(
    args: argparse.Namespace, config_manager: ConfigManager
) -> Dict[str, Any]:
    """
    Build server configuration from command line arguments.

    Args:
        args: Parsed command line arguments
        config_manager: ConfigManager instance for profile handling

    Returns:
        Configuration dictionary for the server
    """
    # Handle special case where model might be a dict (for GGUF/Ollama models from UI)
    if isinstance(args.model, dict):
        # Check if this is an Ollama model with name metadata
        if args.model.get("type") == "ollama_model" and args.model.get("name"):
            # For Ollama models, create special config with served_model_name
            config = {
                "model": {
                    "model": args.model.get("path", args.model.get("model")),
                    "quantization": "gguf",
                    "served_model_name": args.model.get(
                        "name"
                    ),  # Use Ollama model name
                }
            }
            console.print(f"[cyan]Using Ollama model: {args.model.get('name')}[/cyan]")
        else:
            # Extract GGUF-specific configuration (non-Ollama)
            config = {
                "model": args.model.get("model"),
                "quantization": args.model.get("quantization", "gguf"),
            }
        # Add warning about experimental support
        if args.model.get("experimental"):
            console.print("[yellow]⚠ Using experimental GGUF support[/yellow]")
    else:
        config = {"model": args.model}

    # Load profile if specified
    if args.profile:
        profile = config_manager.get_profile(args.profile)
        if profile and "config" in profile:
            config.update(profile["config"])
        else:
            console.print(
                f"[yellow]Warning: Profile '{args.profile}' not found[/yellow]"
            )

    # Override with command line arguments
    if args.port:
        config["port"] = args.port
    if args.host:
        config["host"] = args.host
    if args.quantization:
        config["quantization"] = args.quantization

    # Handle HF token if provided via CLI
    if hasattr(args, "hf_token") and args.hf_token:
        console.print("[cyan]Validating HuggingFace token...[/cyan]")

        from ..validation.token import validate_hf_token

        is_valid, user_info = validate_hf_token(args.hf_token)

        if is_valid:
            # Save the token to config for this session
            config_manager.config["hf_token"] = args.hf_token
            config_manager._save_config()
            console.print("[green]✓ Token validated and saved[/green]")
            if user_info:
                console.print(
                    f"[dim]Authenticated as: {user_info.get('name', 'Unknown')}[/dim]"
                )
        else:
            console.print("[yellow]Warning: Token validation failed[/yellow]")
            console.print(
                "[dim]The token may be invalid or expired. Continuing anyway...[/dim]"
            )
            # Still save it in case it's a network issue or special token type
            config_manager.config["hf_token"] = args.hf_token
            config_manager._save_config()
    if args.tensor_parallel_size:
        config["tensor_parallel_size"] = args.tensor_parallel_size
    if args.gpu_memory_utilization != 0.9:
        config["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_model_len:
        config["max_model_len"] = args.max_model_len
    if args.dtype != "auto":
        config["dtype"] = args.dtype

    # Handle GPU device selection
    if hasattr(args, "device") and args.device:
        config["device"] = args.device

    # Handle LoRA adapters
    if hasattr(args, "lora") and args.lora:
        # Enable LoRA if adapters are specified
        config["enable_lora"] = True

        # Format LoRA modules for vLLM
        lora_modules = []
        for lora_spec in args.lora:
            if "=" in lora_spec:
                # Format: name=path
                lora_modules.append(lora_spec)
            else:
                # Just path, generate a name from the path
                from pathlib import Path

                lora_path = Path(lora_spec)
                lora_name = lora_path.name.replace("-", "_").replace(" ", "_")
                lora_modules.append(f"{lora_name}={lora_spec}")

        # Join modules for command line
        config["lora_modules"] = " ".join(lora_modules)

        console.print(f"[blue]Enabling LoRA with {len(lora_modules)} adapter(s)[/blue]")
        for module in lora_modules:
            console.print(f"  • {module}")

    elif hasattr(args, "enable_lora") and args.enable_lora:
        config["enable_lora"] = True

    if args.extra_args:
        config["extra_args"] = args.extra_args

    return config


def handle_dirs(args: argparse.Namespace) -> bool:
    """
    Directory management is now handled by hf-model-tool.
    This command redirects users to use hf-model-tool.

    Args:
        args: Parsed command line arguments

    Returns:
        True if operation succeeded, False otherwise
    """
    import os
    import subprocess

    console.print("[yellow]Directory management has moved to hf-model-tool[/yellow]")
    console.print("\nYou can manage directories using:")
    console.print(
        "  • [cyan]hf-model-tool[/cyan] - Interactive interface with Config menu"
    )
    console.print(
        "  • [cyan]hf-model-tool --add-path <path>[/cyan] - Add a directory directly"
    )

    if hasattr(args, "dirs_command"):
        if args.dirs_command in ["add", "remove", "list"]:
            console.print(
                "\n[dim]Launching hf-model-tool for directory management...[/dim]"
            )

            try:
                # Launch hf-model-tool
                if args.dirs_command == "add" and hasattr(args, "path"):
                    # If adding a path, use the --add-path argument
                    subprocess.run(
                        ["hf-model-tool", "--add-path", args.path],
                        env=os.environ.copy(),
                    )
                else:
                    # Otherwise launch interactive mode
                    subprocess.run(["hf-model-tool"], env=os.environ.copy())
                return True
            except FileNotFoundError:
                console.print(
                    "\n[red]hf-model-tool not found. Please install it:[/red]"
                )
                console.print("  pip install hf-model-tool")
                return False
            except Exception as e:
                console.print(f"\n[red]Error launching hf-model-tool: {e}[/red]")
                return False

    return True


def handle_proxy(args: argparse.Namespace) -> bool:
    """
    Handle the 'proxy' command for multi-model serving.

    Args:
        args: Parsed command line arguments

    Returns:
        True if command executed successfully
    """
    import json
    from pathlib import Path

    # Initialize managers
    config_manager = ProxyConfigManager()

    # Handle proxy subcommands
    if not hasattr(args, "proxy_command") or not args.proxy_command:
        console.print("[yellow]Please specify a proxy command.[/yellow]")
        console.print("Available commands: start, stop, status, add, remove, config")
        return False

    if args.proxy_command == "start":
        # Start proxy server
        console.print("[bold cyan]Starting Multi-Model Proxy Server[/bold cyan]")

        # Load or create configuration
        if hasattr(args, "config") and args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                console.print(f"[red]Config file not found: {config_path}[/red]")
                return False
            proxy_config = config_manager.load_config(config_path)
        elif hasattr(args, "interactive") and args.interactive:
            # Interactive configuration
            from ..ui.proxy.control import configure_proxy_interactively

            proxy_config = configure_proxy_interactively()
            if not proxy_config:
                return False
        else:
            # Use default or saved configuration
            proxy_config = config_manager.load_config()

        # Override host/port only if explicitly specified on command line
        if hasattr(args, "host") and args.host is not None:
            proxy_config.host = args.host
        if hasattr(args, "port") and args.port is not None:
            proxy_config.port = args.port

        # Validate configuration
        errors = config_manager.validate_config(proxy_config)
        if errors:
            console.print("[red]Configuration errors:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            return False

        # Create proxy manager
        proxy_manager = ProxyManager(proxy_config)

        # Auto-allocate GPUs if requested
        if hasattr(args, "auto_allocate") and args.auto_allocate:
            console.print("[cyan]Auto-allocating GPUs to models...[/cyan]")
            allocated = proxy_manager.allocate_gpus_automatically()
            proxy_config.models = allocated

        # Start all models with monitoring
        console.print("\n[cyan]Launching model servers...[/cyan]")
        launched = proxy_manager.start_all_models_no_wait()

        if launched > 0:
            # Monitor startup progress with live logs
            from ..proxy import monitor_startup_progress

            all_started = monitor_startup_progress(proxy_manager)

            if not all_started:
                console.print("[yellow]Warning: Some models failed to start.[/yellow]")
                console.print("[dim]Continuing with available models...[/dim]")

        # Start proxy
        if proxy_manager.start_proxy():
            console.print(
                f"\n[green]✓ Proxy server running at "
                f"http://{proxy_config.host}:{proxy_config.port}[/green]"
            )
            console.print("\nAvailable endpoints:")
            console.print(
                f"  • OpenAI API: http://{proxy_config.host}:{proxy_config.port}/v1/"
            )
            console.print(
                f"  • Models list: http://{proxy_config.host}:{proxy_config.port}/v1/models"
            )
            console.print(
                f"  • Proxy status: http://{proxy_config.host}:{proxy_config.port}/proxy/status"
            )

            if proxy_config.enable_metrics:
                console.print(
                    f"  • Metrics: http://{proxy_config.host}:{proxy_config.port}/metrics"
                )

            console.print("\n[dim]Press Ctrl+C for monitoring options[/dim]")

            # Keep running with monitoring option
            try:
                import time

                time.sleep(2)  # Give servers a moment to fully start

                while True:
                    # Show monitoring menu
                    console.print("\n[bold cyan]Proxy Server Running[/bold cyan]")
                    options = [
                        "Monitor proxy (all options)",
                        "Continue running (background)",
                        "Stop proxy server",
                    ]

                    try:
                        from ..ui.navigation import unified_prompt

                        choice = unified_prompt(
                            "proxy_running", "Select action", options, allow_back=False
                        )

                        if choice == "Monitor proxy (all options)":
                            from ..ui.proxy.monitor import monitor_proxy

                            result = monitor_proxy(proxy_manager)
                            if result == "back":
                                continue  # Return to main proxy menu
                        elif choice == "Continue running (background)":
                            console.print(
                                "\n[green]Proxy continues running in background[/green]"
                            )
                            console.print(
                                f"Access at: http://{proxy_config.host}:{proxy_config.port}"
                            )
                            console.print(
                                "[dim]The proxy will stop when you exit the program[/dim]"
                            )
                            time.sleep(2)
                        elif choice == "Stop proxy server":
                            console.print("\n[yellow]Stopping proxy server...[/yellow]")
                            proxy_manager.stop_proxy()
                            break

                    except KeyboardInterrupt:
                        # Nested Ctrl+C - ask if they want to stop
                        console.print("\n[yellow]Interrupt received[/yellow]")
                        if (
                            unified_prompt(
                                "confirm_stop",
                                "Stop the proxy server?",
                                ["Yes, stop proxy", "No, continue running"],
                                allow_back=False,
                            )
                            == "Yes, stop proxy"
                        ):
                            console.print("\n[yellow]Stopping proxy server...[/yellow]")
                            proxy_manager.stop_proxy()
                            break

            except KeyboardInterrupt:
                console.print("\n[yellow]Stopping proxy server...[/yellow]")
                proxy_manager.stop_proxy()

            return True
        else:
            console.print("[red]Failed to start proxy server[/red]")
            return False

    elif args.proxy_command == "stop":
        # Stop proxy server
        console.print("[yellow]Stopping proxy server and all models...[/yellow]")
        # This would need a way to find and stop the running proxy
        # For now, we'll just inform the user
        console.print("[dim]Use Ctrl+C in the proxy terminal to stop it[/dim]")
        return True

    elif args.proxy_command == "status":
        # Show proxy status
        import httpx

        from ..proxy.runtime import get_proxy_connection

        # Get proxy connection details dynamically
        proxy_host, proxy_port = get_proxy_connection(
            cli_host=getattr(args, "proxy_host", None),
            cli_port=getattr(args, "proxy_port", None),
        )

        try:
            # Try to connect to proxy
            response = httpx.get(
                f"http://{proxy_host}:{proxy_port}/proxy/status", timeout=5
            )
            if response.status_code == 200:
                status = response.json()

                if hasattr(args, "json") and args.json:
                    console.print(json.dumps(status, indent=2))
                else:
                    # Display formatted status
                    console.print("\n[bold cyan]Proxy Server Status[/bold cyan]")
                    console.print("Status: [green]Running[/green]")
                    console.print(
                        f"Address: http://{status['proxy_host']}:{status['proxy_port']}"
                    )
                    console.print(f"Total Requests: {status.get('total_requests', 0)}")

                    if status.get("models"):
                        console.print("\n[bold]Active Models:[/bold]")
                        table = Table()
                        table.add_column("Model", style="cyan")
                        table.add_column("Port", style="magenta")
                        table.add_column("GPUs", style="green")
                        table.add_column("Status", style="yellow")
                        table.add_column("Requests", style="dim")

                        for model in status["models"]:
                            gpu_str = ",".join(str(g) for g in model.get("gpu_ids", []))
                            table.add_row(
                                model["name"],
                                str(model["port"]),
                                gpu_str or "N/A",
                                model["status"],
                                str(model.get("request_count", 0)),
                            )

                        console.print(table)
                    else:
                        console.print("\n[yellow]No models configured[/yellow]")
            else:
                console.print("[red]Proxy server returned error[/red]")

        except httpx.ConnectError:
            console.print(
                f"[yellow]Proxy server is not running at {proxy_host}:{proxy_port}[/yellow]"
            )
            console.print("Start it with: vllm-cli proxy start")
            console.print(
                "[dim]Tip: Use --proxy-host and --proxy-port to specify "
                "a different proxy server[/dim]"
            )
        except Exception as e:
            console.print(f"[red]Error checking proxy status: {e}[/red]")

        return True

    elif args.proxy_command == "add":
        # Add model to running proxy
        if not all([hasattr(args, "name"), hasattr(args, "model_path")]):
            console.print("[red]Model name and path are required[/red]")
            return False

        model_config = ModelConfig(
            name=args.name,
            model_path=args.model_path,
            gpu_ids=args.gpu if hasattr(args, "gpu") and args.gpu else [],
            port=args.port if hasattr(args, "port") and args.port else 8001,
            profile=args.profile if hasattr(args, "profile") else None,
        )

        # Send request to proxy to add model
        import httpx

        from ..proxy.runtime import get_proxy_connection

        # Get proxy connection details dynamically
        proxy_host, proxy_port = get_proxy_connection(
            cli_host=getattr(args, "proxy_host", None),
            cli_port=getattr(args, "proxy_port", None),
        )

        try:
            response = httpx.post(
                f"http://{proxy_host}:{proxy_port}/proxy/add_model",
                json=model_config.dict(),
                timeout=10,
            )
            if response.status_code == 200:
                console.print(
                    f"[green]✓ Model '{args.name}' added successfully[/green]"
                )
            else:
                console.print(f"[red]Failed to add model: {response.text}[/red]")
        except httpx.ConnectError:
            console.print(
                f"[red]Proxy server is not running at {proxy_host}:{proxy_port}[/red]"
            )
        except Exception as e:
            console.print(f"[red]Error adding model: {e}[/red]")

        return True

    elif args.proxy_command == "remove":
        # Remove model from proxy
        if not hasattr(args, "name"):
            console.print("[red]Model name is required[/red]")
            return False

        import httpx

        from ..proxy.runtime import get_proxy_connection

        # Get proxy connection details dynamically
        proxy_host, proxy_port = get_proxy_connection(
            cli_host=getattr(args, "proxy_host", None),
            cli_port=getattr(args, "proxy_port", None),
        )

        try:
            response = httpx.delete(
                f"http://{proxy_host}:{proxy_port}/proxy/remove_model/{args.name}",
                timeout=10,
            )
            if response.status_code == 200:
                console.print(f"[green]✓ Model '{args.name}' removed[/green]")
            else:
                console.print(f"[red]Failed to remove model: {response.text}[/red]")
        except httpx.ConnectError:
            console.print(
                f"[red]Proxy server is not running at {proxy_host}:{proxy_port}[/red]"
            )
        except Exception as e:
            console.print(f"[red]Error removing model: {e}[/red]")

        return True

    elif args.proxy_command == "config":
        # Manage proxy configuration
        if hasattr(args, "create") and args.create:
            # Create example configuration
            output_path = Path(args.create)
            example_config = config_manager.create_example_config()
            config_manager.save_config(example_config, output_path)
            console.print(
                f"[green]✓ Example configuration saved to {output_path}[/green]"
            )

        elif hasattr(args, "edit") and args.edit:
            # Edit configuration interactively
            from ..ui.proxy.control import edit_proxy_config

            edit_proxy_config()

        elif hasattr(args, "export") and args.export:
            # Export current configuration
            output_path = Path(args.export)
            current_config = config_manager.load_config()
            config_manager.export_config(current_config, output_path)
            console.print(f"[green]✓ Configuration exported to {output_path}[/green]")

        else:
            # Show current configuration
            current_config = config_manager.load_config()
            console.print("\n[bold cyan]Current Proxy Configuration[/bold cyan]")
            console.print(f"Host: {current_config.host}")
            console.print(f"Port: {current_config.port}")
            console.print(f"CORS Enabled: {current_config.enable_cors}")
            console.print(f"Metrics Enabled: {current_config.enable_metrics}")
            console.print(f"Request Logging: {current_config.log_requests}")

            if current_config.models:
                console.print(f"\nConfigured Models ({len(current_config.models)}):")
                for model in current_config.models:
                    status = (
                        "[green]enabled[/green]"
                        if model.enabled
                        else "[dim]disabled[/dim]"
                    )
                    console.print(f"  • {model.name}: {model.model_path} ({status})")
            else:
                console.print("\n[yellow]No models configured[/yellow]")

        return True

    else:
        console.print(f"[red]Unknown proxy command: {args.proxy_command}[/red]")
        return False
