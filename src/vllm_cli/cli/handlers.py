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

from ..config import ConfigManager
from ..models import list_available_models
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
        software_info = "vLLM CLI: 0.1.0\n"

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
                status_str = "🟢 Running"
            else:
                status_str = "🔴 Stopped"

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
