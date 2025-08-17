#!/usr/bin/env python3
"""
CLI utility functions and helpers.

Provides common utilities for CLI operations including
argument processing, output formatting, and command coordination.
"""
import argparse
import logging
from typing import Optional

from rich.console import Console

from .handlers import (
    handle_dirs,
    handle_info,
    handle_models,
    handle_serve,
    handle_status,
    handle_stop,
)
from .parser import parse_args

logger = logging.getLogger(__name__)
console = Console()


def handle_cli_command(args: argparse.Namespace) -> bool:
    """
    Route CLI commands to their appropriate handlers.

    Takes parsed command line arguments and dispatches them to the
    correct handler function based on the command specified.

    Args:
        args: Parsed command line arguments

    Returns:
        True if command executed successfully, False otherwise
    """
    try:
        # Map commands to their handlers
        command_handlers = {
            "serve": handle_serve,
            "info": handle_info,
            "models": handle_models,
            "status": handle_status,
            "stop": handle_stop,
            "dirs": handle_dirs,
        }

        command = args.command
        if not command:
            # No command specified - this should trigger interactive mode
            return False

        handler = command_handlers.get(command)
        if not handler:
            console.print(f"[red]Unknown command: {command}[/red]")
            return False

        # Call the appropriate handler
        if command in ["info", "models", "status"]:
            # Commands that don't need arguments
            return handler()
        else:
            # Commands that need the args namespace
            return handler(args)

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        return False
    except Exception as e:
        logger.exception(f"Error handling CLI command: {e}")
        console.print(f"[red]Command failed: {e}[/red]")
        return False


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for CLI mode.

    Parses command line arguments and executes the appropriate command.
    If no command is specified, returns False to indicate interactive
    mode should be started.

    Args:
        argv: Optional command line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 for success, 1 for failure, 2 for interactive mode)
    """
    try:
        # Parse arguments
        args = parse_args(argv)

        # If no command specified, indicate interactive mode should start
        if not args.command:
            return 2  # Special code for interactive mode

        # Handle the command
        success = handle_cli_command(args)
        return 0 if success else 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error in CLI: {e}")
        console.print(f"[red]Unexpected error: {e}[/red]")
        return 1


def validate_cli_args(args: argparse.Namespace) -> bool:
    """
    Validate CLI arguments for consistency and correctness.

    Performs additional validation beyond what argparse can handle,
    such as checking mutually exclusive options and value ranges.

    Args:
        args: Parsed command line arguments

    Returns:
        True if arguments are valid, False otherwise
    """
    try:
        # Validate serve command arguments
        if args.command == "serve":
            if hasattr(args, "gpu_memory_utilization") and args.gpu_memory_utilization:
                if not (0.0 <= args.gpu_memory_utilization <= 1.0):
                    console.print(
                        "[red]Error: gpu-memory-utilization must be between 0.0 and 1.0[/red]"
                    )
                    return False

            if hasattr(args, "tensor_parallel_size") and args.tensor_parallel_size:
                if args.tensor_parallel_size < 1:
                    console.print("[red]Error: tensor-parallel-size must be >= 1[/red]")
                    return False

            if hasattr(args, "port") and args.port:
                if not (1 <= args.port <= 65535):
                    console.print("[red]Error: port must be between 1 and 65535[/red]")
                    return False

        # Validate stop command arguments
        elif args.command == "stop":
            # Check that exactly one stop method is specified
            stop_methods = [
                bool(getattr(args, "model", None)),
                bool(getattr(args, "all", False)),
                bool(getattr(args, "port", None)),
            ]

            if sum(stop_methods) != 1:
                console.print(
                    "[red]Error: specify exactly one of: model name, --all, or --port[/red]"
                )
                return False

        return True

    except Exception as e:
        logger.exception(f"Error validating CLI arguments: {e}")
        console.print(f"[red]Argument validation error: {e}[/red]")
        return False


def format_cli_output(data: dict, format_type: str = "table") -> str:
    """
    Format output data for CLI display.

    Args:
        data: Data to format
        format_type: Output format ('table', 'json', 'plain')

    Returns:
        Formatted string
    """
    import json as json_module

    if format_type == "json":
        return json_module.dumps(data, indent=2)
    elif format_type == "plain":
        # Simple key-value format
        lines = []
        for key, value in data.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)
    else:
        # Default to rich table formatting (handled by individual commands)
        return str(data)


def get_user_confirmation(message: str, default: bool = False) -> bool:
    """
    Get user confirmation for an action.

    Args:
        message: Message to display to user
        default: Default response if user just presses Enter

    Returns:
        True if user confirmed, False otherwise
    """
    default_str = "Y/n" if default else "y/N"
    try:
        response = input(f"{message} [{default_str}]: ").strip().lower()

        if not response:
            return default

        return response in ["y", "yes", "true", "1"]

    except (KeyboardInterrupt, EOFError):
        return False


def print_cli_error(error: str) -> None:
    """
    Print an error message in CLI format.

    Args:
        error: Error message to display
    """
    console.print(f"[red]Error: {error}[/red]")


def print_cli_warning(warning: str) -> None:
    """
    Print a warning message in CLI format.

    Args:
        warning: Warning message to display
    """
    console.print(f"[yellow]Warning: {warning}[/yellow]")


def print_cli_success(message: str) -> None:
    """
    Print a success message in CLI format.

    Args:
        message: Success message to display
    """
    console.print(f"[green]{message}[/green]")


def print_cli_info(message: str) -> None:
    """
    Print an info message in CLI format.

    Args:
        message: Info message to display
    """
    console.print(f"[blue]{message}[/blue]")
