#!/usr/bin/env python3
"""
Minimal integration with hf-model-tool for model management.

This module provides subprocess launchers for hf-model-tool,
delegating all model management functionality to the external tool.
"""
import logging
import os
import subprocess

from rich.console import Console
from rich.panel import Panel

logger = logging.getLogger(__name__)
console = Console()


def launch_hf_model_tool(args: list = None) -> None:
    """
    Launch hf-model-tool with optional arguments.

    Args:
        args: Optional list of command-line arguments for hf-model-tool
    """
    cmd = ["hf-model-tool"]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(cmd, env=os.environ.copy())
        if result.returncode != 0:
            console.print("[yellow]hf-model-tool exited with an error.[/yellow]")
    except FileNotFoundError:
        console.print("[red]hf-model-tool not found. Please install it:[/red]")
        console.print("  pip install hf-model-tool")
    except Exception as e:
        console.print(f"[red]Error launching hf-model-tool: {e}[/red]")


def check_hf_model_tool_installed() -> bool:
    """
    Check if hf-model-tool is installed and available.

    Returns:
        True if hf-model-tool is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["hf-model-tool", "--version"],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except Exception:
        return False


# Legacy function names for backward compatibility (will be removed in future)
def launch_hf_model_tool_interactive() -> str:
    """Legacy function - launches hf-model-tool in interactive mode."""
    console.clear()
    console.print(
        Panel(
            "[bold cyan]Launching HF-Model-Tool[/bold cyan]\n"
            "[dim]Full model management interface[/dim]",
            border_style="blue",
        )
    )
    launch_hf_model_tool()
    input("\nPress Enter to continue...")
    return "continue"


def launch_hf_model_tool_manage() -> str:
    """Legacy function - launches hf-model-tool in manage mode."""
    console.clear()
    console.print(
        Panel(
            "[bold cyan]Asset Management[/bold cyan]\n"
            "[dim]Delete, deduplicate, and organize models[/dim]",
            border_style="blue",
        )
    )
    launch_hf_model_tool(["--manage"])
    input("\nPress Enter to continue...")
    return "continue"
