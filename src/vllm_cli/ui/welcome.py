#!/usr/bin/env python3
"""
Welcome screen module for vLLM CLI.

Displays ASCII art logo, system information, and features overview.
"""
import logging

from rich.align import Align
from rich.text import Text

from .. import __version__
from .common import console, create_panel
from .components import create_system_overview_panel
from .gpu_utils import create_gpu_status_panel

logger = logging.getLogger(__name__)


def show_welcome_screen() -> None:
    """
    Display a welcome screen with ASCII art and system info.
    """
    logger.info("Displaying welcome screen")

    try:
        # ASCII art logo for vLLM CLI (with lowercase v)
        logo = """
         ██      ██      ███    ███       ██████ ██      ██
         ██      ██      ████  ████      ██      ██      ██
██    ██ ██      ██      ██ ████ ██      ██      ██      ██
 ██  ██  ██      ██      ██  ██  ██      ██      ██      ██
  ████   ███████ ███████ ██      ██       ██████ ███████ ██
"""

        # Create colored logo
        logo_text = Text(logo, style="bold cyan")

        # Subtitle and version
        subtitle = Text(
            "vLLM CLI - Convenient vLLM Serving Tool",
            style="bold yellow",
            justify="center",
        )
        version_text = Text(f"v{__version__}", style="dim white", justify="center")
        tagline = Text(
            "Serve • Configure • Monitor Your LLMs",
            style="italic green",
            justify="center",
        )

        # Get GPU and system panels using shared components
        gpu_panel = create_gpu_status_panel()
        system_panel = create_system_overview_panel()

        # Display the welcome screen
        centered_logo = Align.center(logo_text)
        console.print(
            create_panel(centered_logo, border_style="bright_blue", padding=(1, 2))
        )
        console.print(Align.center(subtitle))
        console.print(Align.center(version_text))
        console.print(Align.center(tagline))
        console.print("")

        # Display GPU panel
        console.print(gpu_panel)

        # Display system panel if available
        if system_panel:
            console.print("")
            console.print(system_panel)

        console.print("")
        console.print(
            create_panel(
                "[bold white]Press Enter to continue...[/bold white]",
                style="dim",
                border_style="dim",
            )
        )

        # Wait for user input
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            logger.info("User interrupted welcome screen")

    except Exception as e:
        logger.error(f"Error displaying welcome screen: {e}")
        console.print(f"[red]Error displaying welcome screen: {e}[/red]")
        console.print("[yellow]Continuing to main menu...[/yellow]")
