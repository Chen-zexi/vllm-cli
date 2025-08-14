#!/usr/bin/env python3
"""
vLLM CLI: Main entry point for the application

Provides both CLI and interactive modes for managing vLLM servers
"""
import sys
import logging
import argparse
from pathlib import Path
from typing import NoReturn, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.columns import Columns

from . import __version__
from .ui import show_welcome_screen, show_main_menu
from .cli import create_parser, handle_cli_command
from .config import ConfigManager
from .system import check_system_requirements

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(Path.home() / ".vllm-cli.log")],
)
logger = logging.getLogger(__name__)


def main() -> NoReturn:
    """
    Main application entry point.

    Handles both CLI arguments and interactive mode.
    """
    logger.info("Starting vLLM CLI application")
    console = Console()

    try:
        # Check system requirements
        if not check_system_requirements():
            console.print(
                "[red]System requirements not met. Please check the log for details.[/red]"
            )
            sys.exit(1)

        # Parse command line arguments
        parser = create_parser()
        args = parser.parse_args()

        # Handle CLI mode if arguments provided
        if len(sys.argv) > 1:
            result = handle_cli_command(args)
            if result:
                sys.exit(0)
            else:
                sys.exit(1)

        # Interactive mode
        logger.info("Entering interactive mode")

        # Initialize configuration
        config_manager = ConfigManager()

        # Show welcome screen
        show_welcome_screen()

        # Main menu loop
        while True:
            try:
                action = show_main_menu()

                if action == "quit":
                    # Show cleanup and goodbye message
                    console.print("")

                    # Check for active servers and show cleanup status
                    from .server import get_active_servers

                    active_servers = get_active_servers()

                    if active_servers:
                        cleanup_message = (
                            "[bold cyan]Thanks for using vLLM CLI![/bold cyan]\n\n"
                            f"[yellow]⏳ Cleaning up {len(active_servers)} active server(s)...[/yellow]\n"
                            "[dim white]This may take a moment for large models[/dim white]\n"
                            "[dim white]Please wait while services are terminated gracefully[/dim white]"
                        )
                    else:
                        cleanup_message = (
                            "[bold cyan]Thanks for using vLLM CLI![/bold cyan]\n\n"
                            "[green]✓ All services cleaned up[/green]\n"
                            "[dim white]Session ended successfully[/dim white]"
                        )

                    console.print(
                        Panel(
                            cleanup_message,
                            style="dim",
                            border_style="blue",
                        )
                    )
                    logger.info("User quit application")
                    break

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt in main loop")
                console.print("\n[yellow]Returning to main menu...[/yellow]")
                continue
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                console.print(f"[red]Error: {e}[/red]")
                console.print("[yellow]Returning to main menu...[/yellow]")
                continue

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        console.print("\n[yellow]Application interrupted by user[/yellow]")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)
    finally:
        logger.info("Application terminated")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        sys.exit(0)
