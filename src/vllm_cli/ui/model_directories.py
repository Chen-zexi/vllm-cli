#!/usr/bin/env python3
"""
Model directories management UI for vLLM CLI.

Provides an interface for managing model directories using hf-model-tool API.
"""
import logging
from pathlib import Path

from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .common import console
from .navigation import unified_prompt

logger = logging.getLogger(__name__)


class ModelDirectoriesUI:
    """UI component for managing model directories."""

    def __init__(self):
        """Initialize the Model Directories UI."""
        self.api = None
        self._init_api()

    def _init_api(self):
        """Initialize the hf-model-tool API."""
        try:
            import sys

            # Add hf-model-tool to path if needed
            hf_tool_path = Path("/home/chen/hf-model-tool")
            if hf_tool_path.exists() and str(hf_tool_path) not in sys.path:
                sys.path.insert(0, str(hf_tool_path))

            from hf_model_tool.api import HFModelAPI

            self.api = HFModelAPI()
            logger.info("Successfully initialized hf-model-tool API")
        except ImportError as e:
            logger.error(f"Failed to import hf-model-tool API: {e}")
            self.api = None
        except Exception as e:
            logger.error(f"Error initializing hf-model-tool API: {e}")
            self.api = None

    def show(self) -> str:
        """
        Show the model directories management interface.

        Returns:
            Action to take after directory management
        """
        if not self.api:
            console.print(
                Panel.fit(
                    "[bold red]Error[/bold red]\n"
                    "[yellow]hf-model-tool is not available.[/yellow]\n\n"
                    "Please ensure hf-model-tool is installed or updated:\n"
                    "  [cyan]pip install --upgrade hf-model-tool[/cyan]",
                    border_style="red",
                )
            )
            input("\nPress Enter to continue...")
            return "continue"

        while True:
            # Show header
            console.print(
                Panel.fit(
                    "[bold cyan]Model Directory Management[/bold cyan]\n"
                    "[dim]Configure directories for model discovery[/dim]",
                    border_style="blue",
                )
            )

            # Display current directories
            self._display_directories()

            # Show menu options
            options = [
                "Add Directory",
                "Remove Directory",
                "Toggle Ollama Scanning",
                "Scan All Directories",
                "View Directory Statistics",
            ]

            action = unified_prompt(
                "model_directories", "Directory Management", options, allow_back=True
            )

            if action == "← Back" or action == "BACK" or not action:
                return "continue"
            elif action == "Add Directory":
                self._add_directory()
            elif action == "Remove Directory":
                self._remove_directory()
            elif action == "Toggle Ollama Scanning":
                self._toggle_ollama_scanning()
            elif action == "Scan All Directories":
                self._scan_directories()
            elif action == "View Directory Statistics":
                self._show_statistics()

    def _display_directories(self):
        """Display currently configured directories."""
        try:
            directories = self.api.list_directories()

            # Get Ollama status
            ollama_status = self.api.get_ollama_status()
            ollama_enabled = ollama_status.get("scan_enabled", False)

            # Display Ollama scanning status
            if ollama_enabled:
                console.print(
                    "\n[bold]Ollama Scanning:[/bold] [green]✓ Enabled[/green]"
                )
            else:
                console.print("\n[bold]Ollama Scanning:[/bold] [red]✗ Disabled[/red]")

            if not directories:
                console.print("\n[yellow]No directories configured.[/yellow]")
                console.print("[dim]Using default locations only.[/dim]\n")
                return

            # Create table for directories
            table = Table(
                title="[bold]Configured Directories[/bold]",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("#", style="cyan", width=3)
            table.add_column("Path", style="white")
            table.add_column("Type", style="yellow", width=15)
            table.add_column("Source", style="magenta", width=15)
            table.add_column("Status", style="green", width=10)

            for idx, dir_info in enumerate(directories, 1):
                path = dir_info.get("path", "")
                dir_type = dir_info.get("type", "custom")

                dir_source = dir_info.get("source", "unknown")

                # Format source for display
                if dir_source == "default_cache":
                    source_display = "Default HF"
                elif dir_source == "default_ollama":
                    source_display = "Default Ollama"
                elif dir_source == "custom_ollama":
                    source_display = "Custom Ollama"
                elif dir_source.startswith("custom"):
                    source_display = "Custom"
                else:
                    source_display = dir_source.capitalize()

                # Check if directory exists
                path_obj = Path(path)
                status = (
                    "[green]✓ Valid[/green]"
                    if path_obj.exists()
                    else "[red]✗ Missing[/red]"
                )

                table.add_row(
                    str(idx), path, dir_type.capitalize(), source_display, status
                )

            console.print(table)
            console.print()

        except Exception as e:
            logger.error(f"Error displaying directories: {e}")
            console.print(f"[red]Error loading directories: {e}[/red]\n")

    def _add_directory(self):
        """Add a new directory for model scanning."""
        console.print("\n[bold cyan]Add Model Directory[/bold cyan]")

        # Get directory path
        console.print("\nEnter the full path to the directory containing models:")
        console.print("[dim]Example: /home/user/models or ~/my_models[/dim]")
        path = input("Path: ").strip()

        if not path:
            console.print("[yellow]No path provided, cancelling.[/yellow]")
            input("\nPress Enter to continue...")
            return

        # Expand user path
        path = str(Path(path).expanduser().resolve())

        # Validate directory exists
        path_obj = Path(path)
        if not path_obj.exists():
            # Use unified prompt for confirmation
            create_choices = ["Yes, create it", "No, cancel"]
            create_choice = unified_prompt(
                "create_dir",
                f"Directory '{path}' does not exist. Create it?",
                create_choices,
                allow_back=False,
            )

            if create_choice == "Yes, create it":
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    console.print(f"\n[green]Created directory: {path}[/green]")
                except Exception as e:
                    console.print(f"\n[red]Failed to create directory: {e}[/red]")
                    input("\nPress Enter to continue...")
                    return
            else:
                console.print("\n[yellow]Directory must exist. Cancelling.[/yellow]")
                input("\nPress Enter to continue...")
                return

        # Ask for directory type using unified prompt
        type_choices = [
            "Auto-detect (recommended)",
            "HuggingFace cache",
            "Custom models",
            "LoRA adapters",
            "Ollama models",
        ]

        type_selection = unified_prompt(
            "dir_type", "Select directory type", type_choices, allow_back=False
        )

        type_map = {
            "Auto-detect (recommended)": "auto",
            "HuggingFace cache": "huggingface",
            "Custom models": "custom",
            "LoRA adapters": "lora",
            "Ollama models": "ollama",
        }
        dir_type = type_map.get(type_selection, "auto")

        # Special handling for Ollama directories
        if dir_type == "ollama":
            # Check for Ollama structure
            has_manifests = (Path(path) / "manifests").exists()
            has_blobs = (Path(path) / "blobs").exists()

            if not (has_manifests and has_blobs):
                console.print(
                    "\n[yellow]Warning: Directory doesn't have standard Ollama structure[/yellow]"
                )
                console.print(f"  Manifests directory: {'✓' if has_manifests else '✗'}")
                console.print(f"  Blobs directory: {'✓' if has_blobs else '✗'}")

                # Ask if they want to add it anyway
                confirm_choices = ["Yes, add anyway", "No, cancel"]
                confirm_choice = unified_prompt(
                    "confirm_ollama",
                    "Add as Ollama directory anyway?",
                    confirm_choices,
                    allow_back=False,
                )

                if confirm_choice != "Yes, add anyway":
                    console.print("\n[yellow]Cancelled[/yellow]")
                    input("\nPress Enter to continue...")
                    return

            # Add as Ollama directory
            try:
                success = self.api.add_ollama_directory(path)
                if success:
                    console.print(
                        f"\n[green]✓ Successfully added Ollama directory:[/green] {path}"
                    )

                    # Enable Ollama scanning if not already enabled
                    ollama_status = self.api.get_ollama_status()
                    if not ollama_status.get("scan_enabled", False):
                        console.print(
                            "\n[yellow]Note: Ollama scanning is currently disabled[/yellow]"
                        )
                        enable_choices = ["Yes, enable now", "No, keep disabled"]
                        enable_choice = unified_prompt(
                            "enable_ollama",
                            "Enable Ollama scanning?",
                            enable_choices,
                            allow_back=False,
                        )

                        if enable_choice == "Yes, enable now":
                            self.api.toggle_ollama_scanning()
                            console.print("[green]✓ Ollama scanning enabled[/green]")
                else:
                    console.print(
                        "\n[red]Failed to add Ollama directory (may already exist)[/red]"
                    )
            except Exception as e:
                console.print(f"\n[red]Error adding Ollama directory: {e}[/red]")

            input("\nPress Enter to continue...")
            return

        # Add regular directory
        try:
            success = self.api.add_directory(path, dir_type)
            if success:
                console.print(
                    f"\n[green]✓ Successfully added directory:[/green] {path}"
                )

                # Inform about manifest file
                manifest_path = Path(path) / "models_manifest.json"
                console.print(
                    Panel.fit(
                        "[bold yellow]Note about Model Manifest:[/bold yellow]\n\n"
                        f"A [cyan]models_manifest.json[/cyan] file will be auto-generated at:\n"
                        f"[dim]{manifest_path}[/dim]\n\n"
                        "You can manually edit this file to customize:\n"
                        "  • Custom display names\n"
                        "  • Model descriptions\n"
                        "  • Publisher information\n"
                        "  • Model categories\n\n"
                        "[dim]The manifest helps organize and customize how models appear.[/dim]",
                        border_style="yellow",
                    )
                )

                # Offer to scan immediately using unified prompt
                scan_choices = ["Yes, scan now", "No, scan later"]
                scan_choice = unified_prompt(
                    "scan_now",
                    "Scan this directory for models now?",
                    scan_choices,
                    allow_back=False,
                )

                if scan_choice == "Yes, scan now":
                    self._scan_single_directory(path)

                    # After scanning, remind about manifest if models were found
                    console.print(
                        "\n[dim]Tip: A models_manifest.json file has been auto-generated.[/dim]"
                    )
                    console.print(
                        "[dim]You can edit it to customize how models appear in the serving menu.[/dim]"
                    )
            else:
                console.print("\n[red]Failed to add directory.[/red]")
        except Exception as e:
            console.print(f"\n[red]Error adding directory: {e}[/red]")

        input("\nPress Enter to continue...")

    def _remove_directory(self):
        """Remove a directory from scanning."""
        directories = self.api.list_directories()

        if not directories:
            console.print("\n[yellow]No directories to remove.[/yellow]")
            input("\nPress Enter to continue...")
            return

        console.print("\n[bold cyan]Remove Directory[/bold cyan]")

        # Build choices list with directory paths
        dir_choices = []
        for dir_info in directories:
            path = dir_info.get("path", "")
            dir_type = dir_info.get("type", "custom")
            dir_choices.append(f"{path} [{dir_type}]")

        # Use unified prompt for selection
        selected = unified_prompt(
            "remove_dir", "Select directory to remove", dir_choices, allow_back=True
        )

        if not selected or selected == "BACK":
            return

        # Extract the path from the selection
        selected_path = selected.split(" [")[0]  # Remove the [type] suffix

        # Find the matching directory
        for dir_info in directories:
            if dir_info.get("path", "") == selected_path:
                # Confirm removal using unified prompt
                confirm_choices = ["Yes, remove this directory", "No, keep it"]
                confirm = unified_prompt(
                    "confirm_remove",
                    f"Remove {selected_path}?",
                    confirm_choices,
                    allow_back=False,
                )

                if confirm == "Yes, remove this directory":
                    try:
                        # Check if it's an Ollama directory
                        dir_source = dir_info.get("source", "")
                        if dir_info.get("type") == "ollama" and "ollama" in dir_source:
                            # Remove as Ollama directory
                            success = self.api.remove_ollama_directory(selected_path)
                        else:
                            # Remove as regular directory
                            success = self.api.remove_directory(selected_path)

                        if success:
                            console.print(
                                f"\n[green]✓ Removed directory: {selected_path}[/green]"
                            )
                        else:
                            console.print("\n[red]Failed to remove directory.[/red]")
                    except Exception as e:
                        console.print(f"\n[red]Error removing directory: {e}[/red]")
                else:
                    console.print("\n[yellow]Directory not removed.[/yellow]")
                break

        input("\nPress Enter to continue...")

    def _scan_directories(self):
        """Scan all configured directories for models."""
        console.print("\n[bold cyan]Scanning Directories[/bold cyan]")
        console.print("[dim]This may take a moment for large directories...[/dim]\n")

        try:
            # Force refresh to get latest data
            assets = self.api.list_assets(force_refresh=True)

            # Group by type
            models = [a for a in assets if a.get("type") == "model"]
            custom_models = [a for a in assets if a.get("type") == "custom_model"]
            lora_adapters = [a for a in assets if a.get("type") == "lora_adapter"]
            datasets = [a for a in assets if a.get("type") == "dataset"]

            # Display summary
            console.print("[bold]Scan Results:[/bold]")
            console.print(f"  Models: [cyan]{len(models)}[/cyan]")
            console.print(f"  Custom Models: [cyan]{len(custom_models)}[/cyan]")
            console.print(f"  LoRA Adapters: [cyan]{len(lora_adapters)}[/cyan]")
            console.print(f"  Datasets: [cyan]{len(datasets)}[/cyan]")
            console.print(f"  [bold]Total Assets: [green]{len(assets)}[/green][/bold]")

            # Show top models by size
            if models or custom_models:
                all_models = models + custom_models
                all_models.sort(key=lambda x: x.get("size", 0), reverse=True)

                console.print("\n[bold]Top Models by Size:[/bold]")
                for model in all_models[:5]:
                    name = model.get("display_name", model.get("name", "Unknown"))
                    size = model.get("size", 0)
                    size_gb = size / (1024**3)
                    console.print(f"  • {name}: [yellow]{size_gb:.2f} GB[/yellow]")

        except Exception as e:
            console.print(f"[red]Error scanning directories: {e}[/red]")

        input("\nPress Enter to continue...")

    def _scan_single_directory(self, path: str):
        """Scan a single directory for models."""
        console.print(f"\n[cyan]Scanning: {path}[/cyan]")

        try:
            assets = self.api.scan_directories([path])

            if assets:
                console.print(f"[green]Found {len(assets)} asset(s):[/green]")
                for asset in assets[:5]:  # Show first 5
                    name = asset.get("display_name", asset.get("name", "Unknown"))
                    asset_type = asset.get("type", "unknown")
                    console.print(f"  • {name} ([yellow]{asset_type}[/yellow])")
                if len(assets) > 5:
                    console.print(f"  [dim]... and {len(assets) - 5} more[/dim]")
            else:
                console.print("[yellow]No models found in this directory.[/yellow]")

        except Exception as e:
            console.print(f"[red]Error scanning directory: {e}[/red]")

    def _toggle_ollama_scanning(self):
        """Toggle Ollama model scanning on/off."""
        console.print("\n[bold cyan]Ollama Model Scanning[/bold cyan]")

        try:
            # Get current status
            ollama_status = self.api.get_ollama_status()
            current_state = ollama_status.get("scan_enabled", False)

            # Display current state
            if current_state:
                console.print(
                    "\nOllama scanning is currently: [green]✓ Enabled[/green]"
                )
                console.print("\nDefault Ollama directories being scanned:")
                for dir_path in ollama_status.get("default_directories", []):
                    if Path(dir_path).exists():
                        console.print(f"  • {dir_path}")
            else:
                console.print("\nOllama scanning is currently: [red]✗ Disabled[/red]")
                console.print("\n[dim]Enable to scan Ollama model directories[/dim]")

            # Ask to toggle
            toggle_choices = [
                "Yes, toggle it" if current_state else "Yes, enable it",
                "No, keep current setting",
            ]

            toggle_choice = unified_prompt(
                "toggle_ollama",
                f"{'Disable' if current_state else 'Enable'} Ollama scanning?",
                toggle_choices,
                allow_back=False,
            )

            if toggle_choice and toggle_choice.startswith("Yes"):
                new_state = self.api.toggle_ollama_scanning()

                if new_state:
                    console.print("\n[green]✓ Ollama scanning enabled[/green]")
                    console.print(
                        "[dim]Ollama models will now be included in scans[/dim]"
                    )
                else:
                    console.print("\n[yellow]Ollama scanning disabled[/yellow]")
                    console.print(
                        "[dim]Ollama models will be excluded from scans[/dim]"
                    )

                # Offer to refresh cache
                refresh_choices = ["Yes, refresh now", "No, refresh later"]
                refresh_choice = unified_prompt(
                    "refresh_cache",
                    "Refresh model cache to apply changes?",
                    refresh_choices,
                    allow_back=False,
                )

                if refresh_choice == "Yes, refresh now":
                    console.print("\n[cyan]Refreshing model cache...[/cyan]")

                    # Import and use model manager to refresh
                    try:
                        from ..models import get_model_manager

                        model_manager = get_model_manager()
                        model_manager.refresh_cache()
                        console.print("[green]✓ Model cache refreshed[/green]")
                    except Exception as e:
                        console.print(f"[red]Failed to refresh cache: {e}[/red]")
            else:
                console.print("\n[yellow]Settings unchanged[/yellow]")

        except Exception as e:
            console.print(f"\n[red]Error toggling Ollama scanning: {e}[/red]")

        input("\nPress Enter to continue...")

    def _show_statistics(self):
        """Show statistics about managed assets."""
        console.print("\n[bold cyan]Asset Statistics[/bold cyan]\n")

        try:
            stats = self.api.get_statistics()

            # Create statistics panels
            panels = []

            # Models panel
            model_text = Text()
            model_text.append("Models\n", style="bold yellow")
            model_text.append(f"Total: {stats.get('total_models', 0)}\n")
            model_text.append(f"Custom: {stats.get('custom_models', 0)}\n")
            model_text.append(f"LoRA: {stats.get('lora_adapters', 0)}")
            panels.append(Panel(model_text, border_style="yellow"))

            # Storage panel
            storage_text = Text()
            storage_text.append("Storage\n", style="bold cyan")
            total_size = stats.get("total_size", 0)
            size_gb = total_size / (1024**3)
            storage_text.append(f"Total: {size_gb:.2f} GB\n")
            storage_text.append(f"Datasets: {stats.get('dataset_count', 0)}")
            panels.append(Panel(storage_text, border_style="cyan"))

            # Directories panel
            dir_text = Text()
            dir_text.append("Directories\n", style="bold green")
            dir_text.append(f"Monitored: {stats.get('directories', 0)}\n")
            dir_text.append("Last scan: Recent")
            panels.append(Panel(dir_text, border_style="green"))

            console.print(Columns(panels))

            # Show breakdown by directory if available
            if "by_directory" in stats:
                console.print("\n[bold]By Directory:[/bold]")
                for dir_path, dir_stats in stats["by_directory"].items():
                    console.print(f"\n  [cyan]{dir_path}[/cyan]")
                    console.print(f"    Models: {dir_stats.get('models', 0)}")
                    console.print(
                        f"    Size: {dir_stats.get('size', 0) / (1024**3):.2f} GB"
                    )

        except Exception as e:
            console.print(f"[red]Error getting statistics: {e}[/red]")

        input("\nPress Enter to continue...")


def manage_model_directories() -> str:
    """
    Main entry point for model directory management.

    Returns:
        Action to take after management
    """
    ui = ModelDirectoriesUI()
    return ui.show()
