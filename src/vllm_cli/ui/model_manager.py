#!/usr/bin/env python3
"""
Model management module for vLLM CLI.

Handles model selection, listing, and information display.
"""
import json
import logging
from typing import Optional
from rich.table import Table

from ..models import list_available_models, get_model_details
from .navigation import unified_prompt
from ..system import format_size
from .common import console, create_panel

logger = logging.getLogger(__name__)


def select_model() -> Optional[str]:
    """
    Select a model from available models with provider categorization.
    """
    console.print("\n[bold cyan]Fetching available models...[/bold cyan]")

    try:
        models = list_available_models()

        if not models:
            console.print("[yellow]No models found.[/yellow]")
            console.print("Please download models using HuggingFace tools first.")
            input("\nPress Enter to continue...")
            return None

        # Group models by provider
        providers_dict = {}
        for model in models:
            provider = model.get("publisher", "unknown")
            if provider == "unknown" or not provider:
                # Try to extract provider from model name
                if "/" in model["name"]:
                    provider = model["name"].split("/")[0]
                else:
                    provider = "local"

            if provider not in providers_dict:
                providers_dict[provider] = []
            providers_dict[provider].append(model)

        # Sort providers alphabetically
        sorted_providers = sorted(providers_dict.keys())

        # First, select provider
        provider_choices = []
        for provider in sorted_providers:
            count = len(providers_dict[provider])
            provider_choices.append(
                f"{provider} ({count} model{'s' if count > 1 else ''})"
            )

        selected_provider = unified_prompt(
            "provider",
            f"Select Provider ({len(sorted_providers)} available)",
            provider_choices,
            allow_back=True,
        )

        if not selected_provider or selected_provider == "BACK":
            return None

        # Extract provider name
        provider_name = selected_provider.split(" (")[0]

        # Now show models for selected provider
        provider_models = providers_dict[provider_name]

        # Create model choices for selected provider
        model_choices = []
        for model in provider_models:
            size_str = format_size(model.get("size", 0))
            # Show only the model name without provider if it's already in the name
            display_name = model["name"]
            if display_name.startswith(f"{provider_name}/"):
                display_name = display_name[len(provider_name) + 1 :]
            model_choices.append(f"{display_name} ({size_str})")

        # Show model selection for the provider
        selected = unified_prompt(
            "model",
            f"Select {provider_name} Model ({len(provider_models)} available)",
            model_choices,
            allow_back=True,
        )

        if not selected or selected == "BACK":
            # Go back to provider selection
            return select_model()

        # Extract model name and reconstruct full name if needed
        model_display_name = selected.split(" (")[0]

        # Find the full model name
        for model in provider_models:
            check_name = model["name"]
            if check_name.startswith(f"{provider_name}/"):
                check_name = check_name[len(provider_name) + 1 :]
            if check_name == model_display_name or model["name"] == model_display_name:
                return model["name"]

        # Fallback
        return model_display_name

    except Exception as e:
        logger.error(f"Error selecting model: {e}")
        console.print(f"[red]Error selecting model: {e}[/red]")
        return None


def handle_model_management() -> str:
    """
    Handle model management operations.
    """
    models = list_available_models()

    if not models:
        console.print("[yellow]No models found.[/yellow]")
        input("\nPress Enter to continue...")
        return "continue"

    # Display models table
    table = Table(
        title=f"[bold green]Available Models ({len(models)} total)[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Model Name", style="cyan")
    table.add_column("Size", style="magenta", justify="right")
    table.add_column("Type", style="yellow")
    table.add_column("Path", style="dim white")

    total_size = 0
    for model in models:
        size = model.get("size", 0)
        total_size += size
        table.add_row(
            model["name"],
            format_size(size),
            model.get("type", "unknown"),
            model.get("path", "N/A"),
        )

    console.print(table)
    console.print(f"\n[bold]Total size: {format_size(total_size)}[/bold]")

    # Model actions
    actions = ["View Model Details", "Refresh Model List"]
    action = unified_prompt("model_action", "Model Actions", actions, allow_back=True)

    if action == "View Model Details":
        model_name = select_model()
        if model_name:
            show_model_details(model_name)
    elif action == "Refresh Model List":
        console.print("[cyan]Refreshing model list...[/cyan]")
        list_available_models(refresh=True)
        return handle_model_management()

    return "continue"


def show_model_details(model_name: str) -> None:
    """
    Show detailed information about a model.
    """
    try:
        details = get_model_details(model_name)

        if not details:
            console.print(f"[yellow]No details found for {model_name}[/yellow]")
            return

        console.print(f"\n[bold cyan]Model Details: {model_name}[/bold cyan]")

        # Create details table
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in details.items():
            if isinstance(value, dict):
                value = json.dumps(value, indent=2)
            elif isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(create_panel(table, border_style="blue"))

    except Exception as e:
        logger.error(f"Error showing model details: {e}")
        console.print(f"[red]Error: {e}[/red]")

    input("\nPress Enter to continue...")
