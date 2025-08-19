#!/usr/bin/env python3
"""
Shortcut management UI for vLLM CLI.

Handles creation, editing, and deletion of shortcuts which are
saved combinations of model + profile.
"""
import logging
from pathlib import Path

import inquirer

from ..config import ConfigManager
from .common import console
from .display import display_config
from .model_manager import select_model
from .navigation import unified_prompt
from .server_control import select_profile

logger = logging.getLogger(__name__)


def manage_shortcuts() -> str:
    """
    Main shortcut management interface.

    Returns:
        "continue" to return to main menu
    """
    config_manager = ConfigManager()

    while True:
        # Display existing shortcuts
        console.print("\n[bold cyan]Shortcut Management[/bold cyan]")
        console.print(
            "[dim]Shortcuts are saved combinations of model + profile for quick launching[/dim]\n"
        )

        shortcuts = config_manager.list_shortcuts()

        if shortcuts:
            console.print("[bold]Your Shortcuts:[/bold]")
            for shortcut in shortcuts:
                name = shortcut["name"]
                model = shortcut["model"]
                profile = shortcut["profile"]
                desc = shortcut.get("description", "")

                # Truncate long model paths for display
                if len(model) > 40:
                    model_display = "..." + model[-37:]
                else:
                    model_display = model

                console.print(f"  → {name}")
                console.print(f"    Model: {model_display} | Profile: {profile}")
                if desc:
                    console.print(f"    [dim]{desc}[/dim]")
        else:
            console.print("[dim]No shortcuts configured yet[/dim]")

        # Action menu
        actions = [
            "Create New Shortcut",
            "View/Edit Shortcut",
            "Delete Shortcut",
            "Import Shortcut",
            "Export Shortcut",
        ]

        action = unified_prompt(
            "shortcut_action", "What would you like to do?", actions, allow_back=True
        )

        if action == "BACK" or not action:
            return "continue"
        elif action == "Create New Shortcut":
            create_shortcut()
        elif action == "View/Edit Shortcut":
            view_edit_shortcut()
        elif action == "Delete Shortcut":
            delete_shortcut()
        elif action == "Import Shortcut":
            import_shortcut()
        elif action == "Export Shortcut":
            export_shortcut()


def create_shortcut() -> None:
    """Create a new shortcut."""
    console.print("\n[bold cyan]Create New Shortcut[/bold cyan]")

    # Step 1: Select model
    console.print("\n[bold]Step 1: Select Model[/bold]")
    model_selection = select_model()
    if not model_selection:
        console.print(
            "[yellow]No model selected. Shortcut creation cancelled.[/yellow]"
        )
        input("\nPress Enter to continue...")
        return

    # Handle different model formats
    if isinstance(model_selection, dict):
        model = model_selection.get("model", model_selection)
        model_name = model_selection.get("name", model)
    else:
        model = model_selection
        model_name = model

    # Step 2: Select profile
    console.print("\n[bold]Step 2: Select Profile[/bold]")
    profile_name = select_profile()
    if not profile_name:
        console.print(
            "[yellow]No profile selected. Shortcut creation cancelled.[/yellow]"
        )
        input("\nPress Enter to continue...")
        return

    # Step 3: Name the shortcut
    console.print("\n[bold]Step 3: Name Your Shortcut[/bold]")

    # Suggest a default name
    if "/" in str(model_name):
        model_short = str(model_name).split("/")[-1]
    else:
        model_short = str(model_name)
    suggested_name = f"{model_short}-{profile_name}"

    name = input(f"Shortcut name [{suggested_name}]: ").strip()
    if not name:
        name = suggested_name

    # Step 4: Optional description
    description = input("Description (optional): ").strip()
    if not description:
        description = f"{model_name} with {profile_name} profile"

    # Save the shortcut
    config_manager = ConfigManager()
    shortcut_data = {
        "model": model,
        "profile": profile_name,
        "description": description,
    }

    try:
        if config_manager.save_shortcut(name, shortcut_data):
            console.print(f"\n[green]✓ Shortcut '{name}' created successfully![/green]")
            console.print("\nYou can now use this shortcut from:")
            console.print("  • Quick Serve menu in interactive mode")
            console.print(f'  • Command line: vllm-cli serve --shortcut "{name}"')
        else:
            console.print(f"[red]Failed to create shortcut '{name}'.[/red]")
    except Exception as e:
        console.print(f"[red]Error creating shortcut: {e}[/red]")

    input("\nPress Enter to continue...")


def view_edit_shortcut() -> None:
    """View and optionally edit a shortcut."""
    config_manager = ConfigManager()
    shortcuts = config_manager.list_shortcuts()

    if not shortcuts:
        console.print("[yellow]No shortcuts available to view.[/yellow]")
        input("\nPress Enter to continue...")
        return

    # Select shortcut to view
    shortcut_names = [s["name"] for s in shortcuts]
    selected_name = unified_prompt(
        "select_shortcut",
        "Select a shortcut to view/edit",
        shortcut_names,
        allow_back=True,
    )

    if selected_name == "BACK" or not selected_name:
        return

    # Get full shortcut data
    shortcut = config_manager.get_shortcut(selected_name)
    if not shortcut:
        console.print(f"[red]Shortcut '{selected_name}' not found.[/red]")
        input("\nPress Enter to continue...")
        return

    # Display shortcut details
    console.print(f"\n[bold cyan]Shortcut: {selected_name}[/bold cyan]")
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Model: {shortcut['model']}")
    console.print(f"  Profile: {shortcut['profile']}")
    console.print(f"  Description: {shortcut.get('description', 'No description')}")

    if shortcut.get("created_at"):
        console.print(f"  Created: {shortcut['created_at']}")
    if shortcut.get("last_used"):
        console.print(f"  Last Used: {shortcut['last_used']}")

    # Show profile configuration
    profile = config_manager.get_profile(shortcut["profile"])
    if profile and profile.get("config"):
        display_config(
            profile["config"], title=f"Profile Settings ({shortcut['profile']})"
        )
        console.print()  # Add spacing after table to prevent display issues

    # Edit options
    edit_actions = [
        "Edit Description",
        "Change Model",
        "Change Profile",
        "Rename Shortcut",
    ]

    action = unified_prompt(
        "edit_shortcut_action",
        "What would you like to do?",
        edit_actions,
        allow_back=True,
    )

    if action == "BACK" or not action:
        return

    modified = False

    if action == "Edit Description":
        new_desc = input(
            f"New description [{shortcut.get('description', '')}]: "
        ).strip()
        if new_desc:
            shortcut["description"] = new_desc
            modified = True

    elif action == "Change Model":
        console.print("\n[bold]Select new model:[/bold]")
        new_model = select_model()
        if new_model:
            if isinstance(new_model, dict):
                shortcut["model"] = new_model.get("model", new_model)
            else:
                shortcut["model"] = new_model
            modified = True

    elif action == "Change Profile":
        console.print("\n[bold]Select new profile:[/bold]")
        new_profile = select_profile()
        if new_profile:
            shortcut["profile"] = new_profile
            modified = True

    elif action == "Rename Shortcut":
        new_name = input(f"New name [{selected_name}]: ").strip()
        if new_name and new_name != selected_name:
            # Check if new name already exists
            if config_manager.get_shortcut(new_name):
                console.print(f"[red]Shortcut '{new_name}' already exists.[/red]")
            else:
                # Rename by saving with new name and deleting old
                shortcut["name"] = new_name
                if config_manager.save_shortcut(new_name, shortcut):
                    config_manager.delete_shortcut(selected_name)
                    console.print(f"[green]✓ Shortcut renamed to '{new_name}'[/green]")
                    selected_name = new_name
                else:
                    console.print("[red]Failed to rename shortcut.[/red]")

    # Save modifications if any
    if modified:
        if config_manager.save_shortcut(selected_name, shortcut):
            console.print(
                f"[green]✓ Shortcut '{selected_name}' updated successfully![/green]"
            )
        else:
            console.print(f"[red]Failed to update shortcut '{selected_name}'.[/red]")

    input("\nPress Enter to continue...")


def delete_shortcut() -> None:
    """Delete a shortcut."""
    config_manager = ConfigManager()
    shortcuts = config_manager.list_shortcuts()

    if not shortcuts:
        console.print("[yellow]No shortcuts available to delete.[/yellow]")
        input("\nPress Enter to continue...")
        return

    # Select shortcut to delete
    shortcut_names = [s["name"] for s in shortcuts]
    selected_name = unified_prompt(
        "delete_shortcut",
        "Select a shortcut to delete",
        shortcut_names,
        allow_back=True,
    )

    if selected_name == "BACK" or not selected_name:
        return

    # Confirm deletion
    confirm = inquirer.confirm(
        f"Delete shortcut '{selected_name}'? This cannot be undone.", default=False
    )

    if confirm:
        if config_manager.delete_shortcut(selected_name):
            console.print(f"[green]✓ Shortcut '{selected_name}' deleted.[/green]")
        else:
            console.print(f"[red]Failed to delete shortcut '{selected_name}'.[/red]")
    else:
        console.print("[yellow]Deletion cancelled.[/yellow]")

    input("\nPress Enter to continue...")


def export_shortcut() -> None:
    """Export a shortcut to a file."""
    config_manager = ConfigManager()
    shortcuts = config_manager.list_shortcuts()

    if not shortcuts:
        console.print("[yellow]No shortcuts available to export.[/yellow]")
        input("\nPress Enter to continue...")
        return

    # Select shortcut to export
    shortcut_names = [s["name"] for s in shortcuts]
    selected_name = unified_prompt(
        "export_shortcut",
        "Select a shortcut to export",
        shortcut_names,
        allow_back=True,
    )

    if selected_name == "BACK" or not selected_name:
        return

    # Get export path
    filepath = input(f"Export path [shortcut_{selected_name}.json]: ").strip()
    if not filepath:
        filepath = f"shortcut_{selected_name}.json"

    file_path = Path(filepath)

    # Add .json extension if not present
    if not file_path.suffix:
        file_path = file_path.with_suffix(".json")

    if config_manager.shortcut_manager.export_shortcut(selected_name, file_path):
        console.print(f"[green]✓ Shortcut exported to {file_path}[/green]")
    else:
        console.print("[red]Failed to export shortcut.[/red]")

    input("\nPress Enter to continue...")


def import_shortcut() -> None:
    """Import a shortcut from a file."""
    console.print("\n[bold cyan]Import Shortcut[/bold cyan]")

    filepath = input("Enter path to shortcut JSON file: ").strip()
    if not filepath:
        console.print("[yellow]No file path provided.[/yellow]")
        input("\nPress Enter to continue...")
        return

    file_path = Path(filepath)

    if not file_path.exists():
        console.print(f"[red]File not found: {filepath}[/red]")
        input("\nPress Enter to continue...")
        return

    # Ask for a name for the imported shortcut
    name = input("Shortcut name (leave empty to use file name): ").strip()

    config_manager = ConfigManager()
    try:
        if config_manager.shortcut_manager.import_shortcut(
            file_path, name if name else None
        ):
            console.print("[green]✓ Shortcut imported successfully![/green]")
        else:
            console.print("[red]Failed to import shortcut.[/red]")
    except Exception as e:
        console.print(f"[red]Error importing shortcut: {e}[/red]")

    input("\nPress Enter to continue...")


def serve_with_shortcut(shortcut_name: str) -> str:
    """
    Start a server using a shortcut configuration.

    Args:
        shortcut_name: Name of the shortcut to use

    Returns:
        Status string for menu navigation
    """
    from .server_control import start_server_with_config

    config_manager = ConfigManager()

    # Get the shortcut
    shortcut = config_manager.get_shortcut(shortcut_name)
    if not shortcut:
        console.print(f"[red]Shortcut '{shortcut_name}' not found.[/red]")
        input("\nPress Enter to continue...")
        return "continue"

    # Get the profile configuration
    profile = config_manager.get_profile(shortcut["profile"])
    if not profile:
        console.print(f"[red]Profile '{shortcut['profile']}' not found.[/red]")
        input("\nPress Enter to continue...")
        return "continue"

    # Build the configuration
    config = profile.get("config", {}).copy()
    config["model"] = shortcut["model"]

    # Apply any config overrides from the shortcut
    if "config_overrides" in shortcut:
        config.update(shortcut["config_overrides"])

    # Apply dynamic defaults
    config_with_defaults = config_manager.profile_manager.apply_dynamic_defaults(config)

    # Display configuration
    console.print(f"\n[bold cyan]Starting with Shortcut: {shortcut_name}[/bold cyan]")
    console.print(f"Model: {shortcut['model']}")
    console.print(f"Profile: {shortcut['profile']}")
    if shortcut.get("description"):
        console.print(f"Description: {shortcut['description']}")

    display_config(config_with_defaults, title="Shortcut Configuration")

    # Confirm and start
    confirm = inquirer.confirm("Start server with this configuration?", default=True)

    if confirm:
        # Update last used timestamp
        config_manager.shortcut_manager.update_last_used(shortcut_name)

        # Save as last config
        config_manager.save_last_config(config_with_defaults)

        return start_server_with_config(config_with_defaults)

    return "continue"
