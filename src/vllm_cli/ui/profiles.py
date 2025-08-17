#!/usr/bin/env python3
"""
Profile management module for vLLM CLI.

Handles creation, editing, and deletion of configuration profiles.
"""
import logging

import inquirer

from ..config import ConfigManager
from .common import console
from .display import display_config
from .navigation import unified_prompt

logger = logging.getLogger(__name__)


def manage_profiles() -> str:
    """
    Manage configuration profiles.
    """
    config_manager = ConfigManager()
    # all_profiles = config_manager.get_all_profiles()  # Not used directly

    # Show existing profiles
    console.print("\n[bold cyan]Configuration Profiles[/bold cyan]")

    # Separate default and user profiles
    default_profiles = config_manager.default_profiles
    user_profiles = config_manager.user_profiles

    # Built-in profiles
    console.print("\n[bold]Built-in Profiles:[/bold]")
    for name, profile in default_profiles.items():
        icon = profile.get("icon", "")
        desc = profile.get("description", "")
        # Check if this built-in profile has been customized
        if config_manager.profile_manager.has_user_override(name):
            console.print(f"  {icon} {name} - {desc} [yellow](customized)[/yellow]")
        else:
            console.print(f"  {icon} {name} - {desc}")

    # User profiles (excluding overrides of built-in profiles)
    user_only_profiles = {
        name: profile
        for name, profile in user_profiles.items()
        if name not in default_profiles
    }

    if user_only_profiles:
        console.print("\n[bold]User Profiles:[/bold]")
        for name, profile in user_only_profiles.items():
            icon = profile.get("icon", "")
            desc = profile.get("description", "Custom profile")
            console.print(f"  {icon} {name} - {desc}")
    else:
        console.print("\n[dim]No user-created profiles[/dim]")

    # Profile actions - streamlined menu
    actions = [
        "Manage Profiles",
        "Create New Profile",
        "Import Profile",
    ]
    action = unified_prompt(
        "profile_action", "Profile Management", actions, allow_back=True
    )

    if action == "Manage Profiles":
        manage_selected_profile()
    elif action == "Create New Profile":
        create_custom_profile()
    elif action == "Import Profile":
        import_profile()

    return "continue"


def manage_selected_profile() -> None:
    """
    Manage a selected profile - view details and perform actions.
    """
    config_manager = ConfigManager()
    all_profiles = config_manager.get_all_profiles()

    if not all_profiles:
        console.print("[yellow]No profiles available to view.[/yellow]")
        input("\nPress Enter to continue...")
        return

    # Prepare profile list with type indicators (plain text for menu)
    profile_choices = []
    profile_map = {}  # Map display name to actual profile name

    # Add built-in profiles
    for name in config_manager.default_profiles.keys():
        if config_manager.profile_manager.has_user_override(name):
            display_name = f"{name} (customized)"
        else:
            display_name = f"{name} (built-in)"
        profile_choices.append(display_name)
        profile_map[display_name] = name

    # Add user-only profiles
    for name in config_manager.user_profiles.keys():
        if name not in config_manager.default_profiles:
            display_name = f"{name} (user-created)"
            profile_choices.append(display_name)
            profile_map[display_name] = name

    # Select profile to manage
    console.print("\n[bold cyan]Manage Profiles[/bold cyan]")
    selected = unified_prompt(
        "profile",
        "Select a profile to manage",
        profile_choices,
        allow_back=True,
    )

    if not selected or selected == "BACK":
        return

    # Get actual profile name
    profile_name = profile_map.get(selected, selected)

    # Clear screen for better view
    console.print("\n" * 2)

    # Display profile header
    console.rule(f"[bold cyan]Profile: {profile_name}[/bold cyan]")

    # Get profile data
    profile = config_manager.get_profile(profile_name)
    if not profile:
        console.print(f"[red]Profile '{profile_name}' not found.[/red]")
        input("\nPress Enter to continue...")
        return

    # Display profile metadata
    console.print("\n[bold]Profile Information:[/bold]")
    console.print(f"  Name: {profile_name}")
    console.print(f"  Description: {profile.get('description', 'No description')}")
    if profile.get("icon"):
        console.print(f"  Icon: {profile.get('icon')}")

    # Determine and display profile type
    if config_manager.profile_manager.has_user_override(profile_name):
        console.print("  Type: [yellow]Customized Built-in Profile[/yellow]")
    elif config_manager.profile_manager.is_user_profile(profile_name):
        if profile_name in config_manager.default_profiles:
            console.print("  Type: [yellow]User Override of Built-in[/yellow]")
        else:
            console.print("  Type: [cyan]User-Created Profile[/cyan]")
    else:
        console.print("  Type: [green]Built-in Default Profile[/green]")

    # Display current configuration
    console.print("\n[bold]Configuration Settings:[/bold]")
    config = profile.get("config", {})

    if config:
        # Apply dynamic defaults for display
        config_with_defaults = config_manager.profile_manager.apply_dynamic_defaults(
            config
        )
        display_config(config_with_defaults)
    else:
        console.print("[dim]  No custom configuration (uses all vLLM defaults)[/dim]")

    # If this is a customized built-in profile, offer to show original
    if config_manager.profile_manager.has_user_override(profile_name):
        console.print("\n[yellow]This is a customized built-in profile.[/yellow]")

        show_original = (
            input("\nShow original default configuration? (y/N): ").strip().lower()
        )
        if show_original in ["y", "yes"]:
            original = config_manager.profile_manager.get_original_default_profile(
                profile_name
            )
            if original:
                console.print("\n[bold]Original Default Configuration:[/bold]")
                original_config = original.get("config", {})
                if original_config:
                    original_with_defaults = (
                        config_manager.profile_manager.apply_dynamic_defaults(
                            original_config
                        )
                    )
                    display_config(original_with_defaults)
                else:
                    console.print(
                        "[dim]  No custom configuration (uses all vLLM defaults)[/dim]"
                    )

    # Offer quick actions using unified navigation
    actions = ["Edit this profile", "Export this profile"]

    # Add delete option for user profiles (not for unmodified built-in)
    if config_manager.profile_manager.is_user_profile(profile_name):
        if profile_name not in config_manager.default_profiles:
            # Only user-created profiles can be deleted
            actions.append("Delete this profile")

    # Add reset option if this is a customized built-in profile
    if config_manager.profile_manager.has_user_override(profile_name):
        actions.append("Reset to default")

    # Use unified prompt for consistent navigation
    action = unified_prompt(
        "profile_detail_action", "What would you like to do?", actions, allow_back=True
    )

    if action == "Edit this profile":
        # Edit the profile
        edit_specific_profile(profile_name)
        # After editing, ask if they want to view the updated profile
        view_again = inquirer.confirm("View the updated profile?", default=True)
        if view_again:
            # Recursive call to view the updated profile
            manage_selected_profile()
    elif action == "Export this profile":
        # Export the profile
        export_specific_profile(profile_name)
    elif action == "Delete this profile":
        # Delete the profile
        confirm = inquirer.confirm(
            f"Delete profile '{profile_name}'? This cannot be undone.", default=False
        )
        if confirm:
            if config_manager.delete_user_profile(profile_name):
                console.print(f"[green]✓ Profile '{profile_name}' deleted.[/green]")
                input("\nPress Enter to continue...")
                return  # Exit to main menu after deletion
            else:
                console.print(f"[red]Failed to delete profile '{profile_name}'.[/red]")
                input("\nPress Enter to continue...")
    elif action == "Reset to default":
        # Reset to default
        confirm = inquirer.confirm(
            f"Reset '{profile_name}' to its default configuration?", default=False
        )
        if confirm:
            if config_manager.profile_manager.reset_to_default(profile_name):
                console.print(
                    f"[green]✓ Profile '{profile_name}' reset to default.[/green]"
                )
                # Ask if they want to view the reset profile
                view_again = inquirer.confirm("View the reset profile?", default=True)
                if view_again:
                    manage_selected_profile()
                else:
                    input("\nPress Enter to continue...")
            else:
                console.print(f"[red]Failed to reset profile '{profile_name}'.[/red]")
                input("\nPress Enter to continue...")
    # If "BACK" or nothing selected, just return


def create_custom_profile() -> None:
    """
    Create a new custom profile.
    """
    console.print("\n[bold cyan]Create Custom Profile[/bold cyan]")

    name = input("Profile name: ").strip()
    if not name:
        console.print("[yellow]Profile name required.[/yellow]")
        return

    # Build profile configuration
    config = {}

    # Use guided configuration
    console.print("\nConfigure profile settings (press Enter for defaults):")

    config["dtype"] = (
        input("Data type (auto/float16/bfloat16/float32) [auto]: ").strip() or "auto"
    )

    # Max model length - allow empty for native model max
    max_model_len_input = input(
        "Max model length (leave empty for model's native max): "
    ).strip()
    if max_model_len_input:
        config["max_model_len"] = int(max_model_len_input)

    # Tensor parallel size - smart defaults based on GPU count
    from ..system.gpu import get_gpu_info

    try:
        gpus = get_gpu_info()
        detected_gpus = len(gpus) if gpus else 1
        console.print(f"[dim]Detected {detected_gpus} GPU(s)[/dim]")

        if detected_gpus == 1:
            console.print("[yellow]Single GPU: vLLM will use 1 GPU by default[/yellow]")
            tensor_parallel_input = input(
                "Tensor parallel size (leave empty for default): "
            ).strip()
            if tensor_parallel_input:
                config["tensor_parallel_size"] = int(tensor_parallel_input)
            # Don't set tensor_parallel_size for single GPU (let vLLM use default)
        else:
            console.print(
                "[green]Multi-GPU system: tensor parallelism recommended[/green]"
            )
            tensor_parallel_input = input(
                f"Tensor parallel size [{detected_gpus}]: "
            ).strip()
            config["tensor_parallel_size"] = (
                int(tensor_parallel_input) if tensor_parallel_input else detected_gpus
            )
    except Exception:
        tensor_parallel_input = input(
            "Tensor parallel size (leave empty for default): "
        ).strip()
        if tensor_parallel_input:
            config["tensor_parallel_size"] = int(tensor_parallel_input)
    config["gpu_memory_utilization"] = float(
        input("GPU memory utilization (0.0-1.0) [0.90]: ").strip() or "0.90"
    )

    # Save profile
    config_manager = ConfigManager()
    config_manager.save_user_profile(
        name, {"name": name, "description": "Custom user profile", "config": config}
    )
    console.print(f"[green]Profile '{name}' created successfully.[/green]")
    input("\nPress Enter to continue...")


def edit_specific_profile(profile_name: str) -> None:
    """
    Edit a specific profile directly.
    Helper function for quick editing from view details.
    """
    config_manager = ConfigManager()

    # Get the profile
    profile = config_manager.get_profile(profile_name)
    if not profile:
        console.print(f"[red]Profile '{profile_name}' not found.[/red]")
        return

    # Get configuration
    config = profile.get("config", {}).copy()

    # If editing a built-in profile that hasn't been customized yet, inform the user
    if (
        profile_name in config_manager.default_profiles
        and not config_manager.profile_manager.has_user_override(profile_name)
    ):
        console.print(
            f"\n[yellow]Note: Editing built-in profile '{profile_name}'.[/yellow]"
        )
        console.print(
            "[yellow]Your changes will create a customized version that overrides the default.[/yellow]"
        )
        console.print("[dim]You can reset to default later if needed.[/dim]\n")

    console.print(f"\n[bold cyan]Editing Profile: {profile_name}[/bold cyan]")
    console.print("Current configuration:")
    display_config(config)

    console.print("\nEnter new values (press Enter to keep current):")

    # Create a list of keys to iterate over (to avoid dictionary modification during iteration)
    config_keys = list(config.keys())
    keys_to_delete = []

    for key in config_keys:
        current_value = config[key]
        if key == "max_model_len":
            new_value = input(
                f"{key} [{current_value}] (leave empty to remove limit): "
            ).strip()
            if new_value == "":
                # Mark for deletion after iteration
                keys_to_delete.append(key)
            elif new_value:
                config[key] = int(new_value)
        else:
            new_value = input(f"{key} [{current_value}]: ").strip()
            if new_value:
                # Convert to appropriate type
                if key in ["tensor_parallel_size"]:
                    config[key] = int(new_value)
                elif key == "gpu_memory_utilization":
                    config[key] = float(new_value)
                elif key in [
                    "trust_remote_code",
                    "enable_chunked_prefill",
                    "enable_expert_parallel",
                    "enable_prefix_caching",
                ]:
                    config[key] = new_value.lower() in ["true", "yes", "1"]
                else:
                    config[key] = new_value

    # Delete keys marked for deletion
    for key in keys_to_delete:
        del config[key]

    # Prepare profile data
    profile_data = {
        "name": profile_name,
        "description": profile.get("description", "Custom profile"),
        "icon": profile.get("icon", ""),
        "config": config,
    }

    # Save updated profile (this will create a user override if editing a built-in profile)
    config_manager.save_user_profile(profile_name, profile_data)

    if profile_name in config_manager.default_profiles:
        console.print(
            f"[green]Profile '{profile_name}' customized successfully.[/green]"
        )
        console.print(
            "[dim]The built-in version is preserved and can be restored later.[/dim]"
        )
    else:
        console.print(f"[green]Profile '{profile_name}' updated.[/green]")

    input("\nPress Enter to continue...")


def export_specific_profile(profile_name: str) -> None:
    """
    Export a specific profile directly.
    Helper function for quick export from view details.
    """
    config_manager = ConfigManager()

    console.print(f"\n[bold cyan]Export Profile: {profile_name}[/bold cyan]")

    # Get export path
    filepath = input("Enter export path (e.g., profile.json): ").strip()
    if not filepath:
        console.print("[yellow]No file path provided.[/yellow]")
        return

    from pathlib import Path

    file_path = Path(filepath)

    # Add .json extension if not present
    if not file_path.suffix:
        file_path = file_path.with_suffix(".json")

    if config_manager.export_profile(profile_name, file_path):
        console.print(f"[green]Profile exported to {file_path}[/green]")
    else:
        console.print("[red]Failed to export profile.[/red]")

    input("\nPress Enter to continue...")


def import_profile() -> None:
    """Import a profile from a JSON file."""
    console.print("\n[bold cyan]Import Profile[/bold cyan]")

    filepath = input("Enter path to profile JSON file: ").strip()
    if not filepath:
        console.print("[yellow]No file path provided.[/yellow]")
        input("\nPress Enter to continue...")
        return

    from pathlib import Path

    file_path = Path(filepath)

    if not file_path.exists():
        console.print(f"[red]File not found: {filepath}[/red]")
        input("\nPress Enter to continue...")
        return

    # Ask for a name for the imported profile
    name = input("Profile name (leave empty to use file name): ").strip()

    config_manager = ConfigManager()
    if config_manager.import_profile(file_path, name if name else None):
        console.print("[green]Profile imported successfully.[/green]")
    else:
        console.print("[red]Failed to import profile.[/red]")

    input("\nPress Enter to continue...")
