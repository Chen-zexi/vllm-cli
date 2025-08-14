#!/usr/bin/env python3
"""
Profile management module for vLLM CLI.

Handles creation, editing, and deletion of configuration profiles.
"""
import logging

import inquirer

from ..config import ConfigManager
from .navigation import unified_prompt
from .common import console
from .display import display_config

logger = logging.getLogger(__name__)


def manage_profiles() -> str:
    """
    Manage configuration profiles.
    """
    config_manager = ConfigManager()
    all_profiles = config_manager.get_all_profiles()

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
        console.print(f"  {icon} {name} - {desc}")

    # User profiles
    if user_profiles:
        console.print("\n[bold]User Profiles:[/bold]")
        for name, profile in user_profiles.items():
            icon = profile.get("icon", "")
            desc = profile.get("description", "Custom profile")
            console.print(f"  {icon} {name} - {desc}")
    else:
        console.print("\n[dim]No user profiles[/dim]")

    # Profile actions
    actions = [
        "Create New Profile",
        "Edit Profile",
        "Delete Profile",
        "Import Profile",
        "Export Profile",
    ]
    action = unified_prompt(
        "profile_action", "Profile Management", actions, allow_back=True
    )

    if action == "Create New Profile":
        create_custom_profile()
    elif action == "Edit Profile":
        edit_profile()
    elif action == "Delete Profile":
        delete_profile()
    elif action == "Import Profile":
        import_profile()
    elif action == "Export Profile":
        export_profile()

    return "continue"


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
                f"[green]Multi-GPU system: tensor parallelism recommended[/green]"
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


def edit_profile() -> None:
    """
    Edit an existing profile.
    """
    config_manager = ConfigManager()
    custom_profiles = config_manager.user_profiles

    if not custom_profiles:
        console.print("[yellow]No custom profiles to edit.[/yellow]")
        input("\nPress Enter to continue...")
        return

    # Select profile
    profile_name = unified_prompt(
        "profile",
        "Select profile to edit",
        list(custom_profiles.keys()),
        allow_back=True,
    )

    if not profile_name or profile_name == "BACK":
        return

    # Edit configuration
    config = custom_profiles[profile_name]
    console.print(f"\n[bold cyan]Editing Profile: {profile_name}[/bold cyan]")
    console.print("Current configuration:")
    display_config(config)

    console.print("\nEnter new values (press Enter to keep current):")

    for key, current_value in config.items():
        if key == "max_model_len":
            new_value = input(
                f"{key} [{current_value}] (leave empty to remove limit): "
            ).strip()
            if new_value == "":
                # Remove max_model_len to use model's native max
                if key in config:
                    del config[key]
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
                elif key in ["trust_remote_code", "enable_chunked_prefill"]:
                    config[key] = new_value.lower() in ["true", "yes", "1"]
                else:
                    config[key] = new_value

    # Save updated profile
    config_manager.save_user_profile(profile_name, {"config": config})
    console.print(f"[green]Profile '{profile_name}' updated.[/green]")
    input("\nPress Enter to continue...")


def delete_profile() -> None:
    """
    Delete a custom profile.
    """
    config_manager = ConfigManager()
    custom_profiles = config_manager.user_profiles

    if not custom_profiles:
        console.print("[yellow]No custom profiles to delete.[/yellow]")
        input("\nPress Enter to continue...")
        return

    # Select profile
    profile_name = unified_prompt(
        "profile",
        "Select profile to delete",
        list(custom_profiles.keys()),
        allow_back=True,
    )

    if not profile_name or profile_name == "BACK":
        return

    # Confirm deletion
    confirm = inquirer.confirm(f"Delete profile '{profile_name}'?", default=False)

    if confirm:
        config_manager.delete_user_profile(profile_name)
        console.print(f"[green]Profile '{profile_name}' deleted.[/green]")

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
        console.print(f"[green]Profile imported successfully.[/green]")
    else:
        console.print(f"[red]Failed to import profile.[/red]")

    input("\nPress Enter to continue...")


def export_profile() -> None:
    """Export a profile to a JSON file."""
    config_manager = ConfigManager()
    all_profiles = config_manager.get_all_profiles()

    if not all_profiles:
        console.print("[yellow]No profiles available to export.[/yellow]")
        input("\nPress Enter to continue...")
        return

    # Select profile to export
    profile_name = unified_prompt(
        "profile",
        "Select profile to export",
        list(all_profiles.keys()),
        allow_back=True,
    )

    if not profile_name or profile_name == "BACK":
        return

    console.print(f"\n[bold cyan]Export Profile: {profile_name}[/bold cyan]")

    # Get export path
    filepath = input("Enter export path (e.g., profile.json): ").strip()
    if not filepath:
        console.print("[yellow]No file path provided.[/yellow]")
        input("\nPress Enter to continue...")
        return

    from pathlib import Path

    file_path = Path(filepath)

    # Add .json extension if not present
    if not file_path.suffix:
        file_path = file_path.with_suffix(".json")

    if config_manager.export_profile(profile_name, file_path):
        console.print(f"[green]Profile exported to {file_path}[/green]")
    else:
        console.print(f"[red]Failed to export profile.[/red]")

    input("\nPress Enter to continue...")
