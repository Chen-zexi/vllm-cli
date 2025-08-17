#!/usr/bin/env python3
"""
Enhanced category-based custom configuration for vLLM CLI with simplified numerical inputs.
"""
import logging
from typing import Any, Dict, List, Optional

import inquirer

from ..config import ConfigManager
from .common import console
from .navigation import unified_prompt

logger = logging.getLogger(__name__)


def configure_by_categories(
    base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Configure vLLM server arguments using a category-based approach.

    Args:
        base_config: Optional base configuration to start from

    Returns:
        Configured dictionary of arguments
    """
    config_manager = ConfigManager()
    config = base_config.copy() if base_config else {}

    console.print("\n[bold cyan]Category-Based Configuration[/bold cyan]")
    console.print("Configure only what you need - press Enter to use defaults\n")

    # Get ordered categories
    categories = config_manager.get_ordered_categories()

    # First handle default categories (essential and performance)
    for category_id, category_info in categories:
        show_by_default = category_info.get("show_by_default", False)

        # Only process default categories in this loop
        if not show_by_default:
            continue

        category_name = category_info["name"]
        category_desc = category_info["description"]
        # icon = category_info.get("icon", "")  # Reserved for future use

        # Show category header
        console.print(f"\n[bold]{category_name}[/bold]")
        console.print(f"[dim]{category_desc}[/dim]")

        # Get arguments for this category
        args = config_manager.get_arguments_by_category(category_id)

        # Filter to show only high/critical importance by default
        important_args = [
            a for a in args if a.get("importance") in ["critical", "high"]
        ]
        other_args = [
            a for a in args if a.get("importance") not in ["critical", "high"]
        ]

        # Configure important arguments
        for arg_info in important_args:
            config = configure_argument(arg_info, config, config_manager)

        # Ask if they want to see additional options
        if other_args:
            show_additional = inquirer.confirm(
                f"Configure additional {category_name.lower()} options?", default=False
            )
            if show_additional:
                for arg_info in other_args:
                    config = configure_argument(arg_info, config, config_manager)

    # Now ask about advanced options with hierarchical menu
    console.print("\n[bold]Advanced Configuration[/bold]")
    configure_advanced = inquirer.confirm("Configure advanced options?", default=False)

    if configure_advanced:
        config = configure_advanced_hierarchical(config, config_manager)

    return config


def configure_advanced_hierarchical(
    config: Dict[str, Any], config_manager: ConfigManager
) -> Dict[str, Any]:
    """
    Configure advanced options using a hierarchical category menu.

    Args:
        config: Current configuration
        config_manager: ConfigManager instance

    Returns:
        Updated configuration
    """
    # Get non-default categories (including parallelism now)
    categories = config_manager.get_ordered_categories()
    advanced_categories = [
        (cat_id, cat_info)
        for cat_id, cat_info in categories
        if not cat_info.get("show_by_default", False)
    ]

    while True:
        # Build category menu
        category_choices = []
        for cat_id, cat_info in advanced_categories:
            # icon = cat_info.get("icon", "")  # Reserved for future use
            name = cat_info["name"]

            # Count configured arguments in this category
            args = config_manager.get_arguments_by_category(cat_id)
            configured_count = sum(1 for arg in args if arg["name"] in config)
            total_count = len(args)

            if configured_count > 0:
                status = f" [{configured_count}/{total_count} configured]"
            else:
                status = f" [{total_count} options]"

            category_choices.append(f"{name}{status}")

        category_choices.append("✓ Done configuring")

        # Show category selection menu
        console.print("\n[bold cyan]Advanced Options - Select Category[/bold cyan]")
        selected = unified_prompt(
            "category_select",
            "Choose category to configure",
            category_choices,
            allow_back=False,
        )

        if selected == "✓ Done configuring" or not selected:
            break

        # Find the selected category
        for cat_id, cat_info in advanced_categories:
            # icon = cat_info.get("icon", "")  # Reserved for future use
            name = cat_info["name"]
            if selected.startswith(name):
                # Configure this category
                config = configure_category_arguments(
                    cat_id, cat_info, config, config_manager
                )
                break

    return config


def configure_category_arguments(
    category_id: str,
    category_info: Dict[str, Any],
    config: Dict[str, Any],
    config_manager: ConfigManager,
) -> Dict[str, Any]:
    """
    Configure arguments within a specific category using list selection.

    Args:
        category_id: Category identifier
        category_info: Category metadata
        config: Current configuration
        config_manager: ConfigManager instance

    Returns:
        Updated configuration
    """
    category_name = category_info["name"]
    # icon = category_info.get("icon", "")  # Reserved for future use

    # Get arguments for this category
    args = config_manager.get_arguments_by_category(category_id)

    while True:
        # Build argument list with current values
        console.print(f"\n[bold cyan]{category_name}[/bold cyan]")
        console.print(f"[dim]{category_info.get('description', '')}[/dim]\n")

        arg_choices = []
        arg_map = {}

        for arg_info in args:
            arg_name = arg_info["name"]
            current_value = config.get(arg_name, arg_info.get("default"))
            importance = arg_info.get("importance", "low")
            description = arg_info.get("description", "")

            # Format display string with better formatting
            if current_value is not None and arg_name in config:
                # Configured value (user set)
                value_str = f"[bold green]{current_value}[/bold green]"
                status_icon = "●"
            elif current_value is not None:
                # Default value
                value_str = f"[dim]{current_value}[/dim]"
                status_icon = "○"
            else:
                # Not set
                value_str = "[dim italic]not set[/dim italic]"
                status_icon = "○"

            # Add importance indicator
            if importance in ["high", "critical"]:
                importance_icon = "!"
            else:
                importance_icon = "  "

            # Create display string with description
            display = f"{status_icon} {importance_icon} {arg_name}: {value_str}"
            if description and len(description) < 40:
                display += f" [dim]- {description[:40]}[/dim]"

            arg_choices.append(display)
            arg_map[display] = arg_info

        # Add navigation options
        arg_choices.append("← Back to categories")

        # Show argument selection
        selected = unified_prompt(
            "arg_select", "Select argument to configure", arg_choices, allow_back=False
        )

        if selected == "← Back to categories" or not selected:
            break

        # Configure selected argument
        if selected in arg_map:
            arg_info = arg_map[selected]
            config = configure_argument(arg_info, config, config_manager)

    return config


def configure_advanced_list(
    args: List[Dict],
    config: Dict[str, Any],
    config_manager: ConfigManager,
    category_name: str,
) -> Dict[str, Any]:
    """
    Configure advanced arguments using a list-based selection approach.
    This is kept for backward compatibility but enhanced with better formatting.

    Args:
        args: List of argument information
        config: Current configuration
        config_manager: ConfigManager instance
        category_name: Name of the category

    Returns:
        Updated configuration
    """
    while True:
        # Build list of arguments with current values
        console.print(f"\n[bold cyan]{category_name} Configuration[/bold cyan]")

        arg_choices = []
        arg_map = {}

        for arg_info in args:
            arg_name = arg_info["name"]
            current_value = config.get(arg_name, arg_info.get("default"))
            importance = arg_info.get("importance", "low")
            # description = arg_info.get("description", "")  # Reserved for future use

            # Format display string with better indicators
            if current_value is not None and arg_name in config:
                # User configured
                value_str = f"[bold green]{current_value}[/bold green]"
                status_icon = "●"
            elif current_value is not None:
                # Default value
                value_str = f"[dim]{current_value}[/dim]"
                status_icon = "○"
            else:
                # Not set
                value_str = "[dim italic]not set[/dim italic]"
                status_icon = "○"

            # Add importance indicator
            if importance in ["high", "critical"]:
                importance_icon = "!"
            else:
                importance_icon = "  "

            display = f"{status_icon} {importance_icon} {arg_name}: {value_str}"

            arg_choices.append(display)
            arg_map[display] = arg_info

        # Add navigation options
        arg_choices.append("← Back")

        selected = unified_prompt(
            "arg_select", "Select argument to configure", arg_choices, allow_back=False
        )

        if selected == "← Back" or not selected:
            break

        # Configure selected argument
        if selected in arg_map:
            arg_info = arg_map[selected]
            config = configure_argument(arg_info, config, config_manager)

    return config


def configure_argument(
    arg_info: Dict[str, Any], config: Dict[str, Any], config_manager: ConfigManager
) -> Dict[str, Any]:
    """
    Configure a single argument with simplified numerical inputs.

    Args:
        arg_info: Argument schema information
        config: Current configuration dictionary
        config_manager: ConfigManager instance

    Returns:
        Updated configuration dictionary
    """
    arg_name = arg_info["name"]
    arg_type = arg_info.get("type")
    description = arg_info.get("description", "")
    default = arg_info.get("default")
    hint = arg_info.get("hint", "")

    # Check dependencies
    if "depends_on" in arg_info:
        dependency = arg_info["depends_on"]
        if not config.get(dependency):
            return config  # Skip if dependency not met

    # Build description
    current_value = config.get(arg_name, default)

    console.print(f"\n[bold]{arg_name}[/bold]")
    console.print(f"[dim]{description}[/dim]")
    if hint:
        console.print(f"[yellow dim]{hint}[/yellow dim]")

    # Handle different argument types
    if arg_type == "boolean":
        # Use inquirer for boolean choices
        current_str = current_value if current_value is not None else default
        result = inquirer.confirm(
            f"Enable {arg_name}?",
            default=current_str if isinstance(current_str, bool) else False,
        )
        config[arg_name] = result

    elif arg_type == "choice":
        choices = arg_info.get("choices", [])

        # Special handling for quantization
        if arg_name == "quantization":
            choice_list = [
                "None - No quantization (full precision)",
                "awq - AutoAWQ (Activation-aware Weight Quantization)",
                "awq_marlin - AutoAWQ with Marlin kernel",
                "bitsandbytes - 8-bit/4-bit quantization",
                "gptq - GPT Quantization",
                "fp8 - 8-bit floating point",
                "gguf - GGML universal format",
                "compressed-tensors - INT4/INT8 compressed",
                "Skip (use default)",
            ]

            selected = unified_prompt(
                arg_name, "Select quantization method", choice_list, allow_back=False
            )

            if selected and selected != "Skip (use default)":
                if selected.startswith("None"):
                    config[arg_name] = None
                else:
                    # Extract method from selection
                    method = selected.split(" - ")[0]
                    config[arg_name] = method
        else:
            # Regular choice field
            choice_list = []
            for choice in choices:
                if choice is None:
                    choice_list.append("None (use vLLM default)")
                else:
                    choice_list.append(str(choice))

            choice_list.append("Skip (use default)")

            if current_value is not None:
                console.print(f"Current: {current_value}")

            selected = unified_prompt(
                arg_name, f"Select {arg_name}", choice_list, allow_back=False
            )

            if selected and selected != "Skip (use default)":
                if selected == "None (use vLLM default)":
                    config[arg_name] = None
                else:
                    for c in choices:
                        if str(c) == selected:
                            config[arg_name] = c
                            break

    elif arg_type == "integer":
        validation = arg_info.get("validation", {})
        min_val = validation.get("min")
        max_val = validation.get("max")

        # Special handling for tensor_parallel_size
        if arg_name == "tensor_parallel_size":
            from ..system.gpu import get_gpu_info

            try:
                gpus = get_gpu_info()
                num_gpus = len(gpus) if gpus else 1
                console.print(f"[dim]Detected {num_gpus} GPU(s)[/dim]")

                if num_gpus == 1:
                    console.print(
                        "[yellow]Single GPU detected, vLLM will use 1 GPU by default[/yellow]"
                    )
                    response = input(
                        "Enter tensor parallel size (1) or press Enter to use default: "
                    ).strip()
                    if response:
                        try:
                            config[arg_name] = int(response)
                        except ValueError:
                            console.print("[red]Invalid number, using default[/red]")
                    # Don't set tensor_parallel_size for single GPU (let vLLM default)
                    return config
                else:
                    # For multi-GPU, suggest using all GPUs but let user choose
                    console.print(
                        "[green]Multi-GPU system detected, tensor parallelism recommended[/green]"
                    )
                    # Set default to detected GPU count
                    default = num_gpus
            except Exception:
                num_gpus = 1

        # Special handling for max_model_len
        if arg_name == "max_model_len":
            console.print(
                "[dim]Leave empty to use model's native maximum context length[/dim]"
            )
            response = input(
                "Enter max model length or press Enter for native max: "
            ).strip()
            if response:
                try:
                    config[arg_name] = int(response)
                except ValueError:
                    console.print("[red]Invalid number, skipping max_model_len[/red]")
            # If empty, don't set max_model_len (let vLLM use model's native max)
            return config

        # Simple numerical input
        range_str = ""
        if min_val is not None or max_val is not None:
            if min_val is not None and max_val is not None:
                range_str = f" (range: {min_val}-{max_val})"
            elif min_val is not None:
                range_str = f" (min: {min_val})"
            elif max_val is not None:
                range_str = f" (max: {max_val})"

        if default is not None:
            prompt_text = (
                f"Enter value{range_str} or press Enter for default ({default}): "
            )
        else:
            prompt_text = f"Enter value{range_str} or press Enter to skip: "

        response = input(prompt_text).strip()

        if response:
            try:
                value = int(response)
                if min_val is not None and value < min_val:
                    console.print(
                        f"[yellow]Value must be at least {min_val}, using {min_val}[/yellow]"
                    )
                    config[arg_name] = min_val
                elif max_val is not None and value > max_val:
                    console.print(
                        f"[yellow]Value must be at most {max_val}, using {max_val}[/yellow]"
                    )
                    config[arg_name] = max_val
                else:
                    config[arg_name] = value
            except ValueError:
                console.print("[yellow]Invalid integer value, skipping[/yellow]")
        # If no response and user pressed Enter, don't add to config (use default)

    elif arg_type == "float":
        validation = arg_info.get("validation", {})
        min_val = validation.get("min")
        max_val = validation.get("max")

        # Simple numerical input for all float fields
        range_str = ""
        if min_val is not None or max_val is not None:
            if min_val is not None and max_val is not None:
                range_str = f" (range: {min_val}-{max_val})"
            elif min_val is not None:
                range_str = f" (min: {min_val})"
            elif max_val is not None:
                range_str = f" (max: {max_val})"

        # Special prompt for gpu_memory_utilization
        if arg_name == "gpu_memory_utilization":
            console.print(
                "[dim]Common values: 0.5 (50%), 0.7 (70%), 0.9 (90%), 0.95 (95%)[/dim]"
            )

        if default is not None:
            prompt_text = (
                f"Enter value{range_str} or press Enter for default ({default}): "
            )
        else:
            prompt_text = f"Enter value{range_str} or press Enter to skip: "

        response = input(prompt_text).strip()

        if response:
            try:
                value = float(response)
                if min_val is not None and value < min_val:
                    console.print(
                        f"[yellow]Value must be at least {min_val}, using {min_val}[/yellow]"
                    )
                    config[arg_name] = min_val
                elif max_val is not None and value > max_val:
                    console.print(
                        f"[yellow]Value must be at most {max_val}, using {max_val}[/yellow]"
                    )
                    config[arg_name] = max_val
                else:
                    config[arg_name] = value
            except ValueError:
                console.print("[yellow]Invalid float value, skipping[/yellow]")
        # If no response and user pressed Enter, don't add to config (use default)

    elif arg_type == "string":
        sensitive = arg_info.get("sensitive", False)

        if sensitive:
            prompt_text = "Enter value (hidden) or press Enter to skip: "
        else:
            if current_value:
                prompt_text = (
                    f"Enter value (current: {current_value}) or press Enter to skip: "
                )
            else:
                prompt_text = "Enter value or press Enter to skip: "

        response = input(prompt_text).strip()
        if response:
            config[arg_name] = response

    return config


def create_custom_profile_interactive() -> Optional[str]:
    """
    Create a custom profile using the category-based configuration.

    Returns:
        Profile name if created successfully, None otherwise
    """
    console.print("\n[bold cyan]Create Custom Profile[/bold cyan]")

    # Get profile name
    name = input("Profile name: ").strip()
    if not name:
        console.print("[yellow]Profile name required.[/yellow]")
        return None

    description = input("Profile description (optional): ").strip()

    # Choose starting point
    config_manager = ConfigManager()
    all_profiles = config_manager.get_all_profiles()

    start_choices = ["Start from scratch"] + list(all_profiles.keys())
    start_choice = unified_prompt(
        "start", "Starting configuration", start_choices, allow_back=True
    )

    if not start_choice or start_choice == "BACK":
        return None

    # Get base configuration
    if start_choice == "Start from scratch":
        base_config = {}
    else:
        profile = config_manager.get_profile(start_choice)
        base_config = profile.get("config", {}).copy() if profile else {}

    # Configure using categories
    config = configure_by_categories(base_config)

    # Show summary
    console.print("\n[bold cyan]Profile Summary:[/bold cyan]")
    if config:
        for key, value in config.items():
            if value is not None:
                console.print(f"  {key}: {value}")
    else:
        console.print("[dim]No custom configuration - will use all defaults[/dim]")

    # Confirm save
    save = inquirer.confirm("Save this profile?", default=True)
    if not save:
        return None

    # Save profile
    profile_data = {
        "name": name,
        "description": description or "Custom profile",
        "icon": "",
        "config": config,
    }

    if config_manager.save_user_profile(name, profile_data):
        console.print(f"[green]Profile '{name}' saved successfully.[/green]")
        return name
    else:
        console.print("[red]Failed to save profile.[/red]")
        return None
