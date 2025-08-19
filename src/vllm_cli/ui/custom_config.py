#!/usr/bin/env python3
"""
Enhanced category-based custom configuration for vLLM CLI with simplified numerical inputs.
"""
import logging
from typing import Any, Dict, List, Optional

import inquirer

from ..config import ConfigManager
from ..system import gpu
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

        # Exclude model field from profiles (it's selected separately when serving)
        args = [a for a in args if a["name"] != "model"]

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

    # Ask about GPU selection
    console.print()  # Add blank line for spacing
    configure_gpus = inquirer.confirm("Configure GPU selection?", default=False)

    if configure_gpus:
        current_device = config.get("device")
        new_device = select_gpus(current_device)
        if new_device is not None:
            config["device"] = new_device
            console.print(f"[green]GPU selection: {new_device}[/green]")

            # Check tensor_parallel_size compatibility if already configured
            if config.get("tensor_parallel_size"):
                num_selected = len(new_device.split(","))
                tp_size = config["tensor_parallel_size"]

                if tp_size > num_selected:
                    console.print(
                        f"[yellow]Warning: tensor_parallel_size ({tp_size}) > selected GPUs ({num_selected})[/yellow]"
                    )
                    if inquirer.confirm(
                        f"Adjust tensor_parallel_size to {num_selected}?", default=True
                    ):
                        config["tensor_parallel_size"] = num_selected
                        console.print(
                            f"[green]tensor_parallel_size adjusted to {num_selected}[/green]"
                        )
                elif num_selected % tp_size != 0:
                    console.print(
                        f"[yellow]Note: {num_selected} GPUs with tensor_parallel_size={tp_size} "
                        f"may not utilize all GPUs efficiently[/yellow]"
                    )

        elif new_device is None and current_device:
            # User cleared the selection
            del config["device"]
            console.print("[yellow]GPU selection cleared[/yellow]")

    # Now ask about advanced options with hierarchical menu
    console.print()  # Add blank line for spacing
    configure_advanced = inquirer.confirm("Configure advanced options?", default=False)

    if configure_advanced:
        config = configure_advanced_hierarchical(config, config_manager)

    # Ask about environment variables - available for all configurations
    console.print()  # Add blank line for spacing
    configure_env = inquirer.confirm("Configure environment variables?", default=False)

    if configure_env:
        # For direct server configuration (has model), store as session_environment
        # For profile creation (no model), store as environment
        if base_config and "model" in base_config:
            # This is for Custom Configuration from main menu
            config["session_environment"] = configure_environment_variables(
                config.get("session_environment", {})
            )
        else:
            # This is for profile creation
            config["environment"] = configure_environment_variables(
                config.get("environment", {})
            )

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

        # Add GPU selection as the first option
        gpu_status = ""
        if "device" in config:
            gpu_status = f" [GPUs: {config['device']}]"
        category_choices.append(f"GPU Selection{gpu_status}")

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

        # Handle GPU selection
        if selected.startswith("GPU Selection"):
            current_device = config.get("device")
            new_device = select_gpus(current_device)
            if new_device is not None:
                config["device"] = new_device
                console.print(f"[green]GPU selection updated: {new_device}[/green]")

                # Check tensor_parallel_size compatibility
                if config.get("tensor_parallel_size"):
                    num_selected = len(new_device.split(","))
                    tp_size = config["tensor_parallel_size"]

                    if tp_size > num_selected:
                        console.print(
                            f"[yellow]Warning: tensor_parallel_size ({tp_size}) > selected GPUs ({num_selected})[/yellow]"
                        )
                        if inquirer.confirm(
                            f"Adjust tensor_parallel_size to {num_selected}?",
                            default=True,
                        ):
                            config["tensor_parallel_size"] = num_selected
                            console.print(
                                f"[green]tensor_parallel_size adjusted to {num_selected}[/green]"
                            )
                    elif num_selected % tp_size != 0:
                        console.print(
                            f"[yellow]Note: {num_selected} GPUs with tensor_parallel_size={tp_size} "
                            f"may not utilize all GPUs efficiently[/yellow]"
                        )

            elif new_device is None and current_device:
                # User cleared the selection
                del config["device"]
                console.print("[yellow]GPU selection cleared[/yellow]")
            continue

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

    # Exclude model field from profiles (it's selected separately when serving)
    args = [a for a in args if a["name"] != "model"]

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
        console.print()  # Add blank line for spacing
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

            # Check if specific GPUs are selected
            if config.get("device"):
                device_str = config["device"]
                num_gpus = len(device_str.split(","))
                console.print(
                    f"[dim]Using {num_gpus} selected GPU(s): {device_str}[/dim]"
                )
            else:
                try:
                    gpus = get_gpu_info()
                    num_gpus = len(gpus) if gpus else 1
                    console.print(f"[dim]Detected {num_gpus} GPU(s) in system[/dim]")
                except Exception:
                    num_gpus = 1

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
                # Suggest common divisors for better GPU utilization
                if num_gpus > 1:
                    divisors = [i for i in range(1, num_gpus + 1) if num_gpus % i == 0]
                    console.print(
                        f"[dim]Recommended values for {num_gpus} GPUs: {', '.join(map(str, divisors))}[/dim]"
                    )
                # Set default to detected GPU count
                default = num_gpus

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


def select_gpus(current_selection: Optional[str] = None) -> Optional[str]:
    """
    Interactive GPU selection with checkbox UI.

    Args:
        current_selection: Current GPU selection as comma-separated string

    Returns:
        Comma-separated GPU indices (e.g., "0,1,2") or None
    """
    # Get available GPUs
    gpu_info = gpu.get_gpu_info()

    if not gpu_info:
        console.print("[yellow]No GPUs detected[/yellow]")
        return None

    # Parse current selection
    selected_indices = []
    if current_selection:
        try:
            selected_indices = [
                int(idx.strip()) for idx in current_selection.split(",")
            ]
        except (ValueError, AttributeError):
            selected_indices = []

    # Build choices for checkbox
    gpu_choices = []
    for gpu_data in gpu_info:
        idx = gpu_data["index"]
        name = gpu_data["name"]
        memory_gb = gpu_data["memory_total"] / (1024**3)  # Convert bytes to GB

        choice_label = f"GPU {idx}: {name} ({memory_gb:.1f}GB)"
        gpu_choices.append((choice_label, idx, idx in selected_indices))

    # Show current selection if exists
    if current_selection:
        console.print(f"\n[dim]Current selection: {current_selection}[/dim]")

    console.print(
        "\n[dim]Select GPUs to use (space to select/deselect, enter to confirm)[/dim]"
    )

    # Create checkbox question
    questions = [
        inquirer.Checkbox(
            "gpus",
            message="Select GPUs to use",
            choices=[(label, idx) for label, idx, _ in gpu_choices],
            default=[idx for _, idx, selected in gpu_choices if selected],
        )
    ]

    answers = inquirer.prompt(questions)

    if not answers:
        return current_selection  # User cancelled

    selected_gpus = answers.get("gpus", [])

    if not selected_gpus:
        # Ask for confirmation if no GPUs selected
        if current_selection:
            if inquirer.confirm("Clear GPU selection?", default=False):
                return None
        else:
            console.print("[yellow]No GPUs selected, using default[/yellow]")
        return current_selection

    # Return comma-separated string
    return ",".join(str(idx) for idx in sorted(selected_gpus))


def configure_environment_variables(
    existing_env: Dict[str, str] = None,
) -> Dict[str, str]:
    """
    Configure environment variables for the profile.

    Args:
        existing_env: Existing environment variables

    Returns:
        Dictionary of environment variables
    """
    env = existing_env.copy() if existing_env else {}

    # Define environment variable presets by category
    # NO DEFAULT VALUES - let vLLM use its own defaults unless user explicitly sets
    env_presets = {
        "GPU/Hardware Optimization": {
            "VLLM_USE_TRITON_FLASH_ATTN": "Enable Triton flash attention (usually 1)",
            "VLLM_ATTENTION_BACKEND": "Attention backend (e.g., TRITON_ATTN_VLLM_V1 for A100)",
            "VLLM_USE_TRTLLM_ATTENTION": "TensorRT-LLM attention for Blackwell GPUs (usually 1)",
            "VLLM_FLASHINFER_FORCE_TENSOR_CORES": "Force tensor core usage (usually 1)",
        },
        "MoE Model Optimization": {
            "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16": "MoE with BF16 precision for GPT-OSS (usually 1)",
            "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": "MoE with FP8 precision (usually 1)",
            "VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING": "Enable MoE activation chunking (usually 1)",
            "VLLM_FUSED_MOE_CHUNK_SIZE": "MoE chunk size (e.g., 65536)",
        },
        "AMD/ROCm Optimization": {
            "VLLM_ROCM_USE_AITER": "Enable AMD async iterator (usually 1)",
            "VLLM_USE_AITER_UNIFIED_ATTENTION": "Unified attention for AMD GPUs (usually 1)",
            "VLLM_ROCM_USE_SKINNY_GEMM": "Enable skinny GEMM for ROCm (usually 1)",
            "VLLM_ROCM_FP8_PADDING": "FP8 padding for ROCm (usually 1)",
        },
        "Logging & Monitoring": {
            "VLLM_LOGGING_LEVEL": "Log level (DEBUG/INFO/WARNING/ERROR)",
            "VLLM_LOG_STATS_INTERVAL": "Stats logging interval in seconds",
            "VLLM_TRACE_FUNCTION": "Enable function tracing (usually 1)",
            "VLLM_CONFIGURE_LOGGING": "Enable vLLM logging configuration (usually 1)",
        },
        "Memory & Performance": {
            "VLLM_CPU_KVCACHE_SPACE": "CPU KV cache space in GiB",
            "VLLM_CPU_OMP_THREADS_BIND": "OpenMP thread binding",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "Allow long context lengths (usually 1)",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "Enable V1 multiprocessing (usually 1)",
        },
        "CUDA Configuration": {
            "CUDA_HOME": "Path to CUDA toolkit",
            "CUDA_VISIBLE_DEVICES": "Visible GPU devices (e.g., 0,1,2)",
            "TORCH_CUDA_ARCH_LIST": "CUDA architectures (e.g., 9.0)",
            "PYTHONUNBUFFERED": "Unbuffered Python output (usually 1)",
        },
    }

    while True:
        console.print("\n[bold cyan]Environment Variable Configuration[/bold cyan]")

        # Show current environment variables
        if env:
            console.print("\n[bold]Current Environment Variables:[/bold]")
            for key, value in env.items():
                if "KEY" in key.upper() or "TOKEN" in key.upper():
                    console.print(f"  • {key}: <hidden>")
                else:
                    console.print(f"  • {key}: {value}")

        # Build menu options
        menu_options = []
        for category in env_presets.keys():
            menu_options.append(f"Add from {category}")
        menu_options.extend(
            [
                "Add custom variable",
                "Remove variable",
                "Clear all variables",
                "✓ Done configuring",
            ]
        )

        choice = unified_prompt(
            "env_config", "Select action", menu_options, allow_back=False
        )

        if choice == "✓ Done configuring" or not choice:
            break
        elif choice == "Clear all variables":
            if env:
                confirm = inquirer.confirm(
                    "Clear all environment variables?", default=False
                )
                if confirm:
                    env.clear()
                    console.print("[green]All environment variables cleared.[/green]")
            else:
                console.print("[yellow]No environment variables to clear.[/yellow]")
        elif choice == "Remove variable":
            if env:
                var_choices = list(env.keys()) + ["← Back"]
                var_to_remove = unified_prompt(
                    "remove_env",
                    "Select variable to remove",
                    var_choices,
                    allow_back=False,
                )
                if var_to_remove and var_to_remove != "← Back":
                    del env[var_to_remove]
                    console.print(f"[green]Removed {var_to_remove}[/green]")
            else:
                console.print("[yellow]No environment variables to remove.[/yellow]")
        elif choice == "Add custom variable":
            console.print("\n[bold]Add Custom Environment Variable[/bold]")
            var_name = input("Variable name (e.g., VLLM_LOGGING_LEVEL): ").strip()
            if var_name:
                var_value = input(f"Value for {var_name}: ").strip()
                if var_value:
                    env[var_name] = var_value
                    console.print(f"[green]Added {var_name}={var_value}[/green]")
        elif choice.startswith("Add from "):
            # Extract category name
            category = choice.replace("Add from ", "")
            if category in env_presets:
                category_vars = env_presets[category]

                # Build list of variables in this category
                var_options = []
                for var_name, description in category_vars.items():
                    if var_name in env:
                        status = f" [green]✓ Set to: {env[var_name]}[/green]"
                    else:
                        status = ""
                    var_options.append(f"{var_name}: {description}{status}")
                var_options.append("← Back")

                selected_var = unified_prompt(
                    "select_env_var",
                    f"Select variable from {category}",  # nosec B608
                    var_options,
                    allow_back=False,
                )

                if selected_var and selected_var != "← Back":
                    # Extract variable name
                    var_name = selected_var.split(":")[0].strip()
                    if var_name in category_vars:
                        description = category_vars[var_name]

                        console.print(f"\n[bold]{var_name}[/bold]")
                        console.print(f"[dim]{description}[/dim]")

                        current = env.get(var_name, "")
                        if current:
                            prompt_text = f"Enter value (current: {current}): "
                        else:
                            prompt_text = (
                                "Enter value (leave empty to use vLLM default): "
                            )

                        new_value = input(prompt_text).strip()
                        if new_value:
                            env[var_name] = new_value
                            console.print(f"[green]Set {var_name}={new_value}[/green]")
                        elif current:
                            # Keep current value
                            console.print(
                                f"[dim]Keeping current value: {current}[/dim]"
                            )
                        else:
                            console.print("[dim]Will use vLLM default (not set)[/dim]")

    return env


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
    console.print()  # Add blank line for spacing
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
