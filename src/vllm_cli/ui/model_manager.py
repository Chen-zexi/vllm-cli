#!/usr/bin/env python3
"""
Model management module for vLLM CLI.

Handles model selection and delegates management to hf-model-tool.
"""
import logging
from typing import Any, Dict, Optional

import inquirer
from rich.panel import Panel

from ..models import list_available_models
from ..system import format_size
from .common import console
from .navigation import unified_prompt

logger = logging.getLogger(__name__)


def enter_remote_model() -> Optional[str]:
    """
    Allow user to enter a HuggingFace model ID for remote serving.

    Returns:
        Model ID string or None if cancelled
    """
    import getpass

    from ..config import ConfigManager

    console.print("\n[bold cyan]Remote Model Selection[/bold cyan]")
    console.print("\nEnter a HuggingFace model ID to serve directly from the Hub.")
    console.print("The model will be automatically downloaded on first use.\n")

    console.print("[dim]Examples:[/dim]")
    console.print("  • openai/gpt-oss-120b\n")

    console.print(
        "[yellow]Note:[/yellow] First-time download may take 10-30 minutes depending on model size.\n"
    )

    while True:
        console.print("[cyan]Model ID (or 'back' to cancel):[/cyan] ", end="")
        model_id = input().strip()

        if model_id.lower() in ["back", "cancel", ""]:
            return None

        # Validate model ID format
        if "/" not in model_id:
            console.print(
                "[red]Invalid format. Model ID should be 'organization/model-name'[/red]"
            )
            continue

        parts = model_id.split("/")
        if len(parts) != 2:
            console.print(
                "[red]Invalid format. Model ID should be 'organization/model-name'[/red]"
            )
            continue

        org, model_name = parts
        if not org or not model_name:
            console.print(
                "[red]Invalid model ID. Both organization and model name are required.[/red]"
            )
            continue

        # Confirm the selection
        console.print(f"\n[bold]Selected model:[/bold] {model_id}")

        # Check for HF token and offer to configure if needed
        config_manager = ConfigManager()
        has_token = bool(config_manager.config.get("hf_token"))

        if has_token:
            console.print("[green]✓ HuggingFace token is configured[/green]")
        else:
            console.print(
                "\n[dim]Note: Some models require a HuggingFace token for access.[/dim]"
            )
            console.print(
                "[dim]If this model is gated, you'll need to provide a token.[/dim]\n"
            )

            token_options = ["Configure HF token now", "Continue without token"]

            token_action = unified_prompt(
                "token_action",
                "Would you like to configure a HuggingFace token?",
                token_options,
            )

            if token_action == "Configure HF token now":
                console.print("\n[cyan]Enter your HuggingFace token:[/cyan]")
                console.print(
                    "[dim]Get your token from: https://huggingface.co/settings/tokens[/dim]"
                )
                console.print("[dim]The token will be hidden as you type.[/dim]\n")

                token = getpass.getpass("Token: ").strip()
                if token:
                    console.print("\n[cyan]Validating token...[/cyan]")

                    from ..validation.token import validate_hf_token

                    is_valid, user_info = validate_hf_token(token)

                    if is_valid:
                        config_manager.config["hf_token"] = token
                        config_manager._save_config()
                        console.print(
                            "[green]✓ Token validated and saved successfully[/green]"
                        )
                        if user_info:
                            console.print(
                                f"[dim]Authenticated as: {user_info.get('name', 'Unknown')}[/dim]\n"
                            )
                    else:
                        console.print("[red]✗ Token validation failed[/red]")
                        console.print("[dim]The token may be invalid or expired.[/dim]")

                        # Ask if they want to continue anyway
                        confirm = (
                            input("\nContinue with this token anyway? (y/N): ")
                            .strip()
                            .lower()
                        )
                        if confirm == "y":
                            config_manager.config["hf_token"] = token
                            config_manager._save_config()
                            console.print(
                                "[yellow]Token saved (but may not work)[/yellow]\n"
                            )
                        else:
                            console.print("[yellow]Continuing without token[/yellow]\n")
                else:
                    console.print(
                        "[yellow]No token provided, continuing without token[/yellow]\n"
                    )

        console.print(
            "\n[yellow]Warning:[/yellow] This model will be downloaded from HuggingFace Hub."
        )
        console.print(
            "Download size can range from a few GB to 100+ GB depending on the model.\n"
        )

        console.print("[cyan]Proceed with this model? (Y/n):[/cyan] ", end="")
        confirm = input().strip().lower()

        if confirm in ["", "y", "yes"]:
            logger.info(f"User selected remote model: {model_id}")
            return model_id
        elif confirm in ["n", "no"]:
            console.print("[yellow]Model selection cancelled.[/yellow]")
            continue
        else:
            console.print("[red]Invalid response. Please enter 'y' or 'n'.[/red]")
            continue


def select_model() -> Optional[Any]:
    """
    Select a model from available models with provider categorization.
    Can return either a string (model name) or a dict (model with LoRA config).
    """
    console.print("\n[bold cyan]Model Selection[/bold cyan]")

    try:
        # First, ask if user wants to use local or remote model
        model_source_choices = [
            "Select from local models",
            "Serve model with LoRA adapters",
            "Use a model from HuggingFace Hub (auto-download)",
        ]

        source_choice = unified_prompt(
            "model_source",
            "How would you like to select a model?",
            model_source_choices,
            allow_back=True,
        )

        if not source_choice or source_choice == "BACK":
            return None

        if source_choice == "Use a model from HuggingFace Hub (auto-download)":
            return enter_remote_model()

        if source_choice == "Serve model with LoRA adapters":
            return select_model_with_lora()

        # Continue with local model selection
        console.print("\n[bold cyan]Fetching available models...[/bold cyan]")
        models = list_available_models()

        if not models:
            console.print("[yellow]No local models found.[/yellow]")
            console.print("\nYou can either:")
            console.print("  1. Download models using HuggingFace tools")
            console.print("  2. Go back and select 'Use a model from HuggingFace Hub'")
            input("\nPress Enter to continue...")
            return None

        # Group models by provider
        providers_dict = {}
        for model in models:
            # Special handling for Ollama models
            if model.get("type") == "ollama_model":
                provider = "ollama"
            else:
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

        # Separate local provider from others
        local_provider = None
        other_providers = []

        for provider in providers_dict.keys():
            if provider == "local":
                local_provider = provider
            else:
                other_providers.append(provider)

        # Sort other providers alphabetically
        other_providers.sort()

        # Build provider choices with separation
        provider_choices = []

        # Add other providers first
        for provider in other_providers:
            count = len(providers_dict[provider])
            provider_choices.append(
                f"{provider} ({count} model{'s' if count > 1 else ''})"
            )

        # Add separator and local provider at the bottom if it exists
        if local_provider:
            # Add separator if there are other providers
            if other_providers:
                provider_choices.append("─" * 30)
            count = len(providers_dict[local_provider])
            provider_choices.append(
                f"{local_provider} ({count} model{'s' if count > 1 else ''})"
            )

        # Count total providers (excluding separator)
        total_providers = len(providers_dict)

        selected_provider = unified_prompt(
            "provider",
            f"Select Provider ({total_providers} available)",
            provider_choices,
            allow_back=True,
        )

        if not selected_provider or selected_provider == "BACK":
            return None

        # Check if separator was selected (shouldn't happen, but handle it)
        if selected_provider.startswith("─"):
            # Retry selection
            return select_model()

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
                display_name = display_name[len(provider_name) + 1 :]  # noqa: E203
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

        # Find the full model and return appropriate identifier
        for model in provider_models:
            check_name = model["name"]
            if check_name.startswith(f"{provider_name}/"):
                check_name = check_name[len(provider_name) + 1 :]  # noqa: E203
            if check_name == model_display_name or model["name"] == model_display_name:
                # Special handling for Ollama models
                if model.get("type") == "ollama_model":
                    console.print("\n[yellow]⚠ Warning: Ollama GGUF Model[/yellow]")
                    console.print(
                        "GGUF support in vLLM is experimental and varies by model architecture."
                    )
                    console.print(
                        "\n[cyan]Important:[/cyan] Not all GGUF models are supported."
                    )
                    console.print("\nFor compatibility information, see:")
                    console.print(
                        "  • vLLM-CLI Guide: [cyan]https://github.com/Chen-zexi/vllm-cli/blob/main/docs/ollama-integration.md[/cyan]"
                    )
                    console.print(
                        "  • vLLM Docs: [cyan]https://docs.vllm.ai/en/latest/models/supported_models.html[/cyan]"
                    )

                    console.print(
                        "\n[cyan]Continue with this model? (Y/n):[/cyan] ", end=""
                    )
                    confirm = input().strip().lower()
                    if confirm not in ["", "y", "yes"]:
                        return select_model()  # Go back to selection

                    # Return the GGUF file path with all metadata including name
                    return {
                        "model": model["path"],
                        "path": model["path"],  # Include path
                        "served_model_name": model[
                            "name"
                        ],  # Use correct field name for vLLM
                        "type": "ollama_model",  # Preserve type
                        "quantization": "gguf",
                        "experimental": True,
                    }

                # For custom models, always use path regardless of publisher
                # Custom models are identified by their type
                if model.get("type") == "custom_model" and model.get("path"):
                    return model["path"]

                # For non-HF pattern models (local), also use path
                model_name = model["name"]
                if model.get("path") and (
                    "/" not in model_name  # No HF org/model pattern
                    or model_name.startswith("/")  # Absolute path
                    or model.get("publisher")
                    in ["local", "unknown", None]  # Local publisher
                ):
                    return model["path"]
                return model["name"]

        # Fallback
        return model_display_name

    except Exception as e:
        logger.error(f"Error selecting model: {e}")
        console.print(f"[red]Error selecting model: {e}[/red]")
        return None


def handle_model_management() -> str:
    """
    Handle model management operations by delegating to hf-model-tool.
    All model management is handled by hf-model-tool for consistency.
    """
    import os
    import subprocess

    while True:  # Loop to stay in model management menu
        # Simplified menu - delegate everything to hf-model-tool
        management_options = [
            "Open Model Management Tool",  # Full hf-model-tool interface
            "List All Models",  # hf-model-tool --list
            "Manage Assets",  # hf-model-tool --manage
            "View Model Details",  # hf-model-tool --details
            "Refresh Model Cache",  # Refresh the cache used by serving menu
        ]

        action = unified_prompt(
            "model_management", "Model Management", management_options, allow_back=True
        )

        if not action or action == "BACK":
            return "continue"  # Return to main menu

        if action == "Open Model Management Tool":
            # Launch full hf-model-tool interface
            console.print(
                Panel(
                    "[bold cyan]Launching HF-Model-Tool[/bold cyan]\n"
                    "[dim]Full model management interface[/dim]",
                    border_style="blue",
                )
            )
            try:
                subprocess.run(["hf-model-tool"], env=os.environ.copy())
            except FileNotFoundError:
                console.print("[red]hf-model-tool not found. Please install it:[/red]")
                console.print("  pip install hf-model-tool")
            except Exception as e:
                console.print(f"[red]Error launching hf-model-tool: {e}[/red]")
            input("\nPress Enter to continue...")
            # Continue loop - stay in model management menu

        elif action == "List All Models":
            # Launch hf-model-tool in list mode
            console.print(
                Panel(
                    "[bold cyan]Model List[/bold cyan]\n"
                    "[dim]Displaying all discovered models[/dim]",
                    border_style="blue",
                )
            )
            try:
                subprocess.run(["hf-model-tool", "--list"], env=os.environ.copy())
            except FileNotFoundError:
                console.print("[red]hf-model-tool not found. Please install it:[/red]")
                console.print("  pip install hf-model-tool")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            input("\nPress Enter to continue...")
            # Continue loop - stay in model management menu

        elif action == "Manage Assets":
            # Launch hf-model-tool in manage mode
            console.print(
                Panel(
                    "[bold cyan]Asset Management[/bold cyan]\n"
                    "[dim]Delete, deduplicate, and organize models[/dim]",
                    border_style="blue",
                )
            )
            try:
                subprocess.run(["hf-model-tool", "--manage"], env=os.environ.copy())
            except FileNotFoundError:
                console.print("[red]hf-model-tool not found. Please install it:[/red]")
                console.print("  pip install hf-model-tool")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            input("\nPress Enter to continue...")
            # Continue loop - stay in model management menu

        elif action == "View Model Details":
            # Launch hf-model-tool in details mode
            console.print(
                Panel(
                    "[bold cyan]Model Details[/bold cyan]\n"
                    "[dim]View detailed information about models[/dim]",
                    border_style="blue",
                )
            )
            try:
                subprocess.run(["hf-model-tool", "--details"], env=os.environ.copy())
            except FileNotFoundError:
                console.print("[red]hf-model-tool not found. Please install it:[/red]")
                console.print("  pip install hf-model-tool")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
            input("\nPress Enter to continue...")
            # Continue loop - stay in model management menu

        elif action == "Refresh Model Cache":
            # Refresh the model cache used by serving menu
            console.print(
                Panel(
                    "[bold cyan]Refreshing Model Cache[/bold cyan]\n"
                    "[dim]Scanning all model directories for updates...[/dim]",
                    border_style="blue",
                )
            )

            try:
                import time

                from ..models import get_model_manager

                # Get the model manager
                model_manager = get_model_manager()

                # Get old cache stats for comparison
                old_stats = model_manager.get_cache_stats()
                old_count = old_stats.get("cached_models_count", 0)

                # Show scanning progress
                console.print("\n[cyan]Step 1/3:[/cyan] Refreshing model registry...")

                # Refresh the cache
                model_manager.refresh_cache()

                console.print("[cyan]Step 2/3:[/cyan] Clearing old cache...")
                time.sleep(0.1)  # Brief pause for visual feedback

                console.print("[cyan]Step 3/3:[/cyan] Loading fresh model data...")
                time.sleep(0.1)  # Brief pause for visual feedback

                # Get new cache stats to show user
                new_stats = model_manager.get_cache_stats()
                new_count = new_stats.get("cached_models_count", 0)

                # Show success message with details
                console.print("\n[green]✓ Model cache refreshed successfully![/green]")
                console.print(
                    "[dim]The serving menu will now show all current models.[/dim]"
                )

                # Show model count changes
                if old_count != new_count:
                    diff = new_count - old_count
                    if diff > 0:
                        console.print(
                            f"\n[cyan]Models in cache: {new_count} (+{diff} new)[/cyan]"
                        )
                    else:
                        console.print(
                            f"\n[cyan]Models in cache: {new_count} ({diff} removed)[/cyan]"
                        )
                else:
                    console.print(
                        f"\n[cyan]Models in cache: {new_count} (no changes)[/cyan]"
                    )

                # Show cache freshness
                console.print(
                    f"[dim]Cache TTL: {new_stats.get('ttl_seconds', 30)} seconds[/dim]"
                )

            except Exception as e:
                console.print(f"[red]Error refreshing cache: {e}[/red]")
                logger.error(f"Cache refresh error: {e}")

            input("\nPress Enter to continue...")
            # Continue loop - stay in model management menu


def select_model_with_lora() -> Optional[Dict[str, Any]]:
    """
    Select a base model and LoRA adapters to serve together.

    Returns:
        Dictionary with model and LoRA configuration for serving
    """
    console.print("\n[bold cyan]Model + LoRA Selection[/bold cyan]")
    console.print(
        "[dim]Select a base model and LoRA adapters to serve together[/dim]\n"
    )

    # First scan for LoRA adapters to determine which models have adapters
    console.print("[cyan]Scanning for LoRA adapters...[/cyan]")
    try:
        from ..models.discovery import scan_for_lora_adapters

        lora_adapters = scan_for_lora_adapters()

        if not lora_adapters:
            console.print("[yellow]No LoRA adapters found.[/yellow]")
            console.print("\nLoRA adapters should be placed in directories with:")
            console.print("  • adapter_config.json")
            console.print("  • adapter_model.safetensors or adapter_model.bin")
            console.print("\nYou can manage LoRA adapters using hf-model-tool")
            input("\nPress Enter to continue...")
            return None

        console.print(f"[green]Found {len(lora_adapters)} LoRA adapter(s)[/green]\n")

        # Extract base models from LoRA metadata
        base_models_with_lora = set()
        lora_by_base = {}

        for lora in lora_adapters:
            # First try metadata which should have the exact base model
            metadata = lora.get("metadata", {})
            base_model = metadata.get("base_model", "")

            # If not in metadata, try config
            if not base_model:
                config = lora.get("config", {})
                if isinstance(config, dict):
                    base_model = config.get("base_model_name_or_path", "")

            # Store the mapping if we found a base model
            if base_model:
                base_models_with_lora.add(base_model)
                if base_model not in lora_by_base:
                    lora_by_base[base_model] = []
                lora_by_base[base_model].append(lora)

        logger.debug(
            f"Found LoRA adapters for these base models: {base_models_with_lora}"
        )

        # Get all available models
        models = list_available_models()

        if not models:
            console.print("[yellow]No local models found.[/yellow]")
            console.print("Please download models first using hf-model-tool")
            input("\nPress Enter to continue...")
            return None

        # Filter models to only show those with LoRA adapters
        models_with_lora = []
        for model in models:
            model_name = model["name"]
            # Check if this model has LoRA adapters
            has_lora = False

            # First try exact match
            if model_name in base_models_with_lora:
                has_lora = True

            # If no exact match, don't do fuzzy matching to avoid false positives
            # Only show models that have exact LoRA matches

            if has_lora:
                models_with_lora.append(model)

        # If no models with LoRA found, show all and let user know
        if not models_with_lora:
            console.print(
                "[yellow]No models found with matching LoRA adapters.[/yellow]"
            )
            console.print(
                "Showing all models. LoRA compatibility will be checked after selection.\n"
            )
            models_with_lora = models

        # Select the base model
        console.print("[cyan]Step 1: Select Base Model[/cyan]")
        console.print(
            f"[dim]Showing {len(models_with_lora)} model(s) with LoRA adapters[/dim]\n"
        )

        model_choices = []
        for model in models_with_lora:
            size_str = format_size(model.get("size", 0))
            # Check how many LoRAs this model has
            model_name = model["name"]
            lora_count = 0

            # Use exact matching for counting
            if model_name in lora_by_base:
                lora_count = len(lora_by_base[model_name])

            if lora_count > 0:
                model_choices.append(
                    f"{model_name} ({size_str}) [{lora_count} LoRA(s)]"
                )
            else:
                model_choices.append(f"{model_name} ({size_str})")

        selected_model = unified_prompt(
            "base_model", "Select Base Model", model_choices, allow_back=True
        )

        if not selected_model or selected_model == "BACK":
            return None

        # Extract model name
        base_model_name = selected_model.split(" (")[0]

        # Find the actual model object to get its path if it's a local/custom model
        base_model_path_or_name = base_model_name
        for model in models_with_lora:
            if model["name"] == base_model_name:
                # For custom models, always use path regardless of publisher
                if model.get("type") == "custom_model" and model.get("path"):
                    base_model_path_or_name = model["path"]
                # For non-HF pattern models, use the path instead of name
                elif model.get("path") and (
                    "/" not in base_model_name  # No HF org/model pattern
                    or base_model_name.startswith("/")  # Absolute path
                    or model.get("publisher")
                    in ["local", "unknown", None]  # Local publisher
                ):
                    base_model_path_or_name = model["path"]
                break

        # Now select LoRA adapters for this model
        console.print("\n[cyan]Step 2: Select LoRA Adapters[/cyan]")
        console.print(f"[dim]Base model: {base_model_name}[/dim]\n")

        # Filter compatible LoRAs for the selected model
        compatible_loras = []
        incompatible_loras = []

        for lora in lora_adapters:
            metadata = lora.get("metadata", {})
            lora_base = metadata.get("base_model", "unknown")

            # Simple compatibility check
            if (
                lora_base == "unknown"
                or lora_base in base_model_name
                or base_model_name in lora_base
            ):
                compatible_loras.append(lora)
            else:
                incompatible_loras.append(lora)

        if not compatible_loras and incompatible_loras:
            console.print(
                "[yellow]No clearly compatible LoRA adapters found for this model.[/yellow]"
            )
            if inquirer.confirm("Show all LoRA adapters anyway?", default=True):
                compatible_loras = incompatible_loras
            else:
                return base_model_name

        # Create LoRA choices
        lora_choices = []
        for lora in compatible_loras:
            name = lora.get("name", lora.get("display_name", "Unknown"))
            rank = lora.get("rank", lora.get("metadata", {}).get("rank", "N/A"))
            size = lora.get("size", 0)
            size_str = f"{size / (1024**2):.1f}MB" if size > 0 else "N/A"
            lora_choices.append(f"{name} (rank={rank}, {size_str})")

        # Allow multiple LoRA selection
        console.print(
            "[dim]You can select multiple LoRA adapters (space to select, enter to confirm)[/dim]"
        )

        questions = [
            inquirer.Checkbox(
                "loras", message="Select LoRA adapters", choices=lora_choices
            )
        ]

        answers = inquirer.prompt(questions)
        if not answers or not answers["loras"]:
            if inquirer.confirm("Continue without LoRA adapters?", default=False):
                return base_model_name
            return None

        # Build the configuration
        selected_lora_configs = []
        for lora_choice in answers["loras"]:
            # Find the matching LoRA
            for lora in compatible_loras:
                name = lora.get("name", lora.get("display_name", "Unknown"))
                if name in lora_choice:
                    selected_lora_configs.append(
                        {
                            "name": name,
                            "path": lora.get("path", ""),
                            "rank": lora.get("rank", 16),
                        }
                    )
                    break

        # Return configuration for serving
        return {"model": base_model_path_or_name, "lora_modules": selected_lora_configs}

    except Exception as e:
        console.print(f"[red]Error selecting LoRA adapters: {e}[/red]")
        logger.error(f"LoRA selection error: {e}")

        if inquirer.confirm("Continue with base model only?", default=True):
            return base_model_path_or_name
        return None


# Removed view_available_models and show_model_details functions
# These are now handled by hf-model-tool --list and --details
