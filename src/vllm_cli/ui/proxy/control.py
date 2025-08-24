#!/usr/bin/env python3
"""
UI components for proxy server control and configuration.
"""
import logging
from typing import List, Optional

from rich.table import Table

from ...proxy.config import ProxyConfigManager
from ...proxy.models import ModelConfig, ProxyConfig
from ..common import console
from ..custom_config import select_gpus
from ..model_manager import select_model
from ..navigation import unified_prompt

logger = logging.getLogger(__name__)


def display_configured_models(models: List[ModelConfig]):
    """
    Display a summary table of configured models.

    Args:
        models: List of configured ModelConfig instances
    """
    if not models:
        return

    table = Table(title=f"Configured Models ({len(models)})")
    table.add_column("#", style="dim", width=3)
    table.add_column("Model", style="cyan")
    table.add_column("Port", style="magenta")
    table.add_column("GPU(s)", style="yellow")
    table.add_column("Profile", style="blue")

    for idx, model in enumerate(models, 1):
        gpu_str = ",".join(str(g) for g in model.gpu_ids) if model.gpu_ids else "Auto"
        profile_str = model.profile or "None"
        table.add_row(
            str(idx),
            model.name[:30] + "..." if len(model.name) > 30 else model.name,
            str(model.port),
            gpu_str,
            profile_str,
        )

    console.print("\n")
    console.print(table)
    console.print()


def configure_proxy_interactively() -> Optional[ProxyConfig]:
    """
    Configure proxy server interactively.

    Returns:
        ProxyConfig instance or None if cancelled
    """
    console.print("\n[bold cyan]Configure Multi-Model Proxy Server[/bold cyan]")

    # Proxy settings
    console.print("\n[bold]Proxy Server Settings[/bold]")

    host = input("Host (default: 0.0.0.0): ").strip() or "0.0.0.0"  # nosec B104
    port_str = input("Port (default: 8000): ").strip() or "8000"

    try:
        port = int(port_str)
    except ValueError:
        console.print("[red]Invalid port number[/red]")
        return None

    # CORS and metrics
    enable_cors = (
        unified_prompt("enable_cors", "Enable CORS?", ["Yes", "No"], allow_back=False)
        == "Yes"
    )

    enable_metrics = (
        unified_prompt(
            "enable_metrics",
            "Enable metrics endpoint?",
            ["Yes", "No"],
            allow_back=False,
        )
        == "Yes"
    )

    log_requests = (
        unified_prompt(
            "log_requests", "Log all requests?", ["Yes", "No"], allow_back=False
        )
        == "Yes"
    )

    # Configure models
    console.print("\n[bold]Configure Models[/bold]")
    models = []

    while True:
        # Display currently configured models if any
        if models:
            display_configured_models(models)

        # Update prompt based on number of configured models
        prompt_msg = "Model configuration"
        if models:
            prompt_msg = f"Model configuration ({len(models)} model{'s' if len(models) > 1 else ''} configured)"

        action = unified_prompt(
            "model_action",
            prompt_msg,
            ["Add a model", "✓ Done configuring models"],
            allow_back=False,
        )

        if action == "✓ Done configuring models":
            break

        # Add a model - pass existing models for conflict checking
        model_config = configure_model_for_proxy(
            len(models), existing_models=models, is_running_proxy=False
        )
        if model_config:
            models.append(model_config)
            console.print(f"\n[green]✓ Added model: {model_config.name}[/green]")

    if not models:
        console.print(
            "[yellow]No models configured. Proxy needs at least one model.[/yellow]"
        )
        return None

    return ProxyConfig(
        host=host,
        port=port,
        models=models,
        enable_cors=enable_cors,
        enable_metrics=enable_metrics,
        log_requests=log_requests,
    )


def _check_parallel_settings_conflict(
    profile: str, gpu_ids: List[int], config_manager
) -> None:
    """
    Check and warn about parallel settings conflicts with GPU selection.

    Args:
        profile: Profile name
        gpu_ids: List of selected GPU IDs
        config_manager: ConfigManager instance
    """
    if profile and len(gpu_ids) == 1:
        all_profiles = config_manager.get_all_profiles()
        profile_config = all_profiles.get(profile, {}).get("config", {})
        parallel_settings = []

        if profile_config.get("tensor_parallel_size"):
            parallel_settings.append(
                f"tensor_parallel_size={profile_config['tensor_parallel_size']}"
            )
        if profile_config.get("pipeline_parallel_size"):
            parallel_settings.append(
                f"pipeline_parallel_size={profile_config['pipeline_parallel_size']}"
            )
        if profile_config.get("enable_expert_parallel"):
            parallel_settings.append("enable_expert_parallel=True")

        if parallel_settings:
            console.print("\n[yellow]⚠ Warning:[/yellow]")
            console.print(
                f"Profile '{profile}' contains parallel settings: {', '.join(parallel_settings)}"
            )
            console.print(
                "These settings will be automatically disabled for single-GPU deployment."
            )
            console.print(
                "[dim]The model will run on a single GPU as configured.[/dim]\n"
            )


def configure_model_for_proxy(
    index: int,
    existing_models: Optional[List[ModelConfig]] = None,
    is_running_proxy: bool = False,
    proxy_manager=None,
) -> Optional[ModelConfig]:
    """
    Unified function to configure a model for proxy serving.

    This function provides a consistent UI experience for both:
    - Initial proxy configuration (multiple models)
    - Adding models to a running proxy

    Args:
        index: Index of this model (for default port calculation)
        existing_models: List of already configured models (for conflict checking)
        is_running_proxy: Whether configuring for a running proxy (affects prompts)

    Returns:
        ModelConfig instance or None if cancelled
    """
    if existing_models is None:
        existing_models = []

    # Display appropriate header
    if is_running_proxy:
        console.print("\n[bold cyan]Add New Model[/bold cyan]\n")
    else:
        console.print(f"\n[cyan]Configure Model #{index + 1}[/cyan]")

    # Import here to avoid circular dependency
    from ...config import ConfigManager
    from ..model_manager import select_shortcut_for_serving
    from ..navigation import unified_prompt

    config_manager = ConfigManager()
    shortcuts = config_manager.list_shortcuts()

    # First ask if user wants to use shortcut or select model
    model_source_choices = []
    if shortcuts:
        model_source_choices.append("Use saved shortcut")
    model_source_choices.extend(
        [
            "Select from models",
        ]
    )

    source_choice = unified_prompt(
        "model_source_proxy",
        "How would you like to configure this model?",
        model_source_choices,
        allow_back=True,
    )

    if not source_choice or source_choice == "BACK":
        return None

    model_selection = None
    if source_choice == "Use saved shortcut":
        # Handle shortcut selection directly
        model_selection = select_shortcut_for_serving()
    else:
        # Select model using existing UI
        model_selection = select_model()

    if not model_selection:
        return None

    # Process model selection (handles shortcuts, ollama, lora, etc.)
    model_path = None
    model_name = None
    shortcut_profile = None
    model_config_overrides = {}
    is_shortcut = False

    if isinstance(model_selection, dict):
        if model_selection.get("type") == "shortcut":
            # Shortcut selected
            is_shortcut = True
            model_path = model_selection["model"]
            model_name = model_path
            shortcut_profile = model_selection["profile"]
            shortcut_name = model_selection["name"]
            model_config_overrides = model_selection.get("config_overrides", {})

            console.print(f"\n[bold cyan]Using Shortcut: {shortcut_name}[/bold cyan]")
            console.print(f"[green]Model: {model_name}[/green]")
            console.print(f"[blue]Profile: {shortcut_profile}[/blue]")

        elif model_selection.get("type") == "ollama_model":
            # Ollama/GGUF model
            model_path = model_selection.get("model", model_selection.get("path"))
            model_name = model_selection.get("name", model_path)
            if model_selection.get("served_model_name"):
                model_config_overrides["served_model_name"] = model_selection[
                    "served_model_name"
                ]
            model_config_overrides["quantization"] = "gguf"

        elif "lora_modules" in model_selection:
            # LoRA configuration
            model_path = model_selection["model"]
            lora_modules = model_selection["lora_modules"]
            model_name = model_path.split("/")[-1] if "/" in model_path else model_path
            model_config_overrides["enable_lora"] = True
            model_config_overrides["lora_modules"] = lora_modules

        else:
            # Other dict format
            model_path = model_selection.get("model", model_selection.get("path"))
            model_name = model_selection.get(
                "name", model_path.split("/")[-1] if "/" in model_path else model_path
            )
    else:
        # Simple string model name/path
        model_path = model_selection
        model_name = model_path.split("/")[-1] if "/" in model_path else model_path

    # Display the model (only if not a shortcut, as shortcuts already display it)
    if not is_shortcut:
        console.print(f"\n[green]Model: {model_name}[/green]")

    # Optional: Allow user to provide an alias
    alias = None
    use_alias = (
        unified_prompt(
            "use_alias",
            "Would you like to add an alias for this model?",
            ["No, use model path/name", "Yes, add an alias"],
            allow_back=False,
        )
        == "Yes, add an alias"
    )

    if use_alias:
        alias_input = input("Enter alias (optional, press Enter to skip): ").strip()
        if alias_input:
            alias = alias_input
            console.print(
                f"[dim]Note: Both '{model_name}' and '{alias}' will route to this model[/dim]"
            )

    # GPU assignment
    console.print("\n[bold]GPU Assignment[/bold]")

    # Get gpu_memory_utilization from profile if available
    required_utilization = None
    if is_shortcut and shortcut_profile:
        # For shortcuts, we know the profile
        from ...config import ConfigManager

        config_manager = ConfigManager()
        profile_data = config_manager.get_profile(shortcut_profile)
        if profile_data and "config" in profile_data:
            required_utilization = profile_data["config"].get(
                "gpu_memory_utilization", 0.9
            )
            console.print(
                f"[dim]Profile '{shortcut_profile}' uses {required_utilization*100:.0f}% GPU memory[/dim]"
            )

    # Show warning if using shortcut with parallel settings
    if is_shortcut:
        console.print(
            "[yellow]⚠ Warning:[/yellow] GPU selection may override profile settings"
        )
        console.print(
            f"[dim]Profile '{shortcut_profile}' may have tensor_parallel_size or pipeline_parallel_size[/dim]"
        )
        console.print(
            "[dim]Your GPU selection will determine the actual parallelism used[/dim]\n"
        )

    gpu_str = select_gpus(
        current_selection=None,
        proxy_manager=proxy_manager,
        required_utilization=required_utilization,
    )
    gpu_ids = []
    if gpu_str:
        try:
            gpu_ids = [int(g.strip()) for g in gpu_str.split(",")]
        except ValueError:
            console.print(
                "[yellow]Invalid GPU IDs, using automatic assignment[/yellow]"
            )
    else:
        console.print("[dim]No GPUs selected, will use automatic assignment[/dim]")

    # Port selection with conflict checking
    console.print("\n[bold]Port Selection[/bold]")

    # Build port usage map from existing models with running status if proxy is running
    used_ports = {}
    for model in existing_models:
        used_ports[model.port] = {"name": model.name, "model": model}

    if used_ports:
        console.print("\n[dim]Currently configured ports:[/dim]")
        for port, info in sorted(used_ports.items()):
            # For running proxy, check if model is actually running
            if is_running_proxy:
                # For running proxy, just show the port is in use
                # The actual running status will be checked by the caller
                console.print(f"  Port {port}: {info['name']}")
            else:
                console.print(f"  Port {port}: {info['name']}")

    # Suggest next available port
    default_port = 8001 + index
    # Find next available port if default is taken
    while default_port in used_ports:
        default_port += 1

    port_str = input(f"Port (default: {default_port}): ").strip() or str(default_port)
    try:
        port = int(port_str)
        # Check for port conflicts
        if port in used_ports:
            if is_running_proxy:
                # For running proxy, the caller will handle port reuse for stopped models
                console.print(
                    f"[yellow]Port {port} is configured for '{used_ports[port]['name']}'[/yellow]"
                )
                console.print(
                    "[dim]Note: If this model is stopped, it will be replaced.[/dim]"
                )
            else:
                # For initial config, don't allow duplicate ports
                console.print(
                    f"[red]Port {port} is already in use by '{used_ports[port]['name']}'[/red]"
                )
                console.print("[yellow]Please choose a different port[/yellow]")
                return None
    except ValueError:
        console.print(f"[yellow]Invalid port, using default: {default_port}[/yellow]")
        port = default_port

    # Profile selection
    profile = None
    if is_shortcut:
        # Use the profile from the shortcut
        profile = shortcut_profile
        console.print(f"\n[dim]Using profile from shortcut: {profile}[/dim]")
    else:
        # Normal profile selection for non-shortcut models
        from ...config import ConfigManager

        config_manager = ConfigManager()
        all_profiles = config_manager.get_all_profiles()

        if all_profiles:
            profile_names = list(all_profiles.keys())
            profile_names.append("No profile")

            selected_profile = unified_prompt(
                "profile_selection", "Select profile", profile_names, allow_back=False
            )

            profile = None if selected_profile == "No profile" else selected_profile

            # If profile selected and no GPUs selected yet, update required_utilization
            if profile and not gpu_ids:
                profile_data = config_manager.get_profile(profile)
                if profile_data and "config" in profile_data:
                    profile_util = profile_data["config"].get(
                        "gpu_memory_utilization", 0.9
                    )
                    console.print(
                        f"\n[yellow]Note: Selected profile uses {profile_util*100:.0f}% GPU memory[/yellow]"
                    )
                    console.print(
                        "[dim]You may want to reconsider GPU selection based on this requirement[/dim]"
                    )
        else:
            profile = None

    # Check for parallel settings conflicts
    if profile:
        from ...config import ConfigManager

        config_manager = ConfigManager()
        _check_parallel_settings_conflict(profile, gpu_ids, config_manager)

    # Create model configuration
    config = ModelConfig(
        name=model_name,
        model_path=model_path,
        gpu_ids=gpu_ids,
        port=port,
        profile=profile,
        enabled=True,
        config_overrides=model_config_overrides,
    )

    # Add alias to config_overrides if provided
    if alias:
        config.config_overrides["aliases"] = [alias]

    return config


def configure_model_interactively(index: int) -> Optional[ModelConfig]:
    """
    Configure a single model interactively.

    This is a wrapper around configure_model_for_proxy for backward compatibility
    and initial proxy configuration context.

    Args:
        index: Index of this model (for default port calculation)

    Returns:
        ModelConfig instance or None if cancelled
    """
    # During initial proxy configuration, we don't have existing models yet
    # The configure_proxy_interactively function will accumulate them
    return configure_model_for_proxy(index, existing_models=[], is_running_proxy=False)


# Note: manage_proxy_configs has been moved to settings.py for consistency
# with other configuration management (profiles, shortcuts)


def edit_proxy_config():
    """
    Edit proxy configuration interactively.

    Loads the current configuration and provides an interactive
    editing interface.

    Returns:
        Modified ProxyConfig or None if cancelled
    """
    config_manager = ProxyConfigManager()
    current_config = config_manager.load_config()
    return edit_proxy_config_interactive(current_config)


def edit_proxy_config_interactive(current_config: ProxyConfig) -> Optional[ProxyConfig]:
    """
    Edit a proxy configuration interactively.

    Args:
        current_config: The configuration to edit

    Returns:
        Modified ProxyConfig or None if cancelled
    """
    # Note: ProxyConfigManager used for validation if needed

    console.print("\n[bold cyan]Edit Proxy Configuration[/bold cyan]")

    while True:
        # Show current configuration
        display_proxy_config(current_config)

        # Menu options
        options = [
            "Edit proxy settings",
            "Add model",
            "Remove model",
            "Enable/disable model",
            "Save and exit",
            "Exit without saving",
        ]

        action = unified_prompt(
            "edit_action", "Select action", options, allow_back=False
        )

        if action == "Save and exit":
            return current_config
        elif action == "Exit without saving":
            return None
        elif action == "Edit proxy settings":
            edit_proxy_settings(current_config)
        elif action == "Add model":
            model = configure_model_for_proxy(
                len(current_config.models),
                existing_models=current_config.models,
                is_running_proxy=False,
            )
            if model:
                current_config.models.append(model)
                console.print(f"[green]✓ Added model: {model.name}[/green]")
        elif action == "Remove model":
            if current_config.models:
                model_names = [m.name for m in current_config.models]
                selected = unified_prompt(
                    "remove_model",
                    "Select model to remove",
                    model_names,
                    allow_back=True,
                )
                if selected != "BACK":
                    current_config.models = [
                        m for m in current_config.models if m.name != selected
                    ]
                    console.print(f"[green]✓ Removed model: {selected}[/green]")
        elif action == "Enable/disable model":
            if current_config.models:
                model_names = [
                    f"{m.name} ({'enabled' if m.enabled else 'disabled'})"
                    for m in current_config.models
                ]
                selected = unified_prompt(
                    "toggle_model",
                    "Select model to toggle",
                    model_names,
                    allow_back=True,
                )
                if selected != "BACK":
                    model_name = selected.split(" (")[0]
                    for model in current_config.models:
                        if model.name == model_name:
                            model.enabled = not model.enabled
                            status = "enabled" if model.enabled else "disabled"
                            console.print(
                                f"[green]✓ Model {model_name} {status}[/green]"
                            )
                            break

    return None  # If we exit the loop without saving


def edit_proxy_settings(config: ProxyConfig):
    """Edit proxy server settings."""
    console.print("\n[bold]Edit Proxy Settings[/bold]")
    console.print(f"Current host: {config.host}")
    console.print(f"Current port: {config.port}")

    new_host = input("New host (press Enter to keep current): ").strip()
    if new_host:
        config.host = new_host

    new_port = input("New port (press Enter to keep current): ").strip()
    if new_port:
        try:
            config.port = int(new_port)
        except ValueError:
            console.print("[red]Invalid port number[/red]")

    # Toggle settings
    cors_choice = unified_prompt(
        "cors_setting",
        f"CORS (currently {'enabled' if config.enable_cors else 'disabled'})",
        ["Enable", "Disable", "Keep current"],
        allow_back=False,
    )
    if cors_choice == "Enable":
        config.enable_cors = True
    elif cors_choice == "Disable":
        config.enable_cors = False
    # "Keep current" - no change needed

    console.print("[green]✓ Settings updated[/green]")


def display_proxy_config(config: ProxyConfig):
    """Display proxy configuration."""
    console.print("\n[bold]Current Configuration[/bold]")
    console.print(f"Host: {config.host}")
    console.print(f"Port: {config.port}")
    console.print(f"CORS: {'Enabled' if config.enable_cors else 'Disabled'}")
    console.print(f"Metrics: {'Enabled' if config.enable_metrics else 'Disabled'}")
    console.print(
        f"Request Logging: {'Enabled' if config.log_requests else 'Disabled'}"
    )

    if config.models:
        console.print(f"\nModels ({len(config.models)}):")
        table = Table()
        table.add_column("Name", style="cyan")
        table.add_column("Model Path", style="green")
        table.add_column("GPUs", style="magenta")
        table.add_column("Port", style="yellow")
        table.add_column("Profile", style="blue")
        table.add_column("Status", style="dim")

        for model in config.models:
            gpu_str = (
                ",".join(str(g) for g in model.gpu_ids) if model.gpu_ids else "Auto"
            )
            status = "Enabled" if model.enabled else "Disabled"
            table.add_row(
                model.name,
                (
                    model.model_path[:40] + "..."
                    if len(model.model_path) > 40
                    else model.model_path
                ),
                gpu_str,
                str(model.port),
                model.profile or "None",
                status,
            )

        console.print(table)
    else:
        console.print("\n[yellow]No models configured[/yellow]")
