#!/usr/bin/env python3
"""
GPU utility functions for UI components.

This module centralizes all GPU-related UI functionality to eliminate
code duplication across various UI components. It provides consistent
GPU status panels, summaries, and display utilities.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..config import ConfigManager
from ..system import format_size, get_gpu_info
from .common import console, create_panel
from .progress_styles import get_progress_bar

logger = logging.getLogger(__name__)


def calculate_gpu_panel_size(gpu_count: int) -> int:
    """
    Calculate optimal panel size based on GPU count.

    Determines the appropriate height for GPU panels based on the number
    of detected GPUs, accounting for headers, borders, and content.

    Args:
        gpu_count: Number of GPUs detected

    Returns:
        Optimal panel height in lines
    """
    if gpu_count == 0:
        return 5  # For "No GPU detected" message with borders
    # Panel needs: 1 border top + 1 header row + N gpu rows + 1 border bottom
    # So total = 3 + gpu_count
    return min(3 + gpu_count, 15)  # Cap at 15 for many GPUs


def get_gpu_panel_config() -> Dict[str, Any]:
    """
    Get GPU panel configuration from user preferences.

    Returns:
        Dictionary containing panel configuration settings
    """
    config_manager = ConfigManager()
    ui_prefs = config_manager.get_ui_preferences()

    return {
        "progress_style": ui_prefs.get("progress_bar_style", "blocks"),
        "compact_mode": ui_prefs.get("compact_gpu_display", False),
        "show_temperature": ui_prefs.get("show_gpu_temperature", True),
        "max_name_length": ui_prefs.get("max_gpu_name_length", 30),
    }


def format_gpu_name(gpu_name: str, max_length: Optional[int] = None) -> str:
    """
    Format GPU name for display by removing verbose suffixes.

    Args:
        gpu_name: Original GPU name
        max_length: Maximum length for the name (None for no limit)

    Returns:
        Formatted GPU name
    """
    # Remove common verbose suffixes
    name = gpu_name.replace(" Workstation Edition", "")
    name = name.replace(" Graphics Card", "")
    name = name.replace(" Gaming", "")

    if max_length and len(name) > max_length:
        name = name[: max_length - 3] + "..."

    return name


def format_gpu_memory_stats(
    gpu: Dict[str, Any], compact: bool = False
) -> Tuple[str, float]:
    """
    Format GPU memory statistics for display.

    Args:
        gpu: GPU information dictionary
        compact: Whether to use compact formatting

    Returns:
        Tuple of (formatted text, memory percentage)
    """
    mem_used = gpu["memory_used"]
    mem_total = gpu["memory_total"]
    mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0

    if compact:
        return f"{mem_percent:.0f}%", mem_percent
    else:
        return f"{format_size(mem_used):>10}/{format_size(mem_total)}", mem_percent


def format_gpu_temperature(temperature: float) -> str:
    """
    Format GPU temperature with appropriate color coding.

    Args:
        temperature: Temperature in Celsius

    Returns:
        Formatted temperature string with color markup
    """
    if temperature <= 0:
        return "[dim]N/A[/dim]"

    if temperature < 60:
        color = "green"
    elif temperature < 80:
        color = "yellow"
    else:
        color = "red"

    return f"[{color}]{temperature:3.0f}°C[/{color}]"


def create_gpu_table(gpu_info: List[Dict[str, Any]], config: Dict[str, Any]) -> Table:
    """
    Create a table showing GPU information.

    Args:
        gpu_info: List of GPU information dictionaries
        config: Panel configuration settings

    Returns:
        Rich Table with GPU information
    """
    gpu_table = Table(
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
        expand=True,
        show_edge=False,
    )

    # Define columns with responsive width distribution
    gpu_table.add_column("GPU", style="cyan", no_wrap=True, width=4)
    gpu_table.add_column("Name", style="white", no_wrap=False, ratio=2)
    gpu_table.add_column("Stats", style="white", no_wrap=True, ratio=5)

    # Determine display settings based on terminal width
    compact_mode = config["compact_mode"] or console.width < 100
    bar_width = 6 if compact_mode else 10
    padding_spaces = 3 if compact_mode else 6  # More padding for wider terminals

    for gpu in gpu_info:
        gpu_name = format_gpu_name(gpu["name"], config["max_name_length"])

        # Format memory statistics
        mem_text, mem_percent = format_gpu_memory_stats(gpu, compact_mode)
        mem_bar = get_progress_bar(
            mem_percent,
            style=config["progress_style"],
            width=bar_width,
            show_percentage=False,
        )

        # Format utilization
        util_percent = gpu["utilization"]
        util_text = f"{util_percent:3.0f}%" if compact_mode else f"{util_percent:5.1f}%"
        util_bar = get_progress_bar(
            util_percent,
            style=config["progress_style"],
            width=bar_width,
            show_percentage=False,
        )

        # Build stats display
        stats_parts = [f"Mem: {mem_bar} {mem_text}", f"Util: {util_bar} {util_text}"]

        # Add temperature if enabled and available
        if config["show_temperature"]:
            temp_text = format_gpu_temperature(gpu["temperature"])
            stats_parts.append(f"Temp: {temp_text}")

        stats_display = (" " * padding_spaces).join(stats_parts)

        gpu_table.add_row(f"[bold]{gpu['index']:^3}[/bold]", gpu_name, stats_display)

    return gpu_table


def create_no_gpu_warning() -> Text:
    """
    Create a warning message for when no GPUs are detected.

    Returns:
        Rich Text object with warning message
    """
    warning_text = Text()
    warning_text.append("No NVIDIA GPUs detected\n\n", style="bold yellow")
    warning_text.append("vLLM requires NVIDIA GPUs with CUDA support.\n", style="white")
    warning_text.append("Please ensure:\n", style="dim white")
    warning_text.append("  • NVIDIA GPU is installed\n", style="dim white")
    warning_text.append("  • NVIDIA drivers are installed\n", style="dim white")
    warning_text.append("  • CUDA toolkit is available\n", style="dim white")
    warning_text.append("\n", style="white")
    warning_text.append("For Jetson devices (Orin, Thor, Xavier):\n", style="dim cyan")
    warning_text.append("  • Ensure JetPack SDK is properly installed\n", style="dim white")
    warning_text.append("  • Verify PyTorch with CUDA support is available\n", style="dim white")

    return warning_text


def create_gpu_status_panel(progress_style: Optional[str] = None) -> Panel:
    """
    Create a GPU status panel showing all GPUs with their stats.

    This is the main function for creating comprehensive GPU status displays.
    It automatically adapts to terminal width and user preferences.

    Args:
        progress_style: Style for progress bars (blocks, brackets, pipes, etc.)
                       If None, will use user's configured preference

    Returns:
        Rich Panel with GPU information
    """
    try:
        gpu_info = get_gpu_info()

        # Get configuration with override for progress style
        config = get_gpu_panel_config()
        if progress_style is not None:
            config["progress_style"] = progress_style

        if gpu_info:
            gpu_table = create_gpu_table(gpu_info, config)
            return create_panel(
                gpu_table,
                title="[bold cyan]GPU Status[/bold cyan]",
                border_style="cyan",
            )
        else:
            warning_text = create_no_gpu_warning()
            return create_panel(
                warning_text,
                title="[bold yellow]GPU Status[/bold yellow]",
                border_style="yellow",
            )

    except Exception as e:
        logger.warning(f"Failed to create GPU status panel: {e}")
        error_text = Text(f"Error getting GPU information: {e}", style="red")
        return create_panel(
            error_text,
            title="[bold red]GPU Status Error[/bold red]",
            border_style="red",
        )


def create_gpu_summary_line(
    gpu_info: List[Dict[str, Any]], progress_style: Optional[str] = None
) -> str:
    """
    Create a single-line GPU summary for compact displays.

    Args:
        gpu_info: List of GPU information dictionaries
        progress_style: Style for progress bars (if None, uses config)

    Returns:
        Single-line summary string with markup
    """
    if not gpu_info:
        return "[yellow]No GPUs detected[/yellow]"

    config = get_gpu_panel_config()
    if progress_style is not None:
        config["progress_style"] = progress_style

    summaries = []

    # Show up to 2 GPUs in summary to avoid overflow
    for gpu in gpu_info[:2]:
        mem_text, mem_percent = format_gpu_memory_stats(gpu, compact=True)
        mem_bar = get_progress_bar(
            mem_percent, style=config["progress_style"], width=4, show_percentage=False
        )
        summaries.append(f"GPU{gpu['index']}: {mem_bar} {mem_text}")

    # Add indicator if there are more GPUs
    if len(gpu_info) > 2:
        summaries.append(f"[dim](+{len(gpu_info)-2} more)[/dim]")

    return " | ".join(summaries)


def get_gpu_memory_dict() -> Dict[int, Dict[str, float]]:
    """
    Get current GPU memory usage as a simple dictionary.

    This provides a simple data structure for GPU memory information,
    useful for programmatic access rather than display formatting.

    Returns:
        Dictionary mapping GPU ID to memory info with:
        - total_gb: Total memory in GB
        - used_gb: Used memory in GB
        - free_gb: Free memory in GB
        - usage_percent: Percentage of memory used
    """
    gpu_memory = {}
    try:
        gpu_infos = get_gpu_info()

        for gpu in gpu_infos:
            gpu_id = gpu["index"]
            total_bytes = gpu["memory_total"]
            used_bytes = gpu["memory_used"]
            free_bytes = gpu["memory_free"]

            gpu_memory[gpu_id] = {
                "total_gb": total_bytes / (1024**3),
                "used_gb": used_bytes / (1024**3),
                "free_gb": free_bytes / (1024**3),
                "usage_percent": (
                    (used_bytes / total_bytes * 100) if total_bytes > 0 else 0
                ),
            }
    except Exception as e:
        logger.warning(f"Failed to get GPU memory dict: {e}")

    return gpu_memory


def check_gpu_memory_warnings(
    gpu_ids: List[int], threshold_percent: float = 10.0
) -> List[str]:
    """
    Check if GPUs have significant memory usage and return warnings.

    This function helps prevent out-of-memory errors by warning when
    GPUs already have significant memory allocated.

    Args:
        gpu_ids: List of GPU IDs to check
        threshold_percent: Memory usage percentage threshold for warnings (default: 10%)

    Returns:
        List of warning messages (empty if no concerns)
    """
    warnings = []
    gpu_memory = get_gpu_memory_dict()

    for gpu_id in gpu_ids:
        if gpu_id in gpu_memory:
            mem_info = gpu_memory[gpu_id]
            # Warn if GPU exceeds threshold
            if mem_info["usage_percent"] > threshold_percent:
                warnings.append(
                    f"⚠️  GPU {gpu_id} already has {mem_info['usage_percent']:.0f}% "
                    f"memory in use ({mem_info['used_gb']:.1f}GB). "
                    f"Loading additional models may cause OOM errors."
                )

    return warnings


def create_compact_gpu_panel(progress_style: Optional[str] = None) -> Panel:
    """
    Create a compact GPU panel for space-constrained displays.

    Args:
        progress_style: Style for progress bars (if None, uses config)

    Returns:
        Rich Panel with compact GPU information
    """
    try:
        gpu_info = get_gpu_info()

        if not gpu_info:
            return create_panel(
                Text("No GPUs detected", style="yellow"),
                title="GPU",
                border_style="yellow",
            )

        config = get_gpu_panel_config()
        config["compact_mode"] = True  # Force compact mode
        if progress_style is not None:
            config["progress_style"] = progress_style

        content_lines = []
        for gpu in gpu_info:
            mem_text, mem_percent = format_gpu_memory_stats(gpu, compact=True)
            mem_bar = get_progress_bar(
                mem_percent,
                style=config["progress_style"],
                width=6,
                show_percentage=False,
            )

            gpu_name = format_gpu_name(gpu["name"], 20)  # Shorter for compact
            line = f"GPU{gpu['index']}: {gpu_name} | {mem_bar} {mem_text}"
            content_lines.append(line)

        content = Text("\n".join(content_lines))

        return create_panel(
            content, title="[bold cyan]GPUs[/bold cyan]", border_style="cyan"
        )

    except Exception as e:
        logger.warning(f"Failed to create compact GPU panel: {e}")
        return create_panel(
            Text(f"GPU Error: {e}", style="red"), title="GPU", border_style="red"
        )
