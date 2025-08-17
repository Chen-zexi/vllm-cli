#!/usr/bin/env python3
"""
Shared UI components for vLLM CLI.

Provides reusable UI components like system overview panels and other
non-GPU specific UI elements. GPU-related components have been moved
to gpu_utils.py to eliminate code duplication.
"""
import logging
from typing import Any, Dict, Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..system import format_size, get_cuda_version, get_gpu_info, get_memory_info
from .common import create_panel

# Import the GPU utility we actually use
from .gpu_utils import create_gpu_summary_line

logger = logging.getLogger(__name__)


def create_system_overview_panel() -> Optional[Panel]:
    """
    Create a system overview panel showing memory, CUDA version, etc.

    Provides a comprehensive system information panel including memory usage,
    GPU counts, VRAM totals, and CUDA version information. This complements
    the detailed GPU panels with high-level system metrics.

    Returns:
        Rich Panel with system information or None if error
    """
    try:
        memory_info = get_memory_info()
        gpu_info = get_gpu_info()
        cuda_version = get_cuda_version()

        system_table = Table(show_header=False, box=None, padding=(0, 1))
        system_table.add_column("Property", style="green")
        system_table.add_column("Value")

        # System memory
        system_table.add_row(
            "System Memory",
            f"{format_size(memory_info['used'])}/{format_size(memory_info['total'])} "
            f"({memory_info['percent']:.1f}%)",
        )

        # GPU info if available
        if gpu_info:
            total_vram = sum(gpu["memory_total"] for gpu in gpu_info)
            total_used = sum(gpu["memory_used"] for gpu in gpu_info)

            system_table.add_row("Total GPUs", f"{len(gpu_info)}")
            system_table.add_row(
                "Total VRAM", f"{format_size(total_used)}/{format_size(total_vram)}"
            )

        # CUDA version if available
        if cuda_version:
            system_table.add_row("CUDA Version", cuda_version)

        return create_panel(
            system_table,
            title="[bold green]System Overview[/bold green]",
            border_style="green",
        )

    except Exception as e:
        logger.warning(f"Failed to create system overview panel: {e}")
        return None


def create_status_summary_panel(
    additional_info: Optional[Dict[str, Any]] = None,
) -> Panel:
    """
    Create a status summary panel for dashboard views.

    Combines system overview with additional status information in a
    compact format suitable for dashboard displays.

    Args:
        additional_info: Optional dictionary with additional status information

    Returns:
        Rich Panel with combined status information
    """
    try:
        content_lines = []

        # System memory
        memory_info = get_memory_info()
        mem_usage = f"{memory_info['percent']:.1f}%"
        content_lines.append(f"[green]System Memory:[/green] {mem_usage}")

        # GPU summary
        gpu_info = get_gpu_info()
        if gpu_info:
            gpu_summary = create_gpu_summary_line(gpu_info)
            content_lines.append(f"[cyan]GPUs:[/cyan] {gpu_summary}")
        else:
            content_lines.append("[yellow]GPUs:[/yellow] None detected")

        # Additional info if provided
        if additional_info:
            for key, value in additional_info.items():
                content_lines.append(f"[white]{key}:[/white] {value}")

        content = Text("\n".join(content_lines))

        return create_panel(
            content,
            title="[bold white]Status Summary[/bold white]",
            border_style="white",
        )

    except Exception as e:
        logger.warning(f"Failed to create status summary panel: {e}")
        return create_panel(
            Text(f"Status error: {e}", style="red"), title="Status", border_style="red"
        )


def create_info_panel(title: str, content: str, style: str = "white") -> Panel:
    """
    Create a simple information panel with styled content.

    Args:
        title: Panel title
        content: Panel content text
        style: Border and title style

    Returns:
        Rich Panel with the specified content
    """
    return create_panel(
        Text(content), title=f"[bold {style}]{title}[/bold {style}]", border_style=style
    )


def create_error_panel(error_message: str, title: str = "Error") -> Panel:
    """
    Create a standardized error panel.

    Args:
        error_message: Error message to display
        title: Panel title (default: "Error")

    Returns:
        Rich Panel with error information
    """
    return create_panel(
        Text(error_message, style="red"),
        title=f"[bold red]{title}[/bold red]",
        border_style="red",
    )


def create_warning_panel(warning_message: str, title: str = "Warning") -> Panel:
    """
    Create a standardized warning panel.

    Args:
        warning_message: Warning message to display
        title: Panel title (default: "Warning")

    Returns:
        Rich Panel with warning information
    """
    return create_panel(
        Text(warning_message, style="yellow"),
        title=f"[bold yellow]{title}[/bold yellow]",
        border_style="yellow",
    )


def create_success_panel(success_message: str, title: str = "Success") -> Panel:
    """
    Create a standardized success panel.

    Args:
        success_message: Success message to display
        title: Panel title (default: "Success")

    Returns:
        Rich Panel with success information
    """
    return create_panel(
        Text(success_message, style="green"),
        title=f"[bold green]{title}[/bold green]",
        border_style="green",
    )
