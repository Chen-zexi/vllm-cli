#!/usr/bin/env python3
"""
Progress bar styles for vLLM CLI.

Provides different visual styles for progress bars used throughout the application.
"""
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class ProgressStyle:
    """Definition of a progress bar style."""

    name: str
    description: str
    render_func: Callable[[float, int, bool], str]
    example: str


def get_color_for_percentage(percentage: float) -> str:
    """Get color based on percentage value."""
    if percentage < 50:
        return "green"
    elif percentage < 80:
        return "yellow"
    else:
        return "red"


def render_blocks(
    percentage: float, width: int = 12, show_percentage: bool = True
) -> str:
    """Render block-style progress bar."""
    filled = int(percentage / 100 * width)
    color = get_color_for_percentage(percentage)
    bar = f"[{color}]{'█' * filled}{'░' * (width - filled)}[/{color}]"
    if show_percentage:
        return f"{bar} {percentage:.1f}%"
    return bar


def render_brackets(
    percentage: float, width: int = 12, show_percentage: bool = True
) -> str:
    """Render bracket-style progress bar."""
    filled = int(percentage / 100 * width)
    color = get_color_for_percentage(percentage)
    bar = f"[{color}][{'=' * filled}{' ' * (width - filled)}][/{color}]"
    if show_percentage:
        return f"{bar} {percentage:.1f}%"
    return bar


def render_pipes(
    percentage: float, width: int = 12, show_percentage: bool = True
) -> str:
    """Render pipe-style progress bar."""
    filled = int(percentage / 100 * width)
    color = get_color_for_percentage(percentage)
    bar = f"|[{color}]{'#' * filled}[/{color}]{'-' * (width - filled)}|"
    if show_percentage:
        return f"{bar} {percentage:.1f}%"
    return bar


def render_smooth(
    percentage: float, width: int = 12, show_percentage: bool = True
) -> str:
    """Render smooth-style progress bar with special characters."""
    filled = int(percentage / 100 * width)
    color = get_color_for_percentage(percentage)
    bar = f"[{color}]{'▰' * filled}{'▱' * (width - filled)}[/{color}]"
    if show_percentage:
        return f"{bar} {percentage:.1f}%"
    return bar


def render_dots(
    percentage: float, width: int = 10, show_percentage: bool = True
) -> str:
    """Render dot-style progress bar."""
    filled = int(percentage / 100 * width)
    color = get_color_for_percentage(percentage)
    bar = f"[{color}]{'●' * filled}[/{color}]{'○' * (width - filled)}"
    if show_percentage:
        return f"{bar} {percentage:.1f}%"
    return bar


def render_ascii(
    percentage: float, width: int = 15, show_percentage: bool = True
) -> str:
    """Render simple ASCII progress bar."""
    filled = int(percentage / 100 * width)
    color = get_color_for_percentage(percentage)
    bar = f"[{color}][{filled * '=':>{width}}][/{color}]"
    if show_percentage:
        return f"{bar} {percentage:.1f}%"
    return bar


def render_minimal(
    percentage: float, width: int = 0, show_percentage: bool = True
) -> str:
    """Render minimal style - just percentage with color."""
    color = get_color_for_percentage(percentage)
    return f"[{color}]{percentage:5.1f}%[/{color}]"


# Registry of available styles
PROGRESS_STYLES: Dict[str, ProgressStyle] = {
    "blocks": ProgressStyle(
        name="blocks",
        description="Block characters with shading",
        render_func=render_blocks,
        example="████░░░░ 50.0%",
    ),
    "brackets": ProgressStyle(
        name="brackets",
        description="Brackets with equals signs",
        render_func=render_brackets,
        example="[====    ] 50.0%",
    ),
    "pipes": ProgressStyle(
        name="pipes",
        description="Pipes with hashes and dashes",
        render_func=render_pipes,
        example="|####----| 50.0%",
    ),
    "smooth": ProgressStyle(
        name="smooth",
        description="Smooth blocks with special characters",
        render_func=render_smooth,
        example="▰▰▰▰▱▱▱▱ 50.0%",
    ),
    "dots": ProgressStyle(
        name="dots",
        description="Filled and empty circles",
        render_func=render_dots,
        example="●●●●○○○○ 50.0%",
    ),
    "ascii": ProgressStyle(
        name="ascii",
        description="Simple ASCII with brackets",
        render_func=render_ascii,
        example="[=======   ] 50.0%",
    ),
    "minimal": ProgressStyle(
        name="minimal",
        description="Just percentage with color",
        render_func=render_minimal,
        example=" 50.0%",
    ),
}


def get_progress_bar(
    percentage: float,
    style: str = "blocks",
    width: int = 12,
    show_percentage: bool = True,
) -> str:
    """
    Get a progress bar with the specified style.

    Args:
        percentage: Progress percentage (0-100)
        style: Style name from PROGRESS_STYLES
        width: Width of the progress bar
        show_percentage: Whether to show percentage after the bar

    Returns:
        Formatted progress bar string
    """
    if style not in PROGRESS_STYLES:
        style = "blocks"  # Default fallback

    return PROGRESS_STYLES[style].render_func(percentage, width, show_percentage)


def get_style_preview(style: str, sample_percentages: list = None) -> str:
    """
    Get a preview of a progress bar style with different percentages.

    Args:
        style: Style name from PROGRESS_STYLES
        sample_percentages: List of percentages to show (default: [25, 50, 75, 100])

    Returns:
        Formatted preview string
    """
    if sample_percentages is None:
        sample_percentages = [25, 50, 75, 100]

    if style not in PROGRESS_STYLES:
        return f"Unknown style: {style}"

    style_obj = PROGRESS_STYLES[style]
    previews = []
    for pct in sample_percentages:
        previews.append(f"  {pct:3d}%: {style_obj.render_func(pct, 12)}")

    return "\n".join(previews)


def list_available_styles() -> list:
    """Get list of available progress bar style names."""
    return list(PROGRESS_STYLES.keys())
