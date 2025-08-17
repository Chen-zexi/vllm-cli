#!/usr/bin/env python3
"""
System information module for vLLM CLI.

Displays comprehensive system information including GPU, memory, dependencies,
attention backends, quantization support, and optimization recommendations.
"""
import logging

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..system import (
    format_size,
    get_dependency_info,
    get_gpu_capabilities,
    get_gpu_info,
    get_memory_info,
    get_performance_recommendations,
)
from .common import console

logger = logging.getLogger(__name__)


def show_system_info() -> str:
    """
    Display comprehensive system information.
    """
    console.print("\n[bold cyan]System Information[/bold cyan]")

    # Get comprehensive information
    gpu_caps = get_gpu_capabilities()
    dep_info = get_dependency_info()
    recommendations = get_performance_recommendations()

    # Enhanced GPU Information with capabilities
    _show_gpu_information(gpu_caps)

    # Memory Information
    _show_memory_information()

    # Attention Backend Information
    _show_attention_backends(dep_info["attention_backends"])

    # Quantization Support Information
    _show_quantization_support(dep_info["quantization"])

    # Core Dependencies
    _show_core_dependencies(dep_info["core_dependencies"])

    # Performance Recommendations
    if recommendations:
        _show_performance_recommendations(recommendations)

    input("\nPress Enter to continue...")
    return "continue"


def _show_gpu_information(gpu_caps):
    """Display enhanced GPU information with capabilities."""
    gpu_info = get_gpu_info()

    if gpu_info and gpu_caps:
        gpu_table = Table(
            title="[bold green]GPU Information[/bold green]",
            show_header=True,
            header_style="bold blue",
        )
        gpu_table.add_column("GPU", style="cyan")
        gpu_table.add_column("Name", style="magenta", width=45)
        gpu_table.add_column("Memory", style="yellow")
        gpu_table.add_column("Compute Cap", style="green")
        gpu_table.add_column("Architecture", style="blue")
        gpu_table.add_column("Features", style="white")

        for i, (gpu, caps) in enumerate(zip(gpu_info, gpu_caps)):
            # Build features string
            features = []
            if caps.get("fp8_support", False):
                features.append("FP8")
            if caps.get("tensor_cores", False):
                features.append("Tensor")
            if caps.get("bf16_support", False):
                features.append("BF16")
            feature_str = " ".join(features) if features else "Basic"

            gpu_table.add_row(
                str(i),
                gpu["name"][:42] + "..." if len(gpu["name"]) > 42 else gpu["name"],
                f"{format_size(gpu['memory_used'])} / {format_size(gpu['memory_total'])}",
                caps.get("compute_capability", "Unknown"),
                caps.get("architecture", "Unknown"),
                feature_str,
            )

        console.print(gpu_table)
    else:
        console.print("[yellow]No NVIDIA GPUs detected[/yellow]")


def _show_memory_information():
    """Display memory information."""
    memory_info = get_memory_info()
    mem_table = Table(
        title="[bold green]Memory Information[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    mem_table.add_column("Type", style="cyan")
    mem_table.add_column("Total", style="magenta")
    mem_table.add_column("Used", style="yellow")
    mem_table.add_column("Available", style="green")
    mem_table.add_column("Usage", style="red")

    mem_table.add_row(
        "System RAM",
        format_size(memory_info["total"]),
        format_size(memory_info["used"]),
        format_size(memory_info["available"]),
        f"{memory_info['percent']:.1f}%",
    )

    console.print(mem_table)


def _show_attention_backends(attention_info):
    """Display attention backend information."""
    attn_table = Table(
        title="[bold green]Attention Backends[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    attn_table.add_column("Backend", style="cyan")
    attn_table.add_column("Status", style="magenta")
    attn_table.add_column("Version", style="yellow")
    attn_table.add_column("Notes", style="white")

    # Current/effective backend
    current_backend = attention_info.get("current_backend", "auto")
    effective_backend = attention_info.get("effective_backend", "unknown")

    if current_backend != "auto":
        attn_table.add_row(
            "Current",
            f"[green]✓[/green] {current_backend.upper()}",
            "",
            "Explicitly set via VLLM_ATTENTION_BACKEND",
        )
    else:
        attn_table.add_row(
            "Effective",
            f"[blue]→[/blue] {effective_backend}",
            "",
            "Auto-detected backend",
        )

    # Available backends
    backends = ["flash_attn", "xformers", "flashinfer"]
    for backend in backends:
        info = attention_info.get(backend, {})
        name = info.get("name", backend)
        available = info.get("available", False)
        version = info.get("version") or "Not installed"

        if available:
            status = "[green]✓ Available[/green]"
            if backend == "flash_attn":
                # Enhanced Flash Attention notes
                version_info = info.get("version_info", {})
                generation = version_info.get("generation", "unknown")
                if generation == "3":
                    notes = "Flash Attention 3 - Best for Hopper GPUs"
                elif generation == "2":
                    notes = "Flash Attention 2 - Widely supported"
                else:
                    notes = "Ready for use"
            else:
                notes = "Ready for use"
        else:
            status = "[red]✗ Not installed[/red]"
            if backend == "flash_attn":
                notes = "Install Flash Attention 2 for optimal performance"
            else:
                notes = "Install for better performance"

        attn_table.add_row(name, status, version, notes)

    console.print(attn_table)


def _show_quantization_support(quant_info):
    """Display quantization support information."""
    quant_table = Table(
        title="[bold green]Quantization Support[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    quant_table.add_column("Method", style="cyan")
    quant_table.add_column("Status", style="magenta")
    quant_table.add_column("Version", style="yellow")
    quant_table.add_column("Use Case", style="white")

    # Built-in vLLM support
    builtin = quant_info.get("builtin_support", [])
    for method in ["fp8", "awq", "gptq"]:
        if method in builtin:
            quant_table.add_row(
                method.upper(),
                "[green]✓ Built-in[/green]",
                "vLLM native",
                _get_quantization_use_case(method),
            )

    # External libraries (excluding those already shown as built-in)
    external_libs = ["auto_gptq", "bitsandbytes"]
    for lib in external_libs:
        info = quant_info.get(lib, {})
        name = info.get("name", lib)
        available = info.get("available", False)
        version = info.get("version") or "Not installed"

        if available:
            status = "[green]✓ Available[/green]"
            use_case = _get_quantization_use_case(lib)
        else:
            status = "[red]✗ Not installed[/red]"
            use_case = "Install for quantization support"

        quant_table.add_row(name, status, version, use_case)

    console.print(quant_table)


def _show_core_dependencies(core_info):
    """Display core dependencies information."""
    deps_table = Table(
        title="[bold green]Core Dependencies[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    deps_table.add_column("Component", style="cyan", width=24)
    deps_table.add_column("Version", style="magenta", width=20)
    deps_table.add_column("Status", style="yellow")

    # Critical dependencies
    critical_deps = ["vllm", "torch", "transformers", "triton"]
    for dep in critical_deps:
        info = core_info.get(dep, {})
        name = info.get("name", dep)
        available = info.get("available", False)
        version = info.get("version") or "Not installed"

        if available:
            status = "[green]✓ OK[/green]"
        else:
            status = "[red]✗ Missing[/red]"

        deps_table.add_row(name, version, status)

    # Optional but recommended
    optional_deps = ["safetensors", "einops", "accelerate", "peft"]
    for dep in optional_deps:
        info = core_info.get(dep, {})
        if info.get("available", False):
            name = info.get("name", dep)
            version = info.get("version", "unknown")
            deps_table.add_row(
                f"{name} (optional)", version, "[blue]✓ Available[/blue]"
            )

    # CUDA information if available
    cuda_info = core_info.get("cuda_info", {})
    if cuda_info.get("available", False):
        deps_table.add_row(
            "CUDA Runtime",
            cuda_info.get("version", "unknown"),
            "[green]✓ Available[/green]",
        )
        if cuda_info.get("cudnn_available", False):
            deps_table.add_row(
                "cuDNN",
                str(cuda_info.get("cudnn_version", "unknown")),
                "[green]✓ Available[/green]",
            )

    console.print(deps_table)


def _show_performance_recommendations(recommendations):
    """Display performance optimization recommendations."""
    console.print("\n[bold yellow]Performance Recommendations[/bold yellow]")

    high_priority = [r for r in recommendations if r["priority"] == "high"]
    medium_priority = [r for r in recommendations if r["priority"] == "medium"]

    if high_priority:
        high_panel = Panel(
            _format_recommendations(high_priority),
            title="[bold red]High Priority[/bold red]",
            border_style="red",
        )
        console.print(high_panel)

    if medium_priority:
        medium_panel = Panel(
            _format_recommendations(medium_priority),
            title="[bold yellow]Medium Priority[/bold yellow]",
            border_style="yellow",
        )
        console.print(medium_panel)


def _format_recommendations(recommendations):
    """Format recommendations for display."""
    text = Text()

    for i, rec in enumerate(recommendations):
        if i > 0:
            text.append("\n")

        # Add bullet point and title
        text.append("• ", style="bold")
        text.append(rec["title"], style="bold")
        text.append("\n  ")

        # Add description
        text.append(rec["description"], style="dim")
        text.append("\n  ")

        # Add action
        text.append("Action: ", style="cyan")
        text.append(rec["action"])

    return text


def _get_quantization_use_case(method):
    """Get use case description for quantization method."""
    use_cases = {
        "fp8": "GPU accelerated, 2x speedup",
        "awq": "Fast inference, good quality",
        "gptq": "Memory efficient, slower",
        "auto_gptq": "GPTQ implementation",
        "bitsandbytes": "Easy 8-bit/4-bit quantization",
        "llmcompressor": "Advanced compression",
    }
    return use_cases.get(method, "General quantization")
