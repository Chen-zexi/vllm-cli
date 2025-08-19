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
from ..system.dependencies import (
    get_vllm_capabilities,
    get_vllm_environment_status,
    get_vllm_kernel_status,
    get_vllm_platform_info,
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

    # Ask if user wants to see detailed vLLM native information
    from .navigation import unified_prompt

    additional_options = [
        "View detailed vLLM native information",
        "Continue without viewing",
    ]

    choice = unified_prompt(
        "vllm_native_info",
        "Additional Information Available",
        additional_options,
        allow_back=False,
    )

    if choice == "View detailed vLLM native information":
        show_vllm_native_info()
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
    attn_table.add_column("Backend", style="cyan", min_width=28)
    attn_table.add_column("Status", style="magenta", min_width=18)
    attn_table.add_column("Version", style="yellow", min_width=14)
    attn_table.add_column("Notes", style="white", overflow="fold")

    # Current/effective backend
    current_backend = attention_info.get("current_backend", "auto")
    effective_backend = attention_info.get("effective_backend", "unknown")

    if current_backend != "auto":
        attn_table.add_row(
            "Current Setting",
            f"[green]✓[/green] {current_backend.upper()}",
            "",
            "Explicitly set via VLLM_ATTENTION_BACKEND",
        )
    else:
        attn_table.add_row(
            "Effective Backend",
            f"[blue]→[/blue] {effective_backend}",
            "",
            "Auto-detected backend",
        )

    # Add separator
    attn_table.add_row("", "", "", "")

    # vLLM Native Flash Attention (if available)
    vllm_fa = attention_info.get("vllm_flash_attn", {})
    if vllm_fa and vllm_fa.get("available"):
        # Main vLLM Flash Attention row
        attn_table.add_row(
            "vLLM Flash Attention (Native)",
            "[green]✓ Built-in[/green]",
            vllm_fa.get("version", "unknown"),
            f"Compute Cap: {vllm_fa.get('compute_capability', 'N/A')}",
        )

        # FA2 status
        fa2_status = (
            "[green]✓ Available[/green]"
            if vllm_fa.get("fa2_available")
            else "[red]✗ Not compiled[/red]"
        )
        fa2_gpu = (
            "[green]✓ Supported[/green]"
            if vllm_fa.get("fa2_gpu_supported")
            else "[yellow]✗ Not supported[/yellow]"
        )
        fa2_notes = (
            "Recommended"
            if vllm_fa.get("fa2_gpu_supported")
            else vllm_fa.get("fa2_unsupported_reason", "GPU incompatible")
        )

        attn_table.add_row(
            "  └─ Flash Attention 2",
            fa2_status if vllm_fa.get("fa2_available") else "[red]✗[/red]",
            fa2_gpu if vllm_fa.get("fa2_available") else "",
            fa2_notes,
        )

        # FA3 status
        fa3_status = (
            "[green]✓ Available[/green]"
            if vllm_fa.get("fa3_available")
            else "[red]✗ Not compiled[/red]"
        )
        fa3_gpu = (
            "[green]✓ Supported[/green]"
            if vllm_fa.get("fa3_gpu_supported")
            else "[yellow]✗ Not supported[/yellow]"
        )
        # Shorten the FA3 unsupported reason for Blackwell
        if not vllm_fa.get("fa3_gpu_supported"):
            if "Blackwell" in vllm_fa.get("fa3_unsupported_reason", ""):
                fa3_notes = "Not for Blackwell GPUs (SM 12.0)"
            else:
                fa3_notes = "Not supported on this GPU"
        else:
            fa3_notes = "Best for Hopper (SM 9.0)"

        attn_table.add_row(
            "  └─ Flash Attention 3",
            fa3_status if vllm_fa.get("fa3_available") else "[red]✗[/red]",
            fa3_gpu if vllm_fa.get("fa3_available") else "",
            fa3_notes,
        )

        # Add separator after vLLM FA
        attn_table.add_row("", "", "", "")

    # External backends
    backends = ["flash_attn", "xformers", "flashinfer"]
    for backend in backends:
        info = attention_info.get(backend, {})
        name = info.get("name", backend)
        available = info.get("available", False)
        version = info.get("version") or "Not installed"

        if available:
            status = "[green]✓ Available[/green]"
            if backend == "flash_attn":
                notes = "External Flash Attention package"
            elif backend == "flashinfer":
                notes = "Optimized for Blackwell/long context"
            else:
                notes = "Memory efficient alternative"
        else:
            status = "[red]✗ Not installed[/red]"
            if backend == "flash_attn":
                notes = "pip install flash-attn"
            elif backend == "flashinfer":
                notes = "pip install flashinfer"
            else:
                notes = "pip install xformers"

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


def show_vllm_native_info():
    """
    Display detailed vLLM native information.
    """
    console.print("\n[bold cyan]vLLM Native Information[/bold cyan]\n")

    # Get all vLLM information
    platform_info = get_vllm_platform_info()
    capabilities = get_vllm_capabilities()
    env_status = get_vllm_environment_status()
    kernel_status = get_vllm_kernel_status()

    # Get Flash Attention info from dependencies
    from ..system.dependencies import get_vllm_flash_attention_info

    flash_attn_info = get_vllm_flash_attention_info()

    # Show platform information
    if platform_info.get("available"):
        _show_vllm_platform(platform_info)

    # Show Flash Attention status
    if flash_attn_info:
        _show_vllm_flash_attention_status(flash_attn_info)

    # Show all attention backends usability
    from ..system.dependencies import get_attention_backend_usability

    backend_info = get_attention_backend_usability()
    if backend_info and not backend_info.get("error"):
        _show_attention_backends_usability(backend_info)

    # Show GPU-specific recommendations
    if platform_info.get("available") and backend_info:
        _show_gpu_recommendations(platform_info, backend_info)

    # Show capabilities
    if capabilities.get("available"):
        _show_vllm_capabilities(capabilities)

    # Show kernel status
    if kernel_status.get("available"):
        _show_vllm_kernels(kernel_status)

    # Show environment variables
    if env_status.get("available"):
        _show_vllm_environment(env_status)


def _show_vllm_flash_attention_status(flash_attn_info):
    """Display vLLM Flash Attention status."""
    fa_table = Table(
        title="[bold green]vLLM Flash Attention Status[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    fa_table.add_column("Component", style="cyan", min_width=30)
    fa_table.add_column("Status", style="magenta", min_width=20)
    fa_table.add_column("Details", style="yellow")

    # Version info
    fa_table.add_row(
        "vLLM Flash Attention",
        f"v{flash_attn_info.get('version', 'Unknown')}",
        f"Compute Cap: {flash_attn_info.get('compute_capability', 'N/A')}",
    )

    # FA2 Status
    fa2_available = flash_attn_info.get("fa2_available", False)
    fa2_gpu_supported = flash_attn_info.get("fa2_gpu_supported", False)

    if fa2_available:
        if fa2_gpu_supported:
            fa2_status = "[green]✓ Available & Supported[/green]"
            fa2_details = "Recommended for this GPU"
        else:
            fa2_status = "[yellow]✓ Available, ✗ Not GPU supported[/yellow]"
            fa2_details = flash_attn_info.get(
                "fa2_unsupported_reason", "GPU incompatible"
            )
    else:
        fa2_status = "[red]✗ Not compiled[/red]"
        fa2_details = "FA2 module not available"

    fa_table.add_row("Flash Attention 2", fa2_status, fa2_details)

    # FA3 Status
    fa3_available = flash_attn_info.get("fa3_available", False)
    fa3_gpu_supported = flash_attn_info.get("fa3_gpu_supported", False)

    if fa3_available:
        if fa3_gpu_supported:
            fa3_status = "[green]✓ Available & Supported[/green]"
            fa3_details = "Best for Hopper GPUs"
        else:
            fa3_status = "[yellow]✓ Available, ✗ Not GPU supported[/yellow]"
            fa3_reason = flash_attn_info.get("fa3_unsupported_reason", "")
            if "Blackwell" in fa3_reason:
                fa3_details = "Not for Blackwell GPUs (SM 12.0)"
            else:
                fa3_details = fa3_reason or "GPU incompatible"
    else:
        fa3_status = "[red]✗ Not compiled[/red]"
        fa3_details = "FA3 module not available"

    fa_table.add_row("Flash Attention 3", fa3_status, fa3_details)

    # Recommended version
    recommended = flash_attn_info.get("recommended_version", "Unknown")
    if recommended != "None (use alternative backend)":
        fa_table.add_row(
            "Recommended Version",
            f"[blue]→ {recommended}[/blue]",
            "Auto-selected for best performance",
        )
    else:
        fa_table.add_row(
            "Recommended Version",
            "[yellow]⚠ Use alternative[/yellow]",
            "Consider FlashInfer or xFormers",
        )

    console.print(fa_table)


def _show_attention_backends_usability(backend_info):
    """Display all attention backends with actual usability status."""
    backend_table = Table(
        title="[bold green]Attention Backends Usability[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    backend_table.add_column("Backend", style="cyan", min_width=30)
    backend_table.add_column("Status", style="magenta", min_width=25)
    backend_table.add_column("Version", style="yellow", min_width=15)
    backend_table.add_column("Notes", style="white")

    # Show current settings first
    env_backend = backend_info.get("env_backend", "auto")
    auto_selected = backend_info.get("auto_selected", "Unknown")

    backend_table.add_row(
        "Environment Setting",
        f"[blue]{env_backend}[/blue]",
        "",
        "VLLM_ATTENTION_BACKEND",
    )

    backend_table.add_row(
        "Auto-Selected Backend",
        f"[green]→ {auto_selected}[/green]",
        "",
        f"For compute cap {backend_info.get('compute_capability', 'unknown')}",
    )

    backend_table.add_row("", "", "", "")  # Separator

    # Sort backends by priority
    backends = backend_info.get("backends", {})
    sorted_backends = sorted(backends.items(), key=lambda x: x[1].get("priority", 99))

    for backend_key, backend_data in sorted_backends:
        name = backend_data.get("name", backend_key)
        available = backend_data.get("available", False)
        usable = backend_data.get("usable", False)
        version = backend_data.get("version", "")
        reason = backend_data.get("reason", "")

        # Special handling for vLLM native FA
        if backend_key == "vllm_flash_attn":
            fa2_usable = backend_data.get("fa2_usable", False)
            fa3_usable = backend_data.get("fa3_usable", False)
            if fa2_usable and fa3_usable:
                status = "[green]✓ FA2+FA3 Usable[/green]"
            elif fa2_usable:
                status = "[green]✓ FA2 Usable[/green]"
            elif fa3_usable:
                status = "[green]✓ FA3 Usable[/green]"
            elif available:
                status = "[yellow]⚠ Available, not usable[/yellow]"
            else:
                status = "[red]✗ Not available[/red]"
        else:
            # Regular backends
            if usable:
                status = "[green]✓ Usable[/green]"
            elif available:
                status = "[yellow]⚠ Available, not usable[/yellow]"
            else:
                status = "[red]✗ Not available[/red]"

        # Format version
        if version and version != "error":
            version_str = (
                f"v{version}" if version and not version.startswith("v") else version
            )
        else:
            version_str = ""

        backend_table.add_row(name, status, version_str, reason)

    console.print(backend_table)


def _show_gpu_recommendations(platform_info, backend_info):
    """Display GPU-specific optimization recommendations."""
    # Get compute capability
    devices = platform_info.get("devices", [])
    if not devices:
        return

    device = devices[0]  # Use first GPU
    compute_cap = device.get("capability", "0.0")
    cap_major = device.get("capability_major", 0)
    cap_minor = device.get("capability_minor", 0)
    gpu_name = device.get("name", "Unknown")

    rec_table = Table(
        title="[bold green]GPU-Specific Recommendations[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    rec_table.add_column("Setting", style="cyan", min_width=40)
    rec_table.add_column("Recommended Value", style="magenta", min_width=30)
    rec_table.add_column("Purpose", style="yellow")

    # Determine GPU architecture and give recommendations
    if cap_major == 12:  # Blackwell
        rec_table.add_row(
            "GPU Architecture",
            f"[bold]Blackwell (SM {compute_cap})[/bold]",
            "Latest generation GPU",
        )
        rec_table.add_row("", "", "")  # Separator
        rec_table.add_row(
            "VLLM_USE_TRTLLM_ATTENTION",
            "[green]1[/green]",
            "Enable TensorRT-LLM for best performance",
        )
        rec_table.add_row(
            "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16",
            "[green]1[/green]",
            "BF16 activation for MoE (reference precision)",
        )
        rec_table.add_row(
            "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8",
            "[yellow]1 (optional)[/yellow]",
            "MXFP8 activation (faster, accuracy trade-off)",
        )
        rec_table.add_row(
            "VLLM_ATTENTION_BACKEND",
            "[blue]FLASHINFER or auto[/blue]",
            "FlashInfer optimized for Blackwell",
        )
    elif cap_major == 9:  # Hopper
        rec_table.add_row(
            "GPU Architecture",
            f"[bold]Hopper (SM {compute_cap})[/bold]",
            "H100 series GPU",
        )
        rec_table.add_row("", "", "")  # Separator
        rec_table.add_row(
            "VLLM_ATTENTION_BACKEND",
            "[green]auto[/green]",
            "Will use Flash Attention 3",
        )
        rec_table.add_row(
            "VLLM_USE_FLASHINFER_MOE_FP8",
            "[green]1[/green]",
            "FP8 support for MoE models",
        )
    elif cap_major == 8 and cap_minor in [0, 6]:  # Ampere
        rec_table.add_row(
            "GPU Architecture",
            f"[bold]Ampere (SM {compute_cap})[/bold]",
            "A100/A40 series GPU",
        )
        rec_table.add_row("", "", "")  # Separator
        rec_table.add_row(
            "VLLM_ATTENTION_BACKEND",
            "[green]TRITON_ATTN_VLLM_V1[/green]",
            "Optimized for Ampere architecture",
        )
        rec_table.add_row(
            "VLLM_USE_TRITON_FLASH_ATTN",
            "[green]1[/green]",
            "Enable Triton flash attention",
        )
    elif cap_major == 8 and cap_minor == 9:  # Ada Lovelace
        rec_table.add_row(
            "GPU Architecture",
            f"[bold]Ada Lovelace (SM {compute_cap})[/bold]",
            "RTX 4090/L40 series GPU",
        )
        rec_table.add_row("", "", "")  # Separator
        rec_table.add_row(
            "VLLM_ATTENTION_BACKEND",
            "[green]auto[/green]",
            "Will use Flash Attention 2",
        )
        rec_table.add_row(
            "VLLM_USE_CUDNN_PREFILL",
            "[yellow]1 (optional)[/yellow]",
            "Can improve prefill performance",
        )
    else:
        rec_table.add_row("GPU Architecture", f"SM {compute_cap}", gpu_name)
        rec_table.add_row("", "", "")  # Separator
        rec_table.add_row(
            "VLLM_ATTENTION_BACKEND",
            "[blue]auto[/blue]",
            "Let vLLM choose best backend",
        )

    # Common recommendations for all GPUs
    rec_table.add_row("", "", "")  # Separator
    rec_table.add_row("[bold]Common Settings[/bold]", "", "")
    rec_table.add_row("VLLM_USE_V1", "[green]1[/green]", "Use V1 engine (recommended)")
    rec_table.add_row(
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
        "[yellow]1 (if needed)[/yellow]",
        "For very long context models",
    )

    console.print(rec_table)


def _show_vllm_platform(platform_info):
    """Display vLLM platform information."""
    platform_table = Table(
        title="[bold green]vLLM Platform & Architecture[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    platform_table.add_column("Property", style="cyan", min_width=25)
    platform_table.add_column("Value", style="magenta")

    # Platform basics
    platform_table.add_row(
        "Platform Type", platform_info.get("platform_type", "Unknown")
    )
    platform_table.add_row("Device Type", platform_info.get("device_type", "Unknown"))
    platform_table.add_row("Ray Device Key", platform_info.get("ray_device_key", "N/A"))

    # Architecture
    if "architecture" in platform_info:
        platform_table.add_row("GPU Architecture", platform_info["architecture"])

    # Device details
    devices = platform_info.get("devices", [])
    if devices:
        for i, device in enumerate(devices):
            if i > 0:
                platform_table.add_row("", "")  # Separator
            platform_table.add_row(f"GPU {i} Name", device["name"])
            platform_table.add_row(f"GPU {i} Compute Capability", device["capability"])
            platform_table.add_row(f"GPU {i} Memory", f"{device['memory_gb']:.1f} GB")
            if device.get("uuid"):
                platform_table.add_row(f"GPU {i} UUID", device["uuid"])

    console.print(platform_table)


def _show_vllm_capabilities(capabilities):
    """Display vLLM capabilities."""
    cap_table = Table(
        title="[bold green]vLLM Capabilities & Features[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    cap_table.add_column("Capability", style="cyan", min_width=30)
    cap_table.add_column("Status", style="magenta", min_width=20)
    cap_table.add_column("Details", style="yellow")

    # FP8 Support
    fp8_support = capabilities.get("supports_fp8", False)
    cap_table.add_row(
        "FP8 Support",
        "[green]✓ Supported[/green]" if fp8_support else "[red]✗ Not supported[/red]",
        capabilities.get("fp8_dtype", "N/A") if fp8_support else "Requires SM 8.9+",
    )

    # Data types
    dtypes = capabilities.get("supported_dtypes", [])
    cap_table.add_row(
        "Supported Data Types",
        "[green]✓ Available[/green]" if dtypes else "[red]✗ Unknown[/red]",
        ", ".join(dtypes) if dtypes else "N/A",
    )

    # V1 Engine
    v1_support = capabilities.get("supports_v1_engine", False)
    cap_table.add_row(
        "V1 Engine Support",
        "[green]✓ Supported[/green]" if v1_support else "[yellow]⚠ Legacy[/yellow]",
        "Optimized engine for latest vLLM" if v1_support else "Using V0 engine",
    )

    # Custom AllReduce
    allreduce = capabilities.get("supports_custom_allreduce", False)
    cap_table.add_row(
        "Custom AllReduce",
        "[green]✓ Enabled[/green]" if allreduce else "[yellow]○ Disabled[/yellow]",
        "Optimized multi-GPU communication" if allreduce else "Standard NCCL",
    )

    # Pin Memory
    pin_memory = capabilities.get("supports_pin_memory", False)
    cap_table.add_row(
        "Pin Memory",
        (
            "[green]✓ Available[/green]"
            if pin_memory
            else "[yellow]○ Not available[/yellow]"
        ),
        "Faster CPU-GPU transfer" if pin_memory else "WSL/Container detected",
    )

    # Sleep Mode
    sleep_mode = capabilities.get("supports_sleep_mode", False)
    cap_table.add_row(
        "Sleep Mode",
        (
            "[green]✓ Available[/green]"
            if sleep_mode
            else "[yellow]○ Not available[/yellow]"
        ),
        "GPU power saving when idle" if sleep_mode else "Always active",
    )

    # Recommended backend
    backend = capabilities.get("recommended_attention_backend", "Unknown")
    cap_table.add_row("Recommended Attention", "[blue]→ Auto-selected[/blue]", backend)

    # CPU Architecture
    cpu_arch = capabilities.get("cpu_architecture", "Unknown")
    cap_table.add_row("CPU Architecture", "[green]✓ Detected[/green]", cpu_arch)

    console.print(cap_table)


def _show_vllm_kernels(kernel_status):
    """Display vLLM kernel availability."""
    kernel_table = Table(
        title="[bold green]vLLM Custom CUDA Kernels[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    kernel_table.add_column("Kernel Category", style="cyan", min_width=20)
    kernel_table.add_column("Kernel Name", style="magenta", min_width=25)
    kernel_table.add_column("Status", style="yellow", min_width=15)

    # Check if custom ops loaded
    if not kernel_status.get("custom_ops_loaded", False):
        kernel_table.add_row("Error", "Custom ops not loaded", "[red]✗ Failed[/red]")
        if "error" in kernel_status:
            kernel_table.add_row("", kernel_status["error"], "")
    else:
        kernels = kernel_status.get("kernels", {})

        for category, kernel_dict in kernels.items():
            first = True
            for kernel_name, available in kernel_dict.items():
                status = "[green]✓[/green]" if available else "[red]✗[/red]"
                if first:
                    kernel_table.add_row(category, kernel_name, status)
                    first = False
                else:
                    kernel_table.add_row("", kernel_name, status)

            # Add separator between categories
            if category != list(kernels.keys())[-1]:
                kernel_table.add_row("", "", "")

    # Show vLLM Flash Attention version
    fa_version = kernel_status.get("vllm_flash_attn_version")
    if fa_version:
        kernel_table.add_row(
            "Flash Attention", "vLLM Native FA", f"[green]v{fa_version}[/green]"
        )

    # Show Python-level MoE implementation
    if kernel_status.get("python_fused_moe"):
        kernel_table.add_row("MoE (Python)", "fused_moe (Triton)", "[green]✓[/green]")

    # Show Flash Attention interface functions
    flash_interface = kernel_status.get("flash_attn_interface")
    if flash_interface:
        kernel_table.add_row("", "", "")  # Separator
        first = True
        for func_name, available in flash_interface.items():
            status = "[green]✓[/green]" if available else "[red]✗[/red]"
            if first:
                kernel_table.add_row("Flash Attn Interface", func_name, status)
                first = False
            else:
                kernel_table.add_row("", func_name, status)

    console.print(kernel_table)


def _show_vllm_environment(env_status):
    """Display vLLM environment variables."""
    env_table = Table(
        title="[bold green]vLLM Environment Variables[/bold green]",
        show_header=True,
        header_style="bold blue",
    )
    env_table.add_column("Category", style="cyan", min_width=15)
    env_table.add_column("Variable", style="magenta", min_width=35)
    env_table.add_column("Current", style="yellow", min_width=15)
    env_table.add_column("Default", style="white", min_width=15)

    categories = env_status.get("categories", {})

    for category_name, variables in categories.items():
        first = True
        has_content = False

        for var_name, var_info in variables.items():
            env_value = var_info.get("env_value")
            default_value = var_info.get("default_value")
            is_modified = var_info.get("is_modified", False)

            # Only show if set or has interesting default
            if env_value is not None or default_value not in [
                None,
                "not set",
                False,
                "",
            ]:
                has_content = True

                # Format values
                current = env_value if env_value is not None else "[dim]not set[/dim]"
                default = (
                    str(default_value)
                    if default_value is not None
                    else "[dim]none[/dim]"
                )

                # Highlight modified values
                if is_modified and env_value is not None:
                    current = f"[bold green]{current}[/bold green]"

                if first:
                    env_table.add_row(category_name, var_name, current, default)
                    first = False
                else:
                    env_table.add_row("", var_name, current, default)

        # Add separator between categories if we had content
        if has_content and category_name != list(categories.keys())[-1]:
            env_table.add_row("", "", "", "")

    console.print(env_table)

    # Show a legend
    console.print(
        "\n[dim]Legend: [bold green]Green[/bold green] = Modified from default | [dim]dim[/dim] = Not set[/dim]"
    )
