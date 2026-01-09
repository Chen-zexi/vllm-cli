#!/usr/bin/env python3
"""
Hardware and software capability detection for vLLM CLI.

Detects GPU architectures, compute capabilities, and available optimizations
based on the current hardware and software environment.
"""
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def get_gpu_capabilities() -> List[Dict[str, Any]]:
    """
    Get detailed GPU capability information for each available GPU.

    Returns:
        List of dictionaries with GPU capability information
    """
    capabilities = []

    try:
        import torch

        if not torch.cuda.is_available():
            return capabilities

        for i in range(torch.cuda.device_count()):
            gpu_caps = _analyze_gpu_capabilities(i)
            capabilities.append(gpu_caps)

    except ImportError:
        logger.debug("PyTorch not available for GPU capability detection")

    return capabilities


def get_attention_backend_capabilities() -> Dict[str, Any]:
    """
    Analyze which attention backends can be used with current hardware/software.

    Returns:
        Dictionary with backend compatibility information
    """
    capabilities = {
        "flash_attn": _check_flash_attention_support(),
        "xformers": _check_xformers_support(),
        "flashinfer": _check_flashinfer_support(),
        "pytorch_native": {"supported": True, "reason": "Always available"},
    }

    return capabilities


def get_quantization_capabilities() -> Dict[str, Any]:
    """
    Analyze quantization method compatibility with current setup.

    Returns:
        Dictionary with quantization method compatibility
    """
    capabilities = {}

    # Check hardware-based capabilities
    gpu_caps = get_gpu_capabilities()
    has_fp8_capable_gpu = any(gpu.get("fp8_support", False) for gpu in gpu_caps)

    capabilities["fp8"] = {
        "supported": has_fp8_capable_gpu,
        "reason": (
            "Requires Ada Lovelace, Hopper, or later GPU"
            if not has_fp8_capable_gpu
            else "Supported"
        ),
    }

    capabilities["int8"] = {
        "supported": len(gpu_caps) > 0,
        "reason": "Requires CUDA GPU" if len(gpu_caps) == 0 else "Supported",
    }

    # Check software-based capabilities
    capabilities.update(_check_quantization_software_support())

    return capabilities


def get_performance_recommendations() -> List[Dict[str, Any]]:
    """
    Generate performance optimization recommendations based on current setup.

    Returns:
        List of recommendation dictionaries with priority and description
    """
    recommendations = []

    # Analyze current setup
    gpu_caps = get_gpu_capabilities()
    attention_caps = get_attention_backend_capabilities()
    quant_caps = get_quantization_capabilities()

    # GPU-specific recommendations
    for i, gpu in enumerate(gpu_caps):
        if gpu.get("fp8_support", False) and not quant_caps.get("fp8", {}).get(
            "supported", False
        ):
            recommendations.append(
                {
                    "priority": "high",
                    "category": "quantization",
                    "title": f"Enable FP8 quantization for GPU {i}",
                    "description": f'GPU {i} ({gpu.get("architecture", "unknown")}) supports FP8 for 2x performance boost',
                    "action": 'Install FP8-capable vLLM build and use quantization="fp8"',
                }
            )

    # Attention backend recommendations
    flash_attn_info = attention_caps.get("flash_attn", {})
    if not flash_attn_info.get("supported", False):
        # Check if GPU architecture supports Flash Attention
        has_compatible_gpu = any(
            gpu.get("compute_capability_float", 0) >= 7.5 for gpu in gpu_caps
        )
        if has_compatible_gpu:
            recommendations.append(
                {
                    "priority": "high",
                    "category": "attention",
                    "title": "Install Flash Attention 2",
                    "description": "Flash Attention provides significant memory and speed improvements",
                    "action": "pip install flash-attn (requires sm_75+)",
                }
            )
    else:
        # Check if user has Flash Attention 3 on Hopper/Blackwell GPUs
        version_info = flash_attn_info.get("version_info", {})
        generation = version_info.get("generation", "unknown")

        has_hopper_blackwell = any(
            gpu.get("architecture") in ["Hopper", "Blackwell"] for gpu in gpu_caps
        )

        if generation == "2" and has_hopper_blackwell:
            # Determine the specific architecture and SM version
            arch = gpu_caps[0].get("architecture", "GPU")

            if arch == "Hopper":
                arch_desc = f"{arch} architecture (sm_90)"
            elif arch == "Blackwell":
                # Blackwell can be sm_100 or sm_120
                arch_desc = f"{arch} architecture (sm_100/sm_120)"
            else:
                arch_desc = f"{arch} architecture"

            recommendations.append(
                {
                    "priority": "medium",
                    "category": "attention",
                    "title": "Consider upgrading to Flash Attention 3",
                    "description": f"Your {arch_desc} may support Flash Attention 3",
                    "action": "See https://github.com/Dao-AILab/flash-attention for installation details",
                }
            )

    if any(gpu.get("compute_capability_float", 0) >= 8.9 for gpu in gpu_caps):
        if not attention_caps.get("flashinfer", {}).get("supported", False):
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "attention",
                    "title": "Consider FlashInfer for advanced workloads",
                    "description": "FlashInfer offers superior performance for FP8 and long sequences",
                    "action": "Install FlashInfer for specialized workloads",
                }
            )

    # Memory optimization recommendations
    total_vram = sum(gpu.get("memory_total", 0) for gpu in gpu_caps)
    if total_vram < 40 * 1024**3:  # Less than 40GB total VRAM
        recommendations.append(
            {
                "priority": "medium",
                "category": "memory",
                "title": "Enable quantization for memory efficiency",
                "description": "Your VRAM may be limited for large models",
                "action": "Use AWQ, GPTQ, or FP8 quantization to reduce memory usage",
            }
        )

    return sorted(
        recommendations, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]]
    )


def _analyze_gpu_capabilities(gpu_index: int) -> Dict[str, Any]:
    """
    Analyze capabilities of a specific GPU.

    Args:
        gpu_index: GPU index to analyze

    Returns:
        Dictionary with GPU capability information
    """
    import torch

    props = torch.cuda.get_device_properties(gpu_index)
    capability = torch.cuda.get_device_capability(gpu_index)

    caps = {
        "index": gpu_index,
        "name": props.name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "compute_capability_float": float(f"{capability[0]}.{capability[1]}"),
        "memory_total": props.total_memory,
        "multiprocessor_count": props.multi_processor_count,
    }

    # Determine architecture and features
    caps.update(_determine_gpu_architecture(capability))

    return caps


def _determine_gpu_architecture(capability: Tuple[int, int]) -> Dict[str, Any]:
    """
    Determine GPU architecture and supported features from compute capability.

    Includes support for Jetson platforms (Xavier SM 7.2, Orin SM 8.7, Thor SM 10.x).

    Args:
        capability: CUDA compute capability tuple (major, minor)

    Returns:
        Dictionary with architecture information
    """
    major, minor = capability
    arch_info = {}

    # Architecture mapping (including Jetson devices)
    if major == 12:
        # Blackwell B100/B200 series (sm_120)
        arch_info["architecture"] = "Blackwell"
        arch_info["generation"] = "Latest"
        arch_info["sm_version"] = f"sm_{major}{minor}"
    elif major == 10:
        # Blackwell consumer/lower-tier (sm_100) or Jetson Thor
        arch_info["architecture"] = "Blackwell"
        arch_info["generation"] = "Latest"
        arch_info["sm_version"] = f"sm_{major}{minor}"
    elif major == 9:
        arch_info["architecture"] = "Hopper"
        arch_info["generation"] = "Latest"
        arch_info["sm_version"] = f"sm_{major}{minor}"
    elif major == 8 and minor >= 9:
        arch_info["architecture"] = "Ada Lovelace"
        arch_info["generation"] = "Current"
        arch_info["sm_version"] = f"sm_{major}{minor}"
    elif major == 8 and minor == 7:
        # Jetson Orin (AGX, NX, Nano) uses SM 8.7
        arch_info["architecture"] = "Ampere (Jetson Orin)"
        arch_info["generation"] = "Current"
        arch_info["sm_version"] = f"sm_{major}{minor}"
        arch_info["is_jetson"] = True
    elif major == 8 and minor >= 6:
        arch_info["architecture"] = "Ampere"
        arch_info["generation"] = "Current"
        arch_info["sm_version"] = f"sm_{major}{minor}"
    elif major == 8 and minor < 6:
        arch_info["architecture"] = "Ampere"
        arch_info["generation"] = "Current"
        arch_info["sm_version"] = f"sm_{major}{minor}"
    elif major == 7 and minor >= 5:
        arch_info["architecture"] = "Turing"
        arch_info["generation"] = "Previous"
        arch_info["sm_version"] = f"sm_{major}{minor}"
    elif major == 7 and minor == 2:
        # Jetson Xavier (AGX, NX) uses SM 7.2
        arch_info["architecture"] = "Volta (Jetson Xavier)"
        arch_info["generation"] = "Previous"
        arch_info["sm_version"] = f"sm_{major}{minor}"
        arch_info["is_jetson"] = True
    elif major == 7 and minor < 5:
        arch_info["architecture"] = "Volta"
        arch_info["generation"] = "Previous"
        arch_info["sm_version"] = f"sm_{major}{minor}"
    else:
        arch_info["architecture"] = "Legacy"
        arch_info["generation"] = "Legacy"
        arch_info["sm_version"] = f"sm_{major}{minor}"

    # Feature support based on compute capability
    arch_info["tensor_cores"] = major >= 7
    arch_info["fp16_support"] = major >= 6
    arch_info["bf16_support"] = major >= 8
    arch_info["fp8_support"] = (major >= 9) or (
        major == 8 and minor >= 9
    )  # Ada Lovelace+
    arch_info["int8_support"] = major >= 6
    arch_info["sparsity_support"] = major >= 8  # Ampere+

    return arch_info


def _parse_flash_attention_version(version: str) -> Dict[str, Any]:
    """
    Parse Flash Attention version to determine generation and capabilities.

    Args:
        version: Version string (e.g., '2.5.3', '3.0.0')

    Returns:
        Dictionary with version information
    """
    try:
        if version == "unknown":
            return {"generation": "unknown", "major": 0, "minor": 0}

        # Parse version string
        parts = version.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0

        if major >= 3:
            generation = "3"
            capabilities = {
                "fp8_support": True,
                "async_support": True,
                "low_precision": True,
                "hopper_optimized": True,
            }
        elif major >= 2:
            generation = "2"
            capabilities = {
                "fp8_support": False,
                "async_support": False,
                "low_precision": True,
                "hopper_optimized": False,
            }
        else:
            generation = "1"
            capabilities = {
                "fp8_support": False,
                "async_support": False,
                "low_precision": False,
                "hopper_optimized": False,
            }

        return {
            "generation": generation,
            "major": major,
            "minor": minor,
            "capabilities": capabilities,
        }
    except (ValueError, IndexError):
        return {"generation": "unknown", "major": 0, "minor": 0}


def _check_flash_attention_support() -> Dict[str, Any]:
    """Check Flash Attention support and compatibility."""
    try:
        import flash_attn

        version = getattr(flash_attn, "__version__", "unknown")

        # Determine Flash Attention generation
        version_info = _parse_flash_attention_version(version)

        return {
            "supported": True,
            "version": version,
            "version_info": version_info,
            "reason": f'Flash Attention {version_info["generation"]} installed',
        }
    except ImportError:
        return {
            "supported": False,
            "version": None,
            "version_info": {"generation": "unknown", "major": 0, "minor": 0},
            "reason": "flash-attn package not installed",
        }


def _check_xformers_support() -> Dict[str, Any]:
    """Check xFormers support and compatibility."""
    try:
        import xformers

        return {
            "supported": True,
            "version": getattr(xformers, "__version__", "unknown"),
            "reason": "Installed and available",
        }
    except ImportError:
        return {
            "supported": False,
            "version": None,
            "reason": "xformers package not installed",
        }


def _check_flashinfer_support() -> Dict[str, Any]:
    """Check FlashInfer support and compatibility."""
    try:
        import flashinfer

        return {
            "supported": True,
            "version": getattr(flashinfer, "__version__", "unknown"),
            "reason": "Installed and available",
        }
    except ImportError:
        return {
            "supported": False,
            "version": None,
            "reason": "flashinfer package not installed",
        }


def _check_quantization_software_support() -> Dict[str, Any]:
    """Check quantization software support."""
    support = {}

    # AWQ support
    try:
        import awq

        support["awq"] = {
            "supported": True,
            "reason": f'AWQ {getattr(awq, "__version__", "unknown")} installed',
        }
    except ImportError:
        support["awq"] = {"supported": False, "reason": "AWQ package not installed"}

    # GPTQ support
    try:
        import auto_gptq

        support["gptq"] = {
            "supported": True,
            "reason": f'AutoGPTQ {getattr(auto_gptq, "__version__", "unknown")} installed',
        }
    except ImportError:
        support["gptq"] = {
            "supported": False,
            "reason": "auto-gptq package not installed",
        }

    # BitsAndBytes support
    try:
        import bitsandbytes

        support["bitsandbytes"] = {
            "supported": True,
            "reason": f'BitsAndBytes {getattr(bitsandbytes, "__version__", "unknown")} installed',
        }
    except ImportError:
        support["bitsandbytes"] = {
            "supported": False,
            "reason": "bitsandbytes package not installed",
        }

    return support
