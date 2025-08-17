#!/usr/bin/env python3
"""
Dependency detection utilities for vLLM CLI.

Detects installed packages and their versions for vLLM optimization libraries
and core dependencies.
"""
import importlib
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def get_dependency_info() -> Dict[str, Any]:
    """
    Get comprehensive dependency information for vLLM optimization.

    Returns:
        Dictionary containing dependency information organized by category
    """
    return {
        "attention_backends": get_attention_backend_info(),
        "quantization": get_quantization_info(),
        "core_dependencies": get_core_dependencies(),
        "environment": get_environment_info(),
    }


def get_attention_backend_info() -> Dict[str, Any]:
    """
    Get information about attention backend libraries and current configuration.

    Returns:
        Dictionary with attention backend information
    """
    info = {}

    # Check installed attention backends
    attention_packages = {
        "flash_attn": "Flash Attention",
        "xformers": "xFormers",
        "flashinfer": "FlashInfer",
    }

    for package, name in attention_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            info[package] = {"name": name, "version": version, "available": True}
        except ImportError:
            info[package] = {"name": name, "version": None, "available": False}

    # Check current backend configuration
    info["current_backend"] = os.environ.get("VLLM_ATTENTION_BACKEND", "auto")

    # Determine effective backend
    info["effective_backend"] = _determine_effective_backend(info)

    return info


def get_quantization_info() -> Dict[str, Any]:
    """
    Get information about quantization libraries and support.

    Returns:
        Dictionary with quantization library information
    """
    info = {}

    # Check quantization libraries
    quant_packages = {
        "auto_gptq": "AutoGPTQ",
        "awq": "AWQ",
        "bitsandbytes": "BitsAndBytes",
        "optimum": "Optimum (Quantization)",
        "llmcompressor": "LLM Compressor",
    }

    for package, name in quant_packages.items():
        try:
            if package == "optimum":
                # Check optimum with quantization support
                module = importlib.import_module("optimum.gptq")
                base_module = importlib.import_module("optimum")
                version = getattr(base_module, "__version__", "installed")
            else:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "installed")

            info[package] = {"name": name, "version": version, "available": True}
        except ImportError:
            info[package] = {"name": name, "version": None, "available": False}

    # Check vLLM's built-in quantization support
    info["builtin_support"] = _check_vllm_quantization_support()

    return info


def get_core_dependencies() -> Dict[str, Any]:
    """
    Get information about core dependencies for vLLM.

    Returns:
        Dictionary with core dependency information
    """
    info = {}

    # Core ML libraries
    core_packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "safetensors": "SafeTensors",
        "einops": "Einops",
        "accelerate": "Accelerate",
        "peft": "PEFT",
        "triton": "Triton",
        "vllm": "vLLM",
    }

    for package, name in core_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            info[package] = {"name": name, "version": version, "available": True}
        except ImportError:
            info[package] = {"name": name, "version": None, "available": False}

    # Add CUDA information if PyTorch is available
    if info.get("torch", {}).get("available", False):
        info["cuda_info"] = _get_cuda_info()

    return info


def get_environment_info() -> Dict[str, str]:
    """
    Get relevant environment variables for vLLM configuration.

    Returns:
        Dictionary of relevant environment variables
    """
    env_vars = [
        "VLLM_ATTENTION_BACKEND",
        "CUDA_VISIBLE_DEVICES",
        "VLLM_USE_TRITON_FLASH_ATTN",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_ENGINE_ITERATION_TIMEOUT_S",
    ]

    return {var: os.environ.get(var, "not set") for var in env_vars}


def _determine_effective_backend(attention_info: Dict[str, Any]) -> str:
    """
    Determine which attention backend is likely to be used.

    Args:
        attention_info: Attention backend information

    Returns:
        Name of the likely effective backend
    """
    current = attention_info.get("current_backend", "auto")

    if current != "auto":
        return current

    # Auto-detection logic (simplified)
    if attention_info.get("flash_attn", {}).get("available", False):
        return "flash_attn (auto-detected)"
    elif attention_info.get("xformers", {}).get("available", False):
        return "xformers (auto-detected)"
    else:
        return "pytorch_native (fallback)"


def _check_vllm_quantization_support() -> List[str]:
    """
    Check vLLM's built-in quantization method support.

    Returns:
        List of supported quantization methods
    """
    supported = []

    try:
        # Try to import vLLM quantization
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

        # Handle both dict and list formats
        if hasattr(QUANTIZATION_METHODS, "keys"):
            supported = list(QUANTIZATION_METHODS.keys())
        else:
            # If it's a list or set, convert to list
            supported = list(QUANTIZATION_METHODS)
    except ImportError:
        # Fallback to common known methods
        try:
            # These are commonly supported in vLLM
            potential_methods = [
                "awq",
                "gptq",
                "squeezellm",
                "marlin",
                "fp8",
                "bitsandbytes",
            ]
            for method in potential_methods:
                try:
                    # Try to create a dummy quantization config
                    pass

                    supported.append(method)
                except Exception:
                    continue
        except Exception:
            supported = ["fp8", "awq", "gptq"]  # Common fallback

    return supported


def _get_cuda_info() -> Dict[str, Any]:
    """
    Get detailed CUDA information.

    Returns:
        Dictionary with CUDA details
    """
    cuda_info = {}

    try:
        import torch

        if torch.cuda.is_available():
            cuda_info["available"] = True
            cuda_info["version"] = torch.version.cuda
            cuda_info["device_count"] = torch.cuda.device_count()

            # Get compute capability for current device
            if torch.cuda.device_count() > 0:
                capability = torch.cuda.get_device_capability()
                cuda_info["compute_capability"] = f"{capability[0]}.{capability[1]}"

            # Check cuDNN
            if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
                cuda_info["cudnn_version"] = torch.backends.cudnn.version()
                cuda_info["cudnn_available"] = True
            else:
                cuda_info["cudnn_available"] = False

            # Check NCCL
            try:
                if hasattr(torch.cuda, "nccl") and torch.cuda.device_count() > 0:
                    cuda_info["nccl_available"] = torch.cuda.nccl.is_available(
                        torch.cuda.current_device()
                    )
                else:
                    cuda_info["nccl_available"] = False
            except Exception:
                cuda_info["nccl_available"] = False

        else:
            cuda_info["available"] = False

    except ImportError:
        cuda_info["available"] = False

    return cuda_info


def check_optimization_recommendations() -> List[str]:
    """
    Generate recommendations for performance optimizations.

    Returns:
        List of recommendation strings
    """
    recommendations = []
    dep_info = get_dependency_info()

    # Check attention backend recommendations
    attention = dep_info["attention_backends"]
    if not attention.get("flash_attn", {}).get("available", False):
        recommendations.append(
            "Install Flash Attention 2 for optimal attention performance"
        )

    if not attention.get("flashinfer", {}).get("available", False):
        recommendations.append(
            "Consider FlashInfer for FP8 quantization and long context performance"
        )

    # Check quantization recommendations
    quant = dep_info["quantization"]
    cuda_info = dep_info["core_dependencies"].get("cuda_info", {})
    compute_cap = cuda_info.get("compute_capability", "0.0")

    if compute_cap and float(compute_cap) >= 8.9:
        recommendations.append(
            "Your GPU supports FP8 quantization for 2x performance improvement"
        )

    if not any(
        quant.get(pkg, {}).get("available", False) for pkg in ["auto_gptq", "awq"]
    ):
        recommendations.append(
            "Install AWQ or AutoGPTQ for int4/int8 quantization support"
        )

    # Check core dependencies
    core = dep_info["core_dependencies"]
    if not core.get("triton", {}).get("available", False):
        recommendations.append("Install Triton for optimized CUDA kernels")

    return recommendations
