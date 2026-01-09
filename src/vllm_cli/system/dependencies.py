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


def get_attention_backend_usability() -> Dict[str, Any]:
    """
    Get comprehensive attention backend usability information.
    Tests actual functionality, not just presence.

    Returns:
        Dictionary with detailed backend usability information
    """
    info = {}

    try:
        import torch
        from vllm.platforms import current_platform

        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            return {"error": "No CUDA devices available", "backends": {}}

        # Get device info
        device_cap = torch.cuda.get_device_capability()
        compute_cap = f"{device_cap[0]}.{device_cap[1]}"

        backends = {}

        # 1. vLLM Native Flash Attention
        vllm_fa_info = get_vllm_flash_attention_info()
        if vllm_fa_info:
            fa2_usable = vllm_fa_info.get("fa2_available") and vllm_fa_info.get(
                "fa2_gpu_supported"
            )
            fa3_usable = vllm_fa_info.get("fa3_available") and vllm_fa_info.get(
                "fa3_gpu_supported"
            )

            backends["vllm_flash_attn"] = {
                "name": "vLLM Flash Attention (Native)",
                "available": True,
                "usable": fa2_usable or fa3_usable,
                "version": vllm_fa_info.get("version"),
                "fa2_usable": fa2_usable,
                "fa3_usable": fa3_usable,
                "reason": (
                    vllm_fa_info.get("recommended_version", "FA2")
                    if (fa2_usable or fa3_usable)
                    else "GPU not supported"
                ),
                "priority": 1,  # Highest priority
            }

        # 2. External Flash Attention
        try:
            import flash_attn

            try:
                test_q = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
                flash_attn.flash_attn_func(test_q, test_q, test_q)
                backends["flash_attn"] = {
                    "name": "Flash Attention (External)",
                    "available": True,
                    "usable": True,
                    "version": getattr(flash_attn, "__version__", "installed"),
                    "reason": "External package, fully functional",
                    "priority": 2,
                }
            except (RuntimeError, torch.cuda.CudaError) as cuda_err:
                logger.debug(f"Flash Attention CUDA test failed: {cuda_err}")
                # Clear CUDA error state
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                backends["flash_attn"] = {
                    "name": "Flash Attention (External)",
                    "available": True,
                    "usable": False,
                    "version": getattr(flash_attn, "__version__", "installed"),
                    "reason": f"Not compatible with SM {compute_cap}",
                    "priority": 2,
                }
        except ImportError:
            backends["flash_attn"] = {
                "name": "Flash Attention (External)",
                "available": False,
                "usable": False,
                "version": None,
                "reason": "Not installed (pip install flash-attn)",
                "priority": 2,
            }
        except Exception as e:
            logger.debug(f"Flash Attention error: {e}")
            backends["flash_attn"] = {
                "name": "Flash Attention (External)",
                "available": True,
                "usable": False,
                "version": "error",
                "reason": f"Runtime error: {str(e)[:50]}",
                "priority": 2,
            }

        # 3. FlashInfer
        try:
            import flashinfer

            # FlashInfer is optimized for newer GPUs
            usable = float(compute_cap) >= 8.0  # Ampere+
            backends["flashinfer"] = {
                "name": "FlashInfer",
                "available": True,
                "usable": usable,
                "version": getattr(flashinfer, "__version__", "installed"),
                "reason": (
                    "Best for Blackwell/long context"
                    if usable
                    else f"Requires compute cap >= 8.0 (have {compute_cap})"
                ),
                "priority": 3,
            }
        except ImportError:
            backends["flashinfer"] = {
                "name": "FlashInfer",
                "available": False,
                "usable": False,
                "version": None,
                "reason": "Not installed (pip install flashinfer)",
                "priority": 3,
            }

        # 4. xFormers
        try:
            import xformers
            import xformers.ops

            # Skip actual test for Blackwell GPUs (SM 12.0) due to known
            # compatibility issues
            if float(compute_cap) >= 12.0:
                logger.debug(
                    f"Skipping xFormers GPU test on Blackwell (SM {compute_cap})"
                )
                backends["xformers"] = {
                    "name": "xFormers",
                    "available": True,
                    "usable": False,
                    "version": xformers.__version__,
                    "reason": f"Not yet compatible with Blackwell (SM {compute_cap})",
                    "priority": 4,
                }
            else:
                # Test memory efficient attention with better error handling
                try:
                    test = torch.randn(1, 8, 64, device="cuda", dtype=torch.float16)
                    xformers.ops.memory_efficient_attention(test, test, test)
                    backends["xformers"] = {
                        "name": "xFormers",
                        "available": True,
                        "usable": True,
                        "version": xformers.__version__,
                        "reason": "Memory efficient, good fallback",
                        "priority": 4,
                    }
                except (RuntimeError, torch.cuda.CudaError) as cuda_err:
                    # Handle CUDA errors gracefully
                    logger.debug(f"xFormers CUDA test failed: {cuda_err}")
                    # Clear CUDA error state
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    # Mark as available but not usable on this GPU
                    backends["xformers"] = {
                        "name": "xFormers",
                        "available": True,
                        "usable": False,
                        "version": xformers.__version__,
                        "reason": f"Not compatible with SM {compute_cap}",
                        "priority": 4,
                    }
                except Exception as e:
                    # Other unexpected errors
                    logger.debug(f"xFormers test error: {e}")
                    backends["xformers"] = {
                        "name": "xFormers",
                        "available": True,
                        "usable": False,
                        "version": xformers.__version__,
                        "reason": f"Test failed: {str(e)[:30]}",
                        "priority": 4,
                    }
        except ImportError:
            backends["xformers"] = {
                "name": "xFormers",
                "available": False,
                "usable": False,
                "version": None,
                "reason": "Not installed (pip install xformers)",
                "priority": 4,
            }

        # 5. Triton Attention Backends
        try:
            import triton

            # Check if Triton flash attention is enabled
            use_triton = os.environ.get(
                "VLLM_USE_TRITON_FLASH_ATTN", "True"
            ).lower() in ["true", "1"]
            # Check if using specific Triton backend variant
            attention_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "auto")
            is_triton_v1 = attention_backend == "TRITON_ATTN_VLLM_V1"

            # Triton V1 backend (for Ampere GPUs)
            if is_triton_v1 or float(compute_cap) in [8.0, 8.6]:  # Ampere
                backends["triton_v1"] = {
                    "name": "Triton Attention V1 (Ampere)",
                    "available": True,
                    "usable": is_triton_v1 or float(compute_cap) in [8.0, 8.6],
                    "version": triton.__version__,
                    "reason": (
                        "Best for Ampere GPUs (A100/A40)"
                        if float(compute_cap) in [8.0, 8.6]
                        else "Set VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1"
                    ),
                    "priority": 2,  # High priority for Ampere
                }

            # Regular Triton Flash Attention
            backends["triton"] = {
                "name": "Triton Flash Attention",
                "available": True,
                "usable": use_triton and not is_triton_v1,
                "version": triton.__version__,
                "reason": (
                    "Enabled via VLLM_USE_TRITON_FLASH_ATTN"
                    if use_triton
                    else "Disabled in environment"
                ),
                "priority": 5,
            }
        except ImportError:
            backends["triton"] = {
                "name": "Triton Flash Attention",
                "available": False,
                "usable": False,
                "version": None,
                "reason": "Triton not installed",
                "priority": 5,
            }

        # 6. TensorRT-LLM Backend
        try:
            import vllm.envs as envs

            # Check if TensorRT-LLM is enabled
            use_trtllm = os.environ.get("VLLM_USE_TRTLLM_ATTENTION")
            if use_trtllm is not None or hasattr(envs, "VLLM_USE_TRTLLM_ATTENTION"):
                # Check if it's actually available
                trtllm_usable = use_trtllm == "1" or (
                    float(compute_cap) >= 12.0
                )  # Blackwell benefits most
                backends["trtllm"] = {
                    "name": "TensorRT-LLM Attention",
                    "available": True,
                    "usable": trtllm_usable,
                    "version": "integrated",
                    "reason": (
                        "Recommended for Blackwell GPUs"
                        if float(compute_cap) >= 12.0
                        else "Set VLLM_USE_TRTLLM_ATTENTION=1"
                    ),
                    "priority": 1 if float(compute_cap) >= 12.0 else 6,
                }
        except Exception:
            pass

        # 7. PyTorch Native (Fallback)
        backends["pytorch_native"] = {
            "name": "PyTorch Native",
            "available": True,
            "usable": True,
            "version": torch.__version__,
            "reason": "Always available (slowest)",
            "priority": 99,
        }

        # Get actual backend that vLLM would select
        try:
            backend_cls = current_platform.get_attn_backend_cls(
                selected_backend=None,
                head_size=128,
                dtype=torch.float16,
                kv_cache_dtype="auto",
                block_size=16,
                use_v1=True,
                use_mla=False,
                has_sink=False,
            )
            # Parse the backend class name
            backend_name = (
                str(backend_cls).split(".")[-1].replace("'", "").replace(">", "")
            )
            info["auto_selected"] = backend_name
        except (RuntimeError, torch.cuda.CudaError) as cuda_err:
            logger.debug(f"CUDA error getting auto-selected backend: {cuda_err}")
            # Clear CUDA error state
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass  # Ignore errors during cleanup
            info["auto_selected"] = "Error: CUDA compatibility issue"
        except Exception as e:
            logger.debug(f"Error getting auto-selected backend: {e}")
            info["auto_selected"] = f"Error: {str(e)[:50]}"

        # Check current environment setting
        info["env_backend"] = os.environ.get("VLLM_ATTENTION_BACKEND", "auto")
        info["backends"] = backends
        info["compute_capability"] = compute_cap

        # Final CUDA cleanup to ensure no lingering errors
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass  # Ignore any cleanup errors

        return info

    except Exception as e:
        logger.debug(f"Error checking attention backend usability: {e}")
        # Try to clean up CUDA state even on general failure
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        return {"error": str(e), "backends": {}}


def get_attention_backend_info() -> Dict[str, Any]:
    """
    Get information about attention backend libraries and current configuration.

    Returns:
        Dictionary with attention backend information
    """
    info = {}

    # Check vLLM's native Flash Attention first
    vllm_fa_info = get_vllm_flash_attention_info()
    if vllm_fa_info:
        info["vllm_flash_attn"] = vllm_fa_info

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


def get_vllm_flash_attention_info() -> Dict[str, Any]:
    """
    Get information about vLLM's native Flash Attention implementation.

    Returns:
        Dictionary with vLLM Flash Attention details or None if not available
    """
    try:
        # Try to import vLLM's flash attention
        import torch
        from vllm.vllm_flash_attn import __version__ as fa_version
        from vllm.vllm_flash_attn.flash_attn_interface import (
            FA2_AVAILABLE,
            FA3_AVAILABLE,
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        info = {
            "name": "vLLM Flash Attention (Native)",
            "version": fa_version,
            "available": True,
            "fa2_available": FA2_AVAILABLE,
            "fa3_available": FA3_AVAILABLE,
        }

        # Check GPU support if CUDA is available
        if torch.cuda.is_available():
            compute_cap = torch.cuda.get_device_capability()
            info["compute_capability"] = f"{compute_cap[0]}.{compute_cap[1]}"

            # Check which versions are supported on this GPU
            try:
                fa2_supported = is_fa_version_supported(2)
                fa3_supported = is_fa_version_supported(3)
                info["fa2_gpu_supported"] = fa2_supported
                info["fa3_gpu_supported"] = fa3_supported

                # Get unsupported reasons if applicable
                if not fa2_supported:
                    info["fa2_unsupported_reason"] = (
                        fa_version_unsupported_reason(2) or "Not supported on this GPU"
                    )
                if not fa3_supported:
                    info["fa3_unsupported_reason"] = (
                        fa_version_unsupported_reason(3) or "Not supported on this GPU"
                    )

                # Determine which version is optimal
                if fa3_supported:
                    info["recommended_version"] = "FA3"
                elif fa2_supported:
                    info["recommended_version"] = "FA2"
                else:
                    info["recommended_version"] = "None (use alternative backend)"

            except Exception as e:
                logger.debug(f"Could not check FA GPU support: {e}")
                info["gpu_support_check_failed"] = str(e)
        else:
            info["compute_capability"] = "No CUDA device"
            info["fa2_gpu_supported"] = False
            info["fa3_gpu_supported"] = False

        return info

    except ImportError as e:
        logger.debug(f"vLLM Flash Attention not available: {e}")
        return None
    except Exception as e:
        logger.debug(f"Error checking vLLM Flash Attention: {e}")
        return None


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

    # Auto-detection logic with vLLM native Flash Attention priority
    vllm_fa = attention_info.get("vllm_flash_attn", {})
    if vllm_fa.get("available") and (
        vllm_fa.get("fa2_gpu_supported") or vllm_fa.get("fa3_gpu_supported")
    ):
        recommended = vllm_fa.get("recommended_version", "FA2")
        return f"vllm_flash_attn_{recommended} (auto-detected)"
    elif attention_info.get("flash_attn", {}).get("available", False):
        return "flash_attn (auto-detected)"
    elif attention_info.get("flashinfer", {}).get("available", False):
        # FlashInfer is preferred for newer GPUs
        return "flashinfer (auto-detected)"
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


def get_vllm_platform_info() -> Dict[str, Any]:
    """
    Get detailed vLLM platform information.

    Returns:
        Dictionary with platform details
    """
    try:
        import torch
        from vllm.platforms import current_platform

        info = {
            "available": True,
            "platform_type": current_platform.__class__.__name__,
            "device_name": current_platform.device_name,
            "device_type": current_platform.device_type,
            "ray_device_key": current_platform.ray_device_key,
            "is_cuda": (
                current_platform.is_cuda()
                if callable(current_platform.is_cuda)
                else current_platform.is_cuda
            ),
            "is_rocm": (
                current_platform.is_rocm()
                if callable(current_platform.is_rocm)
                else current_platform.is_rocm
            ),
            "is_cpu": (
                current_platform.is_cpu()
                if callable(current_platform.is_cpu)
                else current_platform.is_cpu
            ),
        }

        # Get device-specific information
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            devices = []
            for i in range(torch.cuda.device_count()):
                try:
                    cap = current_platform.get_device_capability(i)
                    devices.append(
                        {
                            "index": i,
                            "name": current_platform.get_device_name(i),
                            "capability": f"{cap.major}.{cap.minor}",
                            "capability_major": cap.major,
                            "capability_minor": cap.minor,
                            "memory_gb": current_platform.get_device_total_memory(i)
                            / (1024**3),
                            "uuid": (
                                current_platform.get_device_uuid(i)
                                if hasattr(current_platform, "get_device_uuid")
                                else None
                            ),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Error getting device {i} info: {e}")

            info["devices"] = devices

            # Map compute capability to architecture
            if devices:
                cap_major = devices[0]["capability_major"]
                cap_minor = devices[0]["capability_minor"]

                arch_map = {
                    (7, 0): "Volta V100",
                    (7, 2): "Volta (Jetson Xavier)",
                    (7, 5): "Turing (T4, RTX 20xx)",
                    (8, 0): "Ampere A100",
                    (8, 6): "Ampere (RTX 30xx, A40)",
                    (8, 7): "Ampere (Jetson Orin)",
                    (8, 9): "Ada Lovelace (RTX 40xx, L40)",
                    (9, 0): "Hopper (H100)",
                    (10, 0): "Blackwell (B100/B200, Jetson Thor)",
                    (12, 0): "Blackwell (RTX 6000)",
                }

                info["architecture"] = arch_map.get(
                    (cap_major, cap_minor), f"Unknown (SM {cap_major}.{cap_minor})"
                )

        return info

    except Exception as e:
        logger.debug(f"Error getting vLLM platform info: {e}")
        return {"available": False, "error": str(e)}


def get_vllm_capabilities() -> Dict[str, Any]:
    """
    Get vLLM capability information.

    Returns:
        Dictionary with capability details
    """
    try:
        import torch
        from vllm.platforms import current_platform

        info = {
            "available": True,
            "supports_fp8": (
                current_platform.supports_fp8()
                if hasattr(current_platform, "supports_fp8")
                else False
            ),
            "supports_custom_allreduce": (
                current_platform.use_custom_allreduce()
                if hasattr(current_platform, "use_custom_allreduce")
                else False
            ),
            "supports_pin_memory": (
                current_platform.is_pin_memory_available()
                if hasattr(current_platform, "is_pin_memory_available")
                else False
            ),
            "supports_sleep_mode": (
                current_platform.is_sleep_mode_available()
                if hasattr(current_platform, "is_sleep_mode_available")
                else False
            ),
        }

        # Get supported dtypes
        if hasattr(current_platform, "supported_dtypes"):
            info["supported_dtypes"] = [
                str(dt).split(".")[-1] for dt in current_platform.supported_dtypes
            ]

        # Get recommended attention backend
        if torch.cuda.is_available():
            try:
                backend_cls = current_platform.get_attn_backend_cls(
                    selected_backend=None,
                    head_size=128,
                    dtype=torch.float16,
                    kv_cache_dtype="auto",
                    block_size=16,
                    use_v1=True,
                    use_mla=False,
                    has_sink=False,
                )
                info["recommended_attention_backend"] = (
                    backend_cls.split(".")[-1]
                    if isinstance(backend_cls, str)
                    else str(backend_cls)
                )
            except Exception as e:
                info["recommended_attention_backend"] = f"Error: {e}"

        # Check V1 engine support
        info["supports_v1_engine"] = True  # vLLM v0.5+ supports V1

        # Get other capabilities
        if hasattr(current_platform, "get_cpu_architecture"):
            info["cpu_architecture"] = str(current_platform.get_cpu_architecture())

        if hasattr(current_platform, "fp8_dtype"):
            # fp8_dtype is a method, need to call it
            try:
                dtype = (
                    current_platform.fp8_dtype()
                    if callable(current_platform.fp8_dtype)
                    else current_platform.fp8_dtype
                )
                info["fp8_dtype"] = str(dtype)
            except Exception:
                info["fp8_dtype"] = "Unknown"

        return info

    except Exception as e:
        logger.debug(f"Error getting vLLM capabilities: {e}")
        return {"available": False, "error": str(e)}


def get_vllm_environment_status() -> Dict[str, Any]:
    """
    Get vLLM environment variable status.

    Returns:
        Dictionary with environment variable information
    """
    try:
        import os

        import vllm.envs as envs

        # Comprehensive environment variables by category
        categories = {
            "Core Settings": [
                "VLLM_USE_V1",
                "VLLM_ATTENTION_BACKEND",
                "VLLM_TARGET_DEVICE",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "VLLM_ENGINE_ITERATION_TIMEOUT_S",
                "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH",
            ],
            "TensorRT-LLM Integration": [
                "VLLM_USE_TRTLLM_ATTENTION",
                "VLLM_USE_TRTLLM_FP4_GEMM",
            ],
            "Performance & Optimization": [
                "VLLM_USE_TRITON_FLASH_ATTN",
                "VLLM_FLASHINFER_FORCE_TENSOR_CORES",
                "VLLM_FLASH_ATTN_VERSION",
                "VLLM_CPU_KVCACHE_SPACE",
                "VLLM_CPU_OMP_THREADS_BIND",
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "VLLM_USE_CUDNN_PREFILL",
                "VLLM_SKIP_P2P_CHECK",
                "VLLM_USE_DEEP_GEMM",
                "VLLM_USE_DEEP_GEMM_E8M0",
                "VLLM_SLEEP_WHEN_IDLE",
                "VLLM_ENABLE_CUDAGRAPH_GC",
                "VLLM_V1_USE_PREFILL_DECODE_ATTENTION",
                "VLLM_SKIP_DEEP_GEMM_WARMUP",
                "VLLM_USE_AITER_UNIFIED_ATTENTION",
            ],
            "MoE Models": [
                "VLLM_FUSED_MOE_CHUNK_SIZE",
                "VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING",
                "VLLM_CPU_MOE_PREPACK",
                "VLLM_MOE_DP_CHUNK_SIZE",
                "VLLM_MOE_ROUTING_SIMULATION_STRATEGY",
                "VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE",
                "VLLM_FLASHINFER_MOE_BACKEND",
                "VLLM_USE_FLASHINFER_MOE_FP4",
                "VLLM_USE_FLASHINFER_MOE_FP8",
                "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16",
                "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8",
            ],
            "Quantization": [
                "VLLM_USE_TRITON_AWQ",
                "VLLM_MARLIN_USE_ATOMIC_ADD",
                "VLLM_TEST_FORCE_FP8_MARLIN",
                "VLLM_MXFP4_USE_MARLIN",
                "VLLM_USE_TRTLLM_FP4_GEMM",
                "VLLM_USE_DEEP_GEMM_E8M0",
                "VLLM_USE_NVFP4_CT_EMULATIONS",
            ],
            "Compilation & Cache": [
                "VLLM_DISABLE_COMPILE_CACHE",
                "VLLM_USE_STANDALONE_COMPILE",
                "VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE",
                "VLLM_DISABLED_KERNELS",
                "VLLM_TUNED_CONFIG_FOLDER",
                "VLLM_USE_PRECOMPILED",
                "VLLM_XLA_CHECK_RECOMPILATION",
            ],
            "Logging & Debug": [
                "VLLM_CONFIGURE_LOGGING",
                "VLLM_LOGGING_LEVEL",
                "VLLM_LOG_STATS_INTERVAL",
                "VLLM_TRACE_FUNCTION",
                "VLLM_LOGGING_CONFIG_PATH",
                "VLLM_LOGGING_PREFIX",
                "VLLM_LOG_BATCHSIZE_INTERVAL",
                "VLLM_DEBUG_LOG_API_SERVER_RESPONSE",
                "VLLM_COMPUTE_NANS_IN_LOGITS",
                "VLLM_RINGBUFFER_WARNING_INTERVAL",
            ],
            "CUDA/GPU": [
                "CUDA_VISIBLE_DEVICES",
                "CUDA_HOME",
                "TORCH_CUDA_ARCH_LIST",
                "VLLM_CUDART_SO_PATH",
                "VLLM_NCCL_SO_PATH",
                "VLLM_KV_CACHE_LAYOUT",
            ],
            "Multimodal": [
                "VLLM_IMAGE_FETCH_TIMEOUT",
                "VLLM_AUDIO_FETCH_TIMEOUT",
                "VLLM_VIDEO_FETCH_TIMEOUT",
                "VLLM_VIDEO_LOADER_BACKEND",
                "VLLM_MM_INPUT_CACHE_GIB",
                "VLLM_MEDIA_LOADING_THREAD_COUNT",
                "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB",
            ],
            "Distributed & Networking": [
                "VLLM_HOST_IP",
                "VLLM_PORT",
                "VLLM_LOOPBACK_IP",
                "VLLM_DP_MASTER_IP",
                "VLLM_DP_MASTER_PORT",
                "VLLM_DP_RANK",
                "VLLM_DP_SIZE",
                "VLLM_PP_LAYER_PARTITION",
                "VLLM_ALL2ALL_BACKEND",
                "VLLM_USE_RAY_COMPILED_DAG",
                "VLLM_USE_RAY_SPMD_WORKER",
                "VLLM_RAY_PER_WORKER_GPUS",
            ],
            "LoRA & Adapters": [
                "VLLM_LORA_RESOLVER_CACHE_DIR",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
            ],
            "Paths & Storage": [
                "VLLM_CACHE_ROOT",
                "VLLM_CONFIG_ROOT",
                "VLLM_ASSETS_CACHE",
                "VLLM_MODEL_REDIRECT_PATH",
                "VLLM_RPC_BASE_PATH",
                "VLLM_XLA_CACHE_PATH",
                "VLLM_PLUGINS",
            ],
            "Profiling": [
                "VLLM_TORCH_PROFILER_DIR",
                "VLLM_TORCH_PROFILER_RECORD_SHAPES",
                "VLLM_TORCH_PROFILER_WITH_FLOPS",
                "VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY",
                "VLLM_TORCH_PROFILER_WITH_STACK",
            ],
            "Security & Privacy": [
                "VLLM_API_KEY",
                "VLLM_DO_NOT_TRACK",
                "VLLM_NO_USAGE_STATS",
                "VLLM_USAGE_SOURCE",
                "VLLM_USAGE_STATS_SERVER",
                "VLLM_ALLOW_INSECURE_SERIALIZATION",
            ],
        }

        info = {"available": True, "categories": {}}

        # Check if running on CUDA or ROCm
        import torch

        # Check the actual device name to determine if it's AMD or NVIDIA
        is_rocm = False
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device_name = torch.cuda.get_device_name(0).lower()
            is_rocm = "amd" in device_name or "radeon" in device_name

        # Add ROCm category only if on AMD GPU
        if is_rocm:
            categories["ROCm (AMD GPU)"] = [
                "VLLM_ROCM_CUSTOM_PAGED_ATTN",
                "VLLM_ROCM_FP8_PADDING",
                "VLLM_ROCM_MOE_PADDING",
                "VLLM_ROCM_USE_SKINNY_GEMM",
                "VLLM_ROCM_USE_AITER",
                "VLLM_ROCM_USE_AITER_LINEAR",
                "VLLM_ROCM_USE_AITER_MHA",
                "VLLM_ROCM_USE_AITER_MLA",
                "VLLM_ROCM_USE_AITER_MOE",
                "VLLM_ROCM_USE_AITER_PAGED_ATTN",
                "VLLM_ROCM_USE_AITER_RMSNORM",
                "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16",
                "VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB",
                "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION",
            ]

        for category, var_names in categories.items():
            cat_info = {}
            has_content = False

            for var_name in var_names:
                # Validate that the variable exists in vllm.envs
                if not hasattr(envs, var_name) and var_name not in [
                    "CUDA_VISIBLE_DEVICES",
                    "CUDA_HOME",
                    "TORCH_CUDA_ARCH_LIST",
                    "HF_TOKEN",
                ]:
                    continue  # Skip non-existent variables

                # Get actual environment value
                env_value = os.environ.get(var_name)

                # Get vLLM default value
                default_value = None
                if hasattr(envs, var_name):
                    default_value = getattr(envs, var_name)
                else:
                    # Handle special environment variables
                    if var_name == "CUDA_HOME":
                        default_value = os.environ.get("CUDA_HOME")
                    elif var_name == "CUDA_VISIBLE_DEVICES":
                        default_value = None
                    elif var_name == "TORCH_CUDA_ARCH_LIST":
                        default_value = None
                    elif var_name == "HF_TOKEN":
                        default_value = None

                # Determine if this variable should be shown
                is_set = env_value is not None
                is_modified = (
                    is_set and env_value != str(default_value)
                    if default_value is not None
                    else is_set
                )
                # Show critical variables even if they have None/False defaults
                is_critical = var_name in [
                    "VLLM_USE_TRTLLM_ATTENTION",
                    "VLLM_USE_TRTLLM_FP4_GEMM",
                    "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16",
                    "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8",
                    "VLLM_USE_FLASHINFER_MOE_FP4",
                    "VLLM_USE_FLASHINFER_MOE_FP8",
                    "VLLM_USE_AITER_UNIFIED_ATTENTION",
                    "VLLM_USE_DEEP_GEMM_E8M0",
                ]
                has_interesting_default = (
                    default_value not in [None, False, "", "not set", 0, -1, []]
                    or is_critical
                )

                # Only include if set or has interesting default
                if is_set or has_interesting_default:
                    cat_info[var_name] = {
                        "env_value": env_value,
                        "default_value": default_value,
                        "is_set": is_set,
                        "is_modified": is_modified,
                        "exists": hasattr(envs, var_name)
                        or var_name
                        in [
                            "CUDA_VISIBLE_DEVICES",
                            "CUDA_HOME",
                            "TORCH_CUDA_ARCH_LIST",
                            "HF_TOKEN",
                        ],
                    }
                    has_content = True

            # Only include category if it has content
            if has_content:
                info["categories"][category] = cat_info

        return info

    except Exception as e:
        logger.debug(f"Error getting vLLM environment status: {e}")
        return {"available": False, "error": str(e)}


def get_vllm_kernel_status() -> Dict[str, Any]:
    """
    Get vLLM custom kernel/operation status.

    Returns:
        Dictionary with kernel availability information
    """
    try:
        info = {"available": True, "kernels": {}}

        # Try to import custom ops
        try:
            from vllm import _custom_ops as ops

            info["custom_ops_loaded"] = True

            # Check specific kernel categories - only check kernels that
            # actually exist in _custom_ops
            kernel_groups = {
                "Attention": [
                    "paged_attention_v1",
                    "paged_attention_v2",
                ],
                "Position Encoding": [
                    "rotary_embedding",
                    "batched_rotary_embedding",
                ],
                "Quantization": [
                    "awq_dequantize",
                    "awq_gemm",
                    "gptq_gemm",
                    "gptq_marlin_gemm",
                    "awq_marlin_repack",
                    "scaled_fp8_quant",
                    "convert_fp8",
                ],
                "MoE (CUDA)": [
                    "topk_softmax",
                    "moe_align_block_size",
                    "moe_sum",
                    "cutlass_moe_mm",
                    "moe_wna16_gemm",
                    "marlin_gemm_moe_fake",
                ],
                "Memory": [
                    "copy_blocks",
                    "swap_blocks",
                ],
            }

            for group_name, kernel_names in kernel_groups.items():
                group_info = {}
                for kernel in kernel_names:
                    group_info[kernel] = hasattr(ops, kernel)
                info["kernels"][group_name] = group_info

        except ImportError as e:
            info["custom_ops_loaded"] = False
            info["error"] = str(e)

        # Check if vLLM flash attention is available
        try:
            from vllm.vllm_flash_attn import __version__ as fa_version

            info["vllm_flash_attn_version"] = fa_version
        except ImportError:
            info["vllm_flash_attn_version"] = None

        # Check for Python-level MoE implementations
        try:
            from vllm.model_executor.layers.fused_moe import fused_moe  # noqa: F401

            info["python_fused_moe"] = True
        except ImportError:
            info["python_fused_moe"] = False

        # Check for vLLM Flash Attention interface functions
        try:
            from vllm.vllm_flash_attn import flash_attn_interface

            info["flash_attn_interface"] = {
                "flash_attn_varlen_func": hasattr(
                    flash_attn_interface, "flash_attn_varlen_func"
                ),
                "flash_attn_with_kvcache": hasattr(
                    flash_attn_interface, "flash_attn_with_kvcache"
                ),
                "sparse_attn_func": hasattr(flash_attn_interface, "sparse_attn_func"),
                "sparse_attn_varlen_func": hasattr(
                    flash_attn_interface, "sparse_attn_varlen_func"
                ),
            }
        except ImportError:
            info["flash_attn_interface"] = None

        return info

    except Exception as e:
        logger.debug(f"Error getting vLLM kernel status: {e}")
        return {"available": False, "error": str(e)}


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
