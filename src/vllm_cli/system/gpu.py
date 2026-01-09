#!/usr/bin/env python3
"""
GPU information and detection utilities.

Provides functions for detecting and gathering information about
available GPUs using multiple methods (nvidia-smi, PyTorch fallback).
Includes support for NVIDIA Jetson platforms (Orin, Thor, Xavier).
"""
import logging
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_int(value: str, default: int = 0) -> int:
    """
    Safely convert a string to int, handling N/A and empty values.

    This is particularly important for Jetson devices where nvidia-smi
    returns [N/A] for memory and utilization fields.

    Args:
        value: String value to convert
        default: Default value if conversion fails

    Returns:
        Integer value or default
    """
    if not value:
        return default
    value = value.strip()
    if value in ("[N/A]", "N/A", "[Not Supported]", ""):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get information about available GPUs.

    Supports both discrete NVIDIA GPUs and Jetson integrated GPUs.
    Falls back to PyTorch detection if nvidia-smi fails or returns
    incomplete data.

    Returns:
        List of GPU information dictionaries
    """
    gpus = []

    try:
        # Try nvidia-smi
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,  # Add timeout for Jetson devices
        )

        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(", ")
                if len(parts) >= 2:  # At least index and name required
                    # Safely parse each field, handling [N/A] values (common on Jetson)
                    memory_total = _safe_int(parts[2] if len(parts) > 2 else "0")
                    memory_used = _safe_int(parts[3] if len(parts) > 3 else "0")
                    memory_free = _safe_int(parts[4] if len(parts) > 4 else "0")

                    # Calculate memory_free if not provided but total is available
                    if memory_free == 0 and memory_total > 0 and memory_used >= 0:
                        memory_free = memory_total - memory_used

                    gpus.append(
                        {
                            "index": _safe_int(parts[0]),
                            "name": parts[1].strip() if len(parts) > 1 else "Unknown GPU",
                            "memory_total": memory_total * 1024 * 1024,  # MB to bytes
                            "memory_used": memory_used * 1024 * 1024,
                            "memory_free": memory_free * 1024 * 1024,
                            "utilization": _safe_int(parts[5] if len(parts) > 5 else "0"),
                            "temperature": _safe_int(parts[6] if len(parts) > 6 else "0"),
                        }
                    )

    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("nvidia-smi not available or failed")
    except Exception as e:
        logger.debug(f"nvidia-smi parsing error: {e}")

    # Fallback to PyTorch if nvidia-smi didn't find GPUs or returned incomplete data
    # This is essential for Jetson devices where nvidia-smi returns [N/A] for memory
    if not gpus or all(gpu.get("memory_total", 0) == 0 for gpu in gpus):
        logger.debug("Falling back to PyTorch GPU detection")
        pytorch_gpus = _try_pytorch_gpu_detection()
        if pytorch_gpus:
            gpus = pytorch_gpus

    return gpus


def get_cuda_version() -> Optional[str]:
    """
    Get CUDA version.

    Returns:
        CUDA version string or None
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.version.cuda
    except ImportError:
        pass

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def _try_pytorch_gpu_detection() -> List[Dict[str, Any]]:
    """
    Try to detect GPUs using PyTorch as a fallback method.

    Returns:
        List of GPU information dictionaries or empty list if detection fails
    """
    gpus = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    {
                        "index": i,
                        "name": props.name,
                        "memory_total": props.total_memory,
                        "memory_used": 0,  # PyTorch doesn't provide current usage easily
                        "memory_free": props.total_memory,
                        "utilization": 0,  # Not available through PyTorch
                        "temperature": 0,  # Not available through PyTorch
                    }
                )
            logger.info(f"Detected {len(gpus)} GPU(s) via PyTorch fallback")
    except ImportError:
        logger.debug("PyTorch not available for GPU detection")
    except Exception as e:
        logger.warning(f"PyTorch GPU detection failed: {e}")

    return gpus
