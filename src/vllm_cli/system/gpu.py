#!/usr/bin/env python3
"""
GPU information and detection utilities.

Provides functions for detecting and gathering information about
available GPUs using multiple methods (nvidia-smi, PyTorch fallback).
"""
import logging
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get information about available GPUs.

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
        )

        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(", ")
                if len(parts) >= 7:
                    gpus.append(
                        {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total": int(parts[2])
                            * 1024
                            * 1024,  # Convert to bytes
                            "memory_used": int(parts[3]) * 1024 * 1024,
                            "memory_free": int(parts[4]) * 1024 * 1024,
                            "utilization": int(parts[5]) if parts[5] else 0,
                            "temperature": int(parts[6]) if parts[6] else 0,
                        }
                    )

    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("nvidia-smi not available or failed")

        # Try to get info from torch as fallback
        gpus = _try_pytorch_gpu_detection()
    except Exception as e:
        logger.warning(f"Unexpected error getting GPU info: {e}")

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
