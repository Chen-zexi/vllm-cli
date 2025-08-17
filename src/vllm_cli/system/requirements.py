#!/usr/bin/env python3
"""
System requirements checking utilities.

Provides functions to check if the system meets requirements
for running vLLM and related dependencies.
"""
import logging
import subprocess

from .formatting import format_size
from .gpu import get_gpu_info
from .memory import get_memory_info

logger = logging.getLogger(__name__)


def check_system_requirements() -> bool:
    """
    Check if system meets requirements for running vLLM.

    Returns:
        True if requirements are met (with possible warnings), False for critical issues
    """
    issues = []
    warnings = []

    # Check for PyTorch (critical)
    try:
        import torch

        if not torch.cuda.is_available():
            warnings.append("CUDA is not available")
    except ImportError:
        issues.append("PyTorch is not installed")

    # Check for GPUs (warning only - allow UI to show)
    gpus = get_gpu_info()
    if not gpus:
        warnings.append("No NVIDIA GPUs detected")

    # Check memory (warning for low memory)
    mem_info = get_memory_info()
    if mem_info["total"] < 16 * 1024**3:  # Less than 16GB
        warnings.append(f"Low system memory: {format_size(mem_info['total'])}")

    # Log warnings
    if warnings:
        logger.warning(f"System warnings: {', '.join(warnings)}")

    # Only return False for critical issues
    if issues:
        logger.error(f"Critical system issues: {', '.join(issues)}")
        return False

    return True


def check_vllm_installation() -> bool:
    """
    Check if vLLM is installed in the current environment.

    Returns:
        True if vLLM is installed, False otherwise
    """
    try:
        # Check if vllm command is available
        result = subprocess.run(
            ["vllm", "--version"], capture_output=True, text=True, check=False
        )

        if result.returncode == 0:
            # Extract version from output
            output = result.stdout.strip()
            logger.info(f"vLLM found: {output[:100]}")  # Log first 100 chars
            return True

    except FileNotFoundError:
        pass

    # Try importing as Python module as fallback
    try:
        result = subprocess.run(
            ["python", "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            logger.info(f"vLLM module found: {result.stdout.strip()}")
            return True

    except FileNotFoundError:
        pass

    logger.warning(
        "vLLM not found. Please ensure vLLM is installed and available in your PATH."
    )
    return False
