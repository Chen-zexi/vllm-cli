#!/usr/bin/env python3
"""
Error recovery strategies for common failure scenarios.

Provides automated recovery mechanisms for transient failures
and graceful degradation for persistent issues.
"""
import logging
from typing import Any, Dict, List, Optional

from .config import ConfigurationError
from .model import ModelError, ModelNotFoundError
from .server import PortInUseError, ServerError, ServerStartupError
from .system import GPUError, GPUMemoryError, GPUNotFoundError

logger = logging.getLogger(__name__)


class ErrorRecovery:
    """
    Error recovery strategies for common failure scenarios.

    Provides automated recovery mechanisms for transient failures
    and graceful degradation for persistent issues.
    """

    @staticmethod
    def handle_gpu_error(error: GPUError) -> Dict[str, Any]:
        """
        Handle GPU-related errors with recovery suggestions.

        Args:
            error: GPU error to handle

        Returns:
            Dictionary with recovery suggestions
        """
        recovery_info = {
            "error": error,
            "suggestions": [],
            "fallback_available": False,
            "auto_recovery": None,
        }

        if isinstance(error, GPUNotFoundError):
            recovery_info["suggestions"] = [
                "Check NVIDIA GPU installation",
                "Verify NVIDIA drivers are installed",
                "Ensure CUDA toolkit is available",
                "Try running 'nvidia-smi' to test GPU access",
            ]
            # Offer CPU mode as fallback
            recovery_info["fallback_available"] = True
            recovery_info["auto_recovery"] = "switch_to_cpu_mode"

        elif isinstance(error, GPUMemoryError):
            recovery_info["suggestions"] = [
                "Reduce gpu_memory_utilization (try 0.8 or lower)",
                "Use quantization (awq, gptq, or bitsandbytes)",
                "Increase tensor_parallel_size to distribute across GPUs",
                "Consider CPU offloading with cpu_offload_gb",
            ]
            recovery_info["fallback_available"] = True
            recovery_info["auto_recovery"] = "reduce_memory_usage"

        elif "not found" in str(error).lower():
            recovery_info["suggestions"] = [
                "Check NVIDIA GPU installation",
                "Verify NVIDIA drivers are installed",
                "Ensure CUDA toolkit is available",
                "Try running 'nvidia-smi' to test GPU access",
            ]
        else:
            recovery_info["suggestions"] = [
                "Check GPU status with 'nvidia-smi'",
                "Restart GPU drivers",
                "Check system logs for GPU errors",
                "Try switching to CPU mode temporarily",
            ]

        return recovery_info

    @staticmethod
    def handle_model_error(error: ModelError) -> Dict[str, Any]:
        """
        Handle model-related errors with recovery suggestions.

        Args:
            error: Model error to handle

        Returns:
            Dictionary with recovery suggestions
        """
        recovery_info = {
            "error": error,
            "suggestions": [],
            "fallback_available": False,
            "auto_recovery": None,
        }

        if isinstance(error, ModelNotFoundError):
            recovery_info["suggestions"] = [
                "Check model name spelling",
                "Verify model is downloaded locally",
                "Use 'vllm-cli models' to list available models",
                "Try downloading with hf-model-tool",
            ]
            recovery_info["auto_recovery"] = "suggest_similar_models"

        elif "validation" in str(error).lower():
            recovery_info["suggestions"] = [
                "Check model format (ensure it's compatible with vLLM)",
                "Verify config.json exists in model directory",
                "Try trust_remote_code=true if using custom model",
                "Check model architecture is supported by vLLM",
            ]
        elif "size" in str(error).lower() or "memory" in str(error).lower():
            recovery_info["suggestions"] = [
                "Use model quantization to reduce memory requirements",
                "Try tensor parallelism across multiple GPUs",
                "Use CPU offloading for part of the model",
                "Switch to a smaller model variant",
            ]
            recovery_info["fallback_available"] = True
            recovery_info["auto_recovery"] = "suggest_quantization"
        else:
            recovery_info["suggestions"] = [
                "Check model files are not corrupted",
                "Verify sufficient disk space",
                "Try re-downloading the model",
                "Check file permissions",
            ]

        return recovery_info

    @staticmethod
    def handle_server_error(error: ServerError) -> Dict[str, Any]:
        """
        Handle server-related errors with recovery suggestions.

        Args:
            error: Server error to handle

        Returns:
            Dictionary with recovery suggestions
        """
        recovery_info = {
            "error": error,
            "suggestions": [],
            "fallback_available": False,
            "auto_recovery": None,
        }

        if isinstance(error, PortInUseError):
            recovery_info["suggestions"] = [
                "Try a different port number",
                "Stop existing servers with 'vllm-cli stop --all'",
                "Check for other processes using the port",
                "Use automatic port selection",
            ]
            recovery_info["fallback_available"] = True
            recovery_info["auto_recovery"] = "find_available_port"

        elif isinstance(error, ServerStartupError):
            recovery_info["suggestions"] = [
                "Check server logs for detailed error information",
                "Verify all dependencies are installed",
                "Try reducing model complexity or size",
                "Check available system resources",
            ]
            recovery_info["auto_recovery"] = "diagnose_startup_failure"

        elif "timeout" in str(error).lower():
            recovery_info["suggestions"] = [
                "Increase startup timeout",
                "Check system resource availability",
                "Try with a smaller model",
                "Monitor system performance during startup",
            ]
        else:
            recovery_info["suggestions"] = [
                "Check server logs for detailed information",
                "Verify server configuration",
                "Try restarting the server",
                "Check system resources and permissions",
            ]

        return recovery_info

    @staticmethod
    def handle_configuration_error(error: ConfigurationError) -> Dict[str, Any]:
        """
        Handle configuration-related errors with recovery suggestions.

        Args:
            error: Configuration error to handle

        Returns:
            Dictionary with recovery suggestions
        """
        recovery_info = {
            "error": error,
            "suggestions": [],
            "fallback_available": False,
            "auto_recovery": None,
        }

        if "validation" in str(error).lower():
            recovery_info["suggestions"] = [
                "Check configuration syntax and format",
                "Verify all required fields are present",
                "Use 'vllm-cli config validate' to check settings",
                "Reset to default configuration if needed",
            ]
            recovery_info["auto_recovery"] = "suggest_valid_values"

        elif "profile" in str(error).lower():
            recovery_info["suggestions"] = [
                "Check profile name spelling",
                "Use 'vllm-cli profiles list' to see available profiles",
                "Create profile if it doesn't exist",
                "Verify profile configuration is valid",
            ]
        else:
            recovery_info["suggestions"] = [
                "Check configuration file permissions",
                "Verify configuration directory exists",
                "Reset to default configuration",
                "Check for syntax errors in configuration files",
            ]

        return recovery_info


class AutoRecovery:
    """
    Automated recovery mechanisms for common errors.

    Provides methods to automatically attempt recovery from
    common failure scenarios.
    """

    @staticmethod
    def switch_to_cpu_mode(original_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify configuration to use CPU mode instead of GPU.

        Args:
            original_config: Original server configuration

        Returns:
            Modified configuration for CPU mode
        """
        cpu_config = original_config.copy()

        # Remove GPU-specific options
        gpu_options = [
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "quantization",
            "kv_cache_dtype",
        ]

        for option in gpu_options:
            cpu_config.pop(option, None)

        logger.info("Switched configuration to CPU mode")
        return cpu_config

    @staticmethod
    def reduce_memory_usage(original_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify configuration to reduce memory usage.

        Args:
            original_config: Original server configuration

        Returns:
            Modified configuration with reduced memory usage
        """
        memory_config = original_config.copy()

        # Reduce memory utilization
        memory_config["gpu_memory_utilization"] = 0.8

        # Enable CPU offloading if not set
        if "cpu_offload_gb" not in memory_config:
            memory_config["cpu_offload_gb"] = 4

        # Reduce sequence length if possible
        if memory_config.get("max_model_len", 0) > 4096:
            memory_config["max_model_len"] = 4096

        logger.info("Reduced memory usage in configuration")
        return memory_config

    @staticmethod
    def suggest_quantization(model_name: str) -> List[str]:
        """
        Suggest quantization options for a model.

        Args:
            model_name: Name of the model

        Returns:
            List of suggested quantization methods
        """
        suggestions = []

        # Check if model supports common quantization methods
        if any(arch in model_name.lower() for arch in ["llama", "qwen", "mistral"]):
            suggestions.extend(["awq", "gptq", "bitsandbytes"])
        else:
            suggestions.extend(["bitsandbytes", "fp8"])

        return suggestions

    @staticmethod
    def find_available_port(start_port: int = 8000) -> int:
        """
        Find the next available port.

        Args:
            start_port: Port to start searching from

        Returns:
            Available port number
        """
        import socket

        port = start_port
        while port < 65535:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1

        raise RuntimeError("No available ports found")

    @staticmethod
    def suggest_similar_models(
        model_name: str, available_models: List[str]
    ) -> List[str]:
        """
        Suggest similar models based on name similarity.

        Args:
            model_name: Requested model name
            available_models: List of available model names

        Returns:
            List of suggested model names
        """
        suggestions = []
        model_lower = model_name.lower()

        # Look for exact partial matches first
        for available in available_models:
            if model_lower in available.lower() or available.lower() in model_lower:
                suggestions.append(available)

        # If no partial matches, look for similar patterns
        if not suggestions:
            model_parts = model_lower.split("/")[-1].split("-")
            for available in available_models:
                available_parts = available.lower().split("/")[-1].split("-")
                if any(
                    part in available_parts for part in model_parts if len(part) > 2
                ):
                    suggestions.append(available)

        return suggestions[:5]  # Return top 5 suggestions


def apply_auto_recovery(
    error: Exception, context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Apply automatic recovery for an error if possible.

    Args:
        error: The error that occurred
        context: Context information including configuration

    Returns:
        Recovery result dictionary or None if no recovery available
    """
    recovery_strategies = {
        GPUError: ErrorRecovery.handle_gpu_error,
        ModelError: ErrorRecovery.handle_model_error,
        ServerError: ErrorRecovery.handle_server_error,
        ConfigurationError: ErrorRecovery.handle_configuration_error,
    }

    for error_type, handler in recovery_strategies.items():
        if isinstance(error, error_type):
            recovery_info = handler(error)

            # Attempt automatic recovery if available
            auto_recovery = recovery_info.get("auto_recovery")
            if auto_recovery and context:
                return _execute_auto_recovery(auto_recovery, context)

    return None


def _execute_auto_recovery(
    recovery_type: str, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute automatic recovery strategy."""
    recovery_methods = {
        "switch_to_cpu_mode": AutoRecovery.switch_to_cpu_mode,
        "reduce_memory_usage": AutoRecovery.reduce_memory_usage,
        "find_available_port": lambda config: {
            "port": AutoRecovery.find_available_port()
        },
    }

    if recovery_type in recovery_methods:
        try:
            original_config = context.get("config", {})
            result = recovery_methods[recovery_type](original_config)
            logger.info(f"Applied automatic recovery: {recovery_type}")
            return {"success": True, "modified_config": result}
        except Exception as e:
            logger.error(f"Auto-recovery failed for {recovery_type}: {e}")

    return {"success": False, "error": "Recovery method not available"}
