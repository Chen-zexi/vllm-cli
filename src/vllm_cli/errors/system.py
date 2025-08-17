#!/usr/bin/env python3
"""
System-related exception classes.

Defines exceptions for system operations including GPU, dependency,
and file system errors.
"""
from typing import Optional

from .base import VLLMCLIError


class SystemError(VLLMCLIError):
    """Base class for system-related errors."""

    def __init__(self, message: str, system_component: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="SYSTEM_ERROR", **kwargs)
        if system_component:
            self.add_context("system_component", system_component)

    def _generate_user_message(self) -> str:
        return "System error occurred. Check system requirements and permissions."


class GPUError(SystemError):
    """GPU-related operation error."""

    def __init__(
        self,
        message: str,
        gpu_id: Optional[int] = None,
        gpu_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message, error_code="GPU_ERROR", system_component="GPU", **kwargs
        )
        if gpu_id is not None:
            self.add_context("gpu_id", gpu_id)
        if gpu_name:
            self.add_context("gpu_name", gpu_name)

    def _generate_user_message(self) -> str:
        return "GPU error. Check GPU drivers and CUDA installation."


class GPUNotFoundError(GPUError):
    """No suitable GPU found."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="GPU_NOT_FOUND", **kwargs)

    def _generate_user_message(self) -> str:
        return (
            "No compatible GPU found. Install NVIDIA GPU and drivers, or use CPU mode."
        )


class GPUMemoryError(GPUError):
    """Insufficient GPU memory."""

    def __init__(
        self,
        message: str,
        required_memory: Optional[str] = None,
        available_memory: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="GPU_MEMORY_ERROR", **kwargs)
        if required_memory:
            self.add_context("required_memory", required_memory)
        if available_memory:
            self.add_context("available_memory", available_memory)

    def _generate_user_message(self) -> str:
        return "Insufficient GPU memory. Reduce model size or use quantization."


class CUDAError(SystemError):
    """CUDA-related error."""

    def __init__(self, message: str, cuda_version: Optional[str] = None, **kwargs):
        super().__init__(
            message, error_code="CUDA_ERROR", system_component="CUDA", **kwargs
        )
        if cuda_version:
            self.add_context("cuda_version", cuda_version)

    def _generate_user_message(self) -> str:
        return "CUDA error. Check CUDA installation and compatibility."


class DependencyError(SystemError):
    """Missing or incompatible dependency."""

    def __init__(
        self,
        message: str,
        dependency_name: Optional[str] = None,
        required_version: Optional[str] = None,
        installed_version: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="DEPENDENCY_ERROR", **kwargs)
        if dependency_name:
            self.add_context("dependency_name", dependency_name)
        if required_version:
            self.add_context("required_version", required_version)
        if installed_version:
            self.add_context("installed_version", installed_version)

    def _generate_user_message(self) -> str:
        dep_name = self.get_context("dependency_name", "dependency")
        return f"Missing or incompatible {dep_name}. Install required dependencies."


class FileSystemError(SystemError):
    """File system operation error."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="FILESYSTEM_ERROR", **kwargs)
        if file_path:
            self.add_context("file_path", file_path)
        if operation:
            self.add_context("operation", operation)

    def _generate_user_message(self) -> str:
        operation = self.get_context("operation", "File operation")
        return f"{operation} failed. Check file permissions and disk space."


class PermissionError(FileSystemError):
    """Permission denied for file or directory access."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="PERMISSION_DENIED", **kwargs)

    def _generate_user_message(self) -> str:
        file_path = self.get_context("file_path", "file or directory")
        return f"Permission denied accessing {file_path}. Check file permissions."


class DiskSpaceError(FileSystemError):
    """Insufficient disk space."""

    def __init__(
        self,
        message: str,
        required_space: Optional[str] = None,
        available_space: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="DISK_SPACE_ERROR", **kwargs)
        if required_space:
            self.add_context("required_space", required_space)
        if available_space:
            self.add_context("available_space", available_space)

    def _generate_user_message(self) -> str:
        return "Insufficient disk space. Free up space and try again."


class MemoryError(SystemError):
    """System memory error."""

    def __init__(
        self,
        message: str,
        memory_type: str = "system",
        required_memory: Optional[str] = None,
        available_memory: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="MEMORY_ERROR", **kwargs)
        self.add_context("memory_type", memory_type)
        if required_memory:
            self.add_context("required_memory", required_memory)
        if available_memory:
            self.add_context("available_memory", available_memory)

    def _generate_user_message(self) -> str:
        memory_type = self.get_context("memory_type", "system")
        return f"Insufficient {memory_type} memory. Close other applications or upgrade hardware."


class NetworkError(SystemError):
    """Network connectivity error."""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="NETWORK_ERROR", **kwargs)
        if url:
            self.add_context("url", url)
        if status_code:
            self.add_context("status_code", status_code)

    def _generate_user_message(self) -> str:
        return "Network error. Check internet connection and try again."


class EnvironmentError(SystemError):
    """System environment configuration error."""

    def __init__(
        self,
        message: str,
        env_var: Optional[str] = None,
        expected_value: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="ENVIRONMENT_ERROR", **kwargs)
        if env_var:
            self.add_context("env_var", env_var)
        if expected_value:
            self.add_context("expected_value", expected_value)

    def _generate_user_message(self) -> str:
        env_var = self.get_context("env_var")
        if env_var:
            return f"Environment variable {env_var} not properly configured."
        return "System environment not properly configured."


class ProcessError(SystemError):
    """System process operation error."""

    def __init__(
        self,
        message: str,
        process_name: Optional[str] = None,
        process_id: Optional[int] = None,
        exit_code: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="PROCESS_ERROR", **kwargs)
        if process_name:
            self.add_context("process_name", process_name)
        if process_id:
            self.add_context("process_id", process_id)
        if exit_code is not None:
            self.add_context("exit_code", exit_code)

    def _generate_user_message(self) -> str:
        process_name = self.get_context("process_name", "Process")
        exit_code = self.get_context("exit_code")
        if exit_code:
            return f"{process_name} failed with exit code {exit_code}."
        return f"{process_name} operation failed."
