#!/usr/bin/env python3
"""
Error Handling Package

This package provides comprehensive error handling for vLLM CLI including
exception hierarchies, error reporting, retry mechanisms, and recovery strategies.

Main Components:
- Base exceptions: VLLMCLIError and ValidationError
- Specific exception types: Server, Model, Config, System errors
- Error handlers: Reporting, formatting, and context management
- Retry mechanisms: Exponential backoff and configurable retry policies
- Recovery strategies: Automated recovery for common failure scenarios

The package is organized by error domain (server, model, config, system)
with cross-cutting concerns handled by handlers, retry, and recovery modules.
"""

# Base exception classes
from .base import ValidationError, VLLMCLIError
from .config import (
    ArgumentError,
    CompatibilityError,
    ConfigFileError,
    ConfigParseError,
    ConfigurationError,
    ConfigValidationError,
    DefaultsError,
    ProfileError,
    ProfileExistsError,
    ProfileNotFoundError,
    SchemaError,
)

# Error handling utilities
from .handlers import (
    ErrorReporter,
    error_boundary,
    error_reporter,
    format_error_for_user,
    get_error_help_text,
    graceful_shutdown,
    safe_operation,
)
from .model import (
    ModelCacheError,
    ModelCompatibilityError,
    ModelDownloadError,
    ModelError,
    ModelFormatError,
    ModelLoadError,
    ModelNotFoundError,
    ModelPermissionError,
    ModelRequirementsError,
    ModelSizeError,
    ModelValidationError,
)

# Recovery strategies
from .recovery import AutoRecovery, ErrorRecovery, apply_auto_recovery

# Retry mechanisms
from .retry import (
    RetryableOperation,
    RetryConfig,
    exponential_backoff,
    file_operation_retry,
    jittered_backoff,
    network_retry,
    retry_on_condition,
    retry_with_backoff,
    server_operation_retry,
    with_retries,
)

# Domain-specific exception classes
from .server import (
    PortInUseError,
    ServerCommunicationError,
    ServerConfigurationError,
    ServerError,
    ServerLogError,
    ServerNotFoundError,
    ServerProcessError,
    ServerResourceError,
    ServerStartupError,
    ServerStopError,
    ServerTimeoutError,
)
from .system import (
    CUDAError,
    DependencyError,
    DiskSpaceError,
    EnvironmentError,
    FileSystemError,
    GPUError,
    GPUMemoryError,
    GPUNotFoundError,
    MemoryError,
    NetworkError,
    PermissionError,
    ProcessError,
    SystemError,
)

__all__ = [
    # Base exceptions
    "VLLMCLIError",
    "ValidationError",
    # Server exceptions
    "ServerError",
    "ServerStartupError",
    "ServerStopError",
    "ServerCommunicationError",
    "ServerNotFoundError",
    "PortInUseError",
    "ServerTimeoutError",
    "ServerConfigurationError",
    "ServerResourceError",
    "ServerProcessError",
    "ServerLogError",
    # Model exceptions
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "ModelValidationError",
    "ModelFormatError",
    "ModelSizeError",
    "ModelPermissionError",
    "ModelDownloadError",
    "ModelCompatibilityError",
    "ModelCacheError",
    "ModelRequirementsError",
    # Config exceptions
    "ConfigurationError",
    "ConfigValidationError",
    "ConfigFileError",
    "ConfigParseError",
    "ProfileError",
    "ProfileNotFoundError",
    "ProfileExistsError",
    "SchemaError",
    "ArgumentError",
    "CompatibilityError",
    "DefaultsError",
    # System exceptions
    "SystemError",
    "GPUError",
    "GPUNotFoundError",
    "GPUMemoryError",
    "CUDAError",
    "DependencyError",
    "FileSystemError",
    "PermissionError",
    "DiskSpaceError",
    "MemoryError",
    "NetworkError",
    "EnvironmentError",
    "ProcessError",
    # Error handlers
    "ErrorReporter",
    "error_boundary",
    "safe_operation",
    "graceful_shutdown",
    "format_error_for_user",
    "get_error_help_text",
    "error_reporter",
    # Retry mechanisms
    "RetryConfig",
    "retry_with_backoff",
    "retry_on_condition",
    "RetryableOperation",
    "with_retries",
    "network_retry",
    "file_operation_retry",
    "server_operation_retry",
    "exponential_backoff",
    "jittered_backoff",
    # Recovery
    "ErrorRecovery",
    "AutoRecovery",
    "apply_auto_recovery",
]
