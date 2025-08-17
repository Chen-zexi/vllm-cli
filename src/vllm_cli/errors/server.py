#!/usr/bin/env python3
"""
Server-related exception classes.

Defines exceptions for vLLM server operations including startup,
communication, and lifecycle management errors.
"""
from typing import Any, Optional

from .base import VLLMCLIError


class ServerError(VLLMCLIError):
    """Base class for server-related errors."""

    def __init__(
        self,
        message: str,
        server_id: Optional[str] = None,
        port: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="SERVER_ERROR", **kwargs)
        if server_id:
            self.add_context("server_id", server_id)
        if port:
            self.add_context("port", port)

    def _generate_user_message(self) -> str:
        return "Server operation failed. Check logs for details."


class ServerStartupError(ServerError):
    """Error during server startup process."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="SERVER_STARTUP_ERROR", **kwargs)

    def _generate_user_message(self) -> str:
        return "Failed to start vLLM server. Check configuration and system resources."


class ServerStopError(ServerError):
    """Error during server shutdown process."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="SERVER_STOP_ERROR", **kwargs)

    def _generate_user_message(self) -> str:
        return "Failed to stop vLLM server. Process may need manual termination."


class ServerCommunicationError(ServerError):
    """Error communicating with running server."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="SERVER_COMMUNICATION_ERROR", **kwargs)

    def _generate_user_message(self) -> str:
        return (
            "Cannot communicate with server. Server may not be running or responding."
        )


class ServerNotFoundError(ServerError):
    """Server not found in registry."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="SERVER_NOT_FOUND", **kwargs)

    def _generate_user_message(self) -> str:
        return "Server not found. Use 'vllm-cli status' to see active servers."


class PortInUseError(ServerError):
    """Port is already in use by another process."""

    def __init__(self, message: str, port: int, **kwargs):
        super().__init__(message, error_code="PORT_IN_USE", port=port, **kwargs)

    def _generate_user_message(self) -> str:
        port = self.get_context("port")
        return f"Port {port} is already in use. Try a different port or stop the conflicting service."


class ServerTimeoutError(ServerError):
    """Server operation timed out."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="SERVER_TIMEOUT", **kwargs)
        if operation:
            self.add_context("operation", operation)
        if timeout_seconds:
            self.add_context("timeout_seconds", timeout_seconds)

    def _generate_user_message(self) -> str:
        operation = self.get_context("operation", "Server operation")
        return f"{operation} timed out. Server may be overloaded or unresponsive."


class ServerConfigurationError(ServerError):
    """Invalid server configuration."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Any = None,
        **kwargs,
    ):
        super().__init__(message, error_code="SERVER_CONFIG_ERROR", **kwargs)
        if config_key:
            self.add_context("config_key", config_key)
        if config_value is not None:
            self.add_context("config_value", config_value)

    def _generate_user_message(self) -> str:
        config_key = self.get_context("config_key")
        if config_key:
            return (
                f"Invalid server configuration for '{config_key}'. Check your settings."
            )
        return "Invalid server configuration. Check your settings."


class ServerResourceError(ServerError):
    """Insufficient resources for server operation."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        required: Optional[str] = None,
        available: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="SERVER_RESOURCE_ERROR", **kwargs)
        if resource_type:
            self.add_context("resource_type", resource_type)
        if required:
            self.add_context("required", required)
        if available:
            self.add_context("available", available)

    def _generate_user_message(self) -> str:
        resource_type = self.get_context("resource_type", "system resources")
        return f"Insufficient {resource_type}. Free up resources or reduce model requirements."


class ServerProcessError(ServerError):
    """Error with server process management."""

    def __init__(
        self,
        message: str,
        process_id: Optional[int] = None,
        process_status: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="SERVER_PROCESS_ERROR", **kwargs)
        if process_id:
            self.add_context("process_id", process_id)
        if process_status:
            self.add_context("process_status", process_status)

    def _generate_user_message(self) -> str:
        return "Server process error. Check system logs and process status."


class ServerLogError(ServerError):
    """Error accessing or parsing server logs."""

    def __init__(self, message: str, log_path: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="SERVER_LOG_ERROR", **kwargs)
        if log_path:
            self.add_context("log_path", log_path)

    def _generate_user_message(self) -> str:
        return "Cannot access server logs. Check file permissions and disk space."
