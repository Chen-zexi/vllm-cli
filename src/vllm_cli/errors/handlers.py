#!/usr/bin/env python3
"""
Error handling utilities for vLLM CLI.

This module provides centralized error handling functionality including
error reporting, context management, and user-friendly error formatting.
"""
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Type

from .base import VLLMCLIError
from .config import ValidationError
from .model import ModelError
from .server import ServerError
from .system import GPUError

logger = logging.getLogger(__name__)


class ErrorReporter:
    """
    Centralized error reporting and logging.

    Provides consistent error reporting across the application
    with different verbosity levels for users and developers.
    """

    def __init__(self):
        self.error_counts: Dict[str, int] = {}

    def report_error(
        self,
        error: VLLMCLIError,
        user_facing: bool = True,
        include_context: bool = False,
    ) -> str:
        """
        Report an error with appropriate formatting.

        Args:
            error: The error to report
            user_facing: Whether this is for user display
            include_context: Whether to include technical context

        Returns:
            Formatted error message
        """
        # Track error frequency
        error_code = getattr(error, "error_code", "UNKNOWN")
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1

        if user_facing:
            message = error.get_user_message()
            if include_context and hasattr(error, "context") and error.context:
                context_str = self._format_context(error.context)
                message += f"\nDetails: {context_str}"
        else:
            message = str(error)
            if hasattr(error, "context") and error.context:
                context_str = self._format_context(error.context)
                message += f"\nContext: {context_str}"

        # Log with appropriate level
        if isinstance(error, (ValidationError,)):
            logger.warning(f"Configuration error: {error}")
        elif isinstance(error, (ServerError, ModelError, GPUError)):
            logger.error(f"Runtime error: {error}")
        else:
            logger.error(f"General error: {error}")

        return message

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format error context for display."""
        formatted_items = []
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                formatted_items.append(f"{key}={value}")
            else:
                formatted_items.append(f"{key}={type(value).__name__}")

        return ", ".join(formatted_items)

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts."""
        return self.error_counts.copy()

    def reset_error_counts(self) -> None:
        """Reset error counting."""
        self.error_counts.clear()


@contextmanager
def error_boundary(
    operation: str,
    error_type: Type[VLLMCLIError] = VLLMCLIError,
    context: Optional[Dict[str, Any]] = None,
    fallback_result: Any = None,
    suppress_errors: bool = False,
):
    """
    Context manager for consistent error handling and conversion.

    Args:
        operation: Description of the operation being performed
        error_type: VLLMCLIError subclass to raise on failure
        context: Additional context to include in errors
        fallback_result: Value to return if error is suppressed
        suppress_errors: If True, return fallback_result instead of raising

    Yields:
        Context for the protected operation

    Raises:
        error_type: If operation fails and suppress_errors is False
    """
    try:
        yield
    except VLLMCLIError:
        # Re-raise VLLMCLIError as-is
        if not suppress_errors:
            raise
        logger.exception(f"Suppressed error during {operation}")
        return fallback_result
    except Exception as e:
        error_message = f"Error during {operation}: {e}"
        logger.error(error_message)

        if suppress_errors:
            logger.exception(f"Suppressed error during {operation}")
            return fallback_result

        # Convert to appropriate VLLMCLIError subclass
        raise error_type(
            error_message,
            error_code=f"{operation.upper().replace(' ', '_')}_ERROR",
            context=context or {},
        ) from e


def safe_operation(operation_name: str, fallback_result: Any = None):
    """
    Decorator for operations that should never crash the application.

    Args:
        operation_name: Name of the operation for error reporting
        fallback_result: Value to return if operation fails

    Returns:
        Decorated function that returns fallback_result on error
    """

    def decorator(func: Callable) -> Callable:
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Safe operation '{operation_name}' failed: {e}")

                # Convert to VLLMCLIError if needed
                if not isinstance(e, VLLMCLIError):
                    e = VLLMCLIError(
                        f"Operation '{operation_name}' failed: {e}",
                        error_code="SAFE_OPERATION_ERROR",
                    )

                # Report error but don't raise
                error_reporter.report_error(e, user_facing=False)
                return fallback_result

        return wrapper

    return decorator


def graceful_shutdown(cleanup_functions: list = None):
    """
    Context manager for graceful shutdown with cleanup.

    Args:
        cleanup_functions: List of functions to call during cleanup
    """
    cleanup_functions = cleanup_functions or []

    @contextmanager
    def shutdown_context():
        try:
            yield
        except KeyboardInterrupt:
            logger.info("Received shutdown signal, cleaning up...")
            for cleanup_func in cleanup_functions:
                try:
                    cleanup_func()
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")
            logger.info("Cleanup completed")
            raise
        except Exception as e:
            logger.error(f"Unexpected error, attempting cleanup: {e}")
            for cleanup_func in cleanup_functions:
                try:
                    cleanup_func()
                except Exception as cleanup_error:
                    logger.error(f"Error during emergency cleanup: {cleanup_error}")
            raise

    return shutdown_context()


def format_error_for_user(error: Exception) -> str:
    """
    Format an error for user display.

    Args:
        error: Exception to format

    Returns:
        User-friendly error message
    """
    if isinstance(error, VLLMCLIError):
        return error.get_user_message()

    # Handle common standard exceptions
    error_messages = {
        FileNotFoundError: "File not found",
        PermissionError: "Permission denied",
        ConnectionError: "Connection error",
        TimeoutError: "Operation timed out",
        KeyboardInterrupt: "Operation cancelled by user",
    }

    error_type = type(error)
    if error_type in error_messages:
        return error_messages[error_type]

    # Generic fallback
    return f"An error occurred: {str(error)}"


def get_error_help_text(error: VLLMCLIError) -> Optional[str]:
    """
    Get helpful text for resolving an error.

    Args:
        error: The error to get help for

    Returns:
        Help text or None if no specific help available
    """
    help_texts = {
        "MODEL_NOT_FOUND": "Use 'vllm-cli models' to list available models",
        "GPU_NOT_FOUND": "Install NVIDIA GPU drivers and CUDA toolkit",
        "PORT_IN_USE": "Use 'vllm-cli status' to see active servers",
        "PERMISSION_DENIED": "Check file permissions and ownership",
        "DISK_SPACE_ERROR": "Free up disk space and try again",
        "NETWORK_ERROR": "Check internet connection and firewall settings",
    }

    return help_texts.get(error.error_code)


# Global error reporter instance
error_reporter = ErrorReporter()
