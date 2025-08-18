#!/usr/bin/env python3
"""
Base exception classes for vLLM CLI.

Defines the base VLLMCLIError class and common error handling patterns.
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class VLLMCLIError(Exception):
    """
    Base exception class for all vLLM CLI errors.

    Provides structured error handling with error codes, context information,
    and user-friendly messaging capabilities.

    Attributes:
        message: Technical error message for developers
        error_code: Unique error code for programmatic handling
        context: Additional context information as dictionary
        user_message: Optional user-friendly error message
    """

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ):
        """
        Initialize a vLLM CLI error.

        Args:
            message: Technical error message for developers
            error_code: Unique error code for categorization
            context: Additional context information
            user_message: Optional user-friendly message
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.user_message = user_message

        # Log the error when created
        logger.debug(f"VLLMCLIError created: {error_code} - {message}")

    def get_user_message(self) -> str:
        """
        Get a user-friendly error message.

        Returns the user_message if available, otherwise returns
        a cleaned-up version of the technical message.

        Returns:
            User-friendly error message string
        """
        if self.user_message:
            return self.user_message

        # Provide a generic user-friendly message based on error code
        return self._generate_user_message()

    def _generate_user_message(self) -> str:
        """Generate a user-friendly message based on error code."""
        code_messages = {
            "UNKNOWN_ERROR": "An unexpected error occurred.",
            "VALIDATION_ERROR": "Configuration validation failed.",
            "FILE_NOT_FOUND": "Required file was not found.",
            "PERMISSION_DENIED": "Permission denied - check file permissions.",
            "NETWORK_ERROR": "Network connection error occurred.",
            "TIMEOUT_ERROR": "Operation timed out.",
        }

        return code_messages.get(self.error_code, "An error occurred.")

    def add_context(self, key: str, value: Any) -> None:
        """Add additional context information."""
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context information by key."""
        return self.context.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.get_user_message(),
            "context": self.context,
        }

    def __str__(self) -> str:
        """String representation showing error code and message."""
        context_str = ""
        if self.context:
            context_items = [f"{k}={v}" for k, v in self.context.items()]
            context_str = f" ({', '.join(context_items)})"

        return f"[{self.error_code}] {self.message}{context_str}"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"{self.__class__.__name__}(message={self.message!r}, "
            f"error_code={self.error_code!r}, context={self.context!r})"
        )


class ValidationError(VLLMCLIError):
    """Base class for validation errors."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Any = None,
        **kwargs,
    ):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        if field_name:
            self.add_context("field_name", field_name)
        if field_value is not None:
            self.add_context("field_value", field_value)

    def _generate_user_message(self) -> str:
        field_name = self.get_context("field_name")
        if field_name:
            return f"Invalid value for '{field_name}'. {self.message}"
        return f"Validation error: {self.message}"
