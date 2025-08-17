#!/usr/bin/env python3
"""
Configuration-related exception classes.

Defines exceptions for configuration operations including validation,
profile management, and file operations.
"""
from typing import Any, List, Optional

from .base import ValidationError, VLLMCLIError


class ConfigurationError(VLLMCLIError):
    """Base class for configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        config_section: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        if config_file:
            self.add_context("config_file", config_file)
        if config_section:
            self.add_context("config_section", config_section)

    def _generate_user_message(self) -> str:
        return "Configuration error. Check your settings and try again."


class ConfigValidationError(ConfigurationError, ValidationError):
    """Configuration validation failed."""

    def __init__(
        self, message: str, validation_errors: Optional[List[str]] = None, **kwargs
    ):
        super().__init__(message, error_code="CONFIG_VALIDATION_ERROR", **kwargs)
        if validation_errors:
            self.add_context("validation_errors", validation_errors)

    def _generate_user_message(self) -> str:
        errors = self.get_context("validation_errors", [])
        if errors:
            return f"Configuration validation failed: {'; '.join(errors[:3])}"
        return "Configuration validation failed. Check your settings."


class ConfigFileError(ConfigurationError):
    """Error reading or writing configuration files."""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIG_FILE_ERROR", **kwargs)
        if operation:
            self.add_context("operation", operation)

    def _generate_user_message(self) -> str:
        operation = self.get_context("operation", "File operation")
        return f"{operation} failed. Check file permissions and disk space."


class ConfigParseError(ConfigurationError):
    """Error parsing configuration file format."""

    def __init__(
        self,
        message: str,
        file_format: Optional[str] = None,
        line_number: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="CONFIG_PARSE_ERROR", **kwargs)
        if file_format:
            self.add_context("file_format", file_format)
        if line_number:
            self.add_context("line_number", line_number)

    def _generate_user_message(self) -> str:
        file_format = self.get_context("file_format", "configuration file")
        line_number = self.get_context("line_number")
        if line_number:
            return f"Invalid {file_format} format at line {line_number}. Check syntax."
        return f"Invalid {file_format} format. Check syntax."


class ProfileError(ConfigurationError):
    """Profile-related operation error."""

    def __init__(
        self,
        message: str,
        profile_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="PROFILE_ERROR", **kwargs)
        if profile_name:
            self.add_context("profile_name", profile_name)
        if operation:
            self.add_context("operation", operation)

    def _generate_user_message(self) -> str:
        profile_name = self.get_context("profile_name", "Profile")
        operation = self.get_context("operation", "operation")
        return f"{profile_name} {operation} failed. Check profile configuration."


class ProfileNotFoundError(ProfileError):
    """Profile not found."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="PROFILE_NOT_FOUND", **kwargs)

    def _generate_user_message(self) -> str:
        profile_name = self.get_context("profile_name", "Profile")
        return f"{profile_name} not found. Use 'vllm-cli profiles list' to see available profiles."


class ProfileExistsError(ProfileError):
    """Profile already exists."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="PROFILE_EXISTS", **kwargs)

    def _generate_user_message(self) -> str:
        profile_name = self.get_context("profile_name", "Profile")
        return f"{profile_name} already exists. Use a different name or update existing profile."


class SchemaError(ConfigurationError):
    """Schema definition or validation error."""

    def __init__(
        self,
        message: str,
        schema_type: Optional[str] = None,
        schema_field: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="SCHEMA_ERROR", **kwargs)
        if schema_type:
            self.add_context("schema_type", schema_type)
        if schema_field:
            self.add_context("schema_field", schema_field)

    def _generate_user_message(self) -> str:
        return "Schema validation error. Configuration format may be incorrect."


class ArgumentError(ConfigurationError):
    """Invalid command line argument."""

    def __init__(
        self,
        message: str,
        argument_name: Optional[str] = None,
        argument_value: Any = None,
        **kwargs,
    ):
        super().__init__(message, error_code="ARGUMENT_ERROR", **kwargs)
        if argument_name:
            self.add_context("argument_name", argument_name)
        if argument_value is not None:
            self.add_context("argument_value", argument_value)

    def _generate_user_message(self) -> str:
        arg_name = self.get_context("argument_name")
        if arg_name:
            return f"Invalid argument '{arg_name}'. Check usage with --help."
        return "Invalid command line argument. Check usage with --help."


class CompatibilityError(ConfigurationError):
    """Configuration compatibility issue."""

    def __init__(
        self,
        message: str,
        incompatible_options: Optional[List[str]] = None,
        severity: str = "error",
        **kwargs,
    ):
        super().__init__(message, error_code="COMPATIBILITY_ERROR", **kwargs)
        if incompatible_options:
            self.add_context("incompatible_options", incompatible_options)
        self.add_context("severity", severity)

    def _generate_user_message(self) -> str:
        options = self.get_context("incompatible_options", [])
        if len(options) >= 2:
            return f"Incompatible options: {', '.join(options)}. Check configuration."
        return "Configuration options are incompatible. Review your settings."


class DefaultsError(ConfigurationError):
    """Error with default configuration values."""

    def __init__(self, message: str, defaults_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DEFAULTS_ERROR", **kwargs)
        if defaults_type:
            self.add_context("defaults_type", defaults_type)

    def _generate_user_message(self) -> str:
        defaults_type = self.get_context("defaults_type", "default")
        return f"Error loading {defaults_type} settings. Reset to system defaults."
