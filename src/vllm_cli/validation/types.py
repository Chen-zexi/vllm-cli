#!/usr/bin/env python3
"""
Type-specific validator implementations.

Provides validators for common data types including integers, floats,
strings, booleans, and choice selections with customizable constraints.
"""
import logging
import re
from typing import Any, List, Optional, Pattern, Union

from .base import BaseValidator, ValidationResult

logger = logging.getLogger(__name__)


class IntegerValidator(BaseValidator):
    """
    Validates integer values with optional range constraints.

    Supports minimum/maximum bounds and provides clear error messages
    for out-of-range values and type mismatches.
    """

    def __init__(
        self,
        field_name: str,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(field_name, **kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any) -> ValidationResult:
        """Validate integer value and range constraints."""
        result = ValidationResult()

        if value is None:
            if self.required:
                result.add_error(
                    self._create_error(value, "is required", "REQUIRED_ERROR")
                )
            return result

        # Type checking with conversion attempt
        if isinstance(value, bool):
            # Booleans are technically int subclass in Python, but reject them
            result.add_error(
                self._create_error(
                    value, "must be an integer, not boolean", "TYPE_ERROR"
                )
            )
            return result

        if not isinstance(value, int):
            # Try to convert string to int
            if isinstance(value, str) and value.strip().isdigit():
                try:
                    value = int(value.strip())
                except ValueError:
                    result.add_error(
                        self._create_error(value, "must be an integer", "TYPE_ERROR")
                    )
                    return result
            else:
                result.add_error(
                    self._create_error(value, "must be an integer", "TYPE_ERROR")
                )
                return result

        # Range validation
        if self.min_value is not None and value < self.min_value:
            result.add_error(
                self._create_error(value, f"must be >= {self.min_value}", "RANGE_ERROR")
            )

        if self.max_value is not None and value > self.max_value:
            result.add_error(
                self._create_error(value, f"must be <= {self.max_value}", "RANGE_ERROR")
            )

        return result


class FloatValidator(BaseValidator):
    """
    Validates floating-point values with optional range constraints.

    Handles both float and int inputs, with customizable precision
    and range validation.
    """

    def __init__(
        self,
        field_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(field_name, **kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any) -> ValidationResult:
        """Validate float value and range constraints."""
        result = ValidationResult()

        if value is None:
            if self.required:
                result.add_error(
                    self._create_error(value, "is required", "REQUIRED_ERROR")
                )
            return result

        # Type checking with conversion
        if isinstance(value, bool):
            result.add_error(
                self._create_error(value, "must be a number, not boolean", "TYPE_ERROR")
            )
            return result

        if not isinstance(value, (int, float)):
            # Try to convert string to float
            if isinstance(value, str):
                try:
                    value = float(value.strip())
                except ValueError:
                    result.add_error(
                        self._create_error(value, "must be a number", "TYPE_ERROR")
                    )
                    return result
            else:
                result.add_error(
                    self._create_error(value, "must be a number", "TYPE_ERROR")
                )
                return result

        # Convert to float for consistent comparison
        float_value = float(value)

        # Range validation
        if self.min_value is not None and float_value < self.min_value:
            result.add_error(
                self._create_error(value, f"must be >= {self.min_value}", "RANGE_ERROR")
            )

        if self.max_value is not None and float_value > self.max_value:
            result.add_error(
                self._create_error(value, f"must be <= {self.max_value}", "RANGE_ERROR")
            )

        return result


class StringValidator(BaseValidator):
    """
    Validates string values with length and pattern constraints.

    Supports minimum/maximum length validation and regex pattern matching
    for complex string validation requirements.
    """

    def __init__(
        self,
        field_name: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[Union[str, Pattern]] = None,
        **kwargs,
    ):
        super().__init__(field_name, **kwargs)
        self.min_length = min_length
        self.max_length = max_length

        # Compile pattern if provided as string
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern

    def validate(self, value: Any) -> ValidationResult:
        """Validate string value, length, and pattern constraints."""
        result = ValidationResult()

        if value is None:
            if self.required:
                result.add_error(
                    self._create_error(value, "is required", "REQUIRED_ERROR")
                )
            return result

        # Convert to string if not already
        if not isinstance(value, str):
            value = str(value)

        # Length validation
        if self.min_length is not None and len(value) < self.min_length:
            result.add_error(
                self._create_error(
                    value,
                    f"must be at least {self.min_length} characters long",
                    "LENGTH_ERROR",
                )
            )

        if self.max_length is not None and len(value) > self.max_length:
            result.add_error(
                self._create_error(
                    value,
                    f"must be at most {self.max_length} characters long",
                    "LENGTH_ERROR",
                )
            )

        # Pattern validation
        if self.pattern and not self.pattern.match(value):
            result.add_error(
                self._create_error(
                    value, "does not match required pattern", "PATTERN_ERROR"
                )
            )

        return result


class BooleanValidator(BaseValidator):
    """
    Validates boolean values with flexible input acceptance.

    Accepts actual booleans, common string representations,
    and numeric equivalents (0/1).
    """

    def validate(self, value: Any) -> ValidationResult:
        """Validate boolean value with flexible input handling."""
        result = ValidationResult()

        if value is None:
            if self.required:
                result.add_error(
                    self._create_error(value, "is required", "REQUIRED_ERROR")
                )
            return result

        # Accept actual booleans
        if isinstance(value, bool):
            return result

        # Accept string representations
        if isinstance(value, str):
            lower_value = value.lower().strip()
            if lower_value in ("true", "yes", "1", "on", "enabled"):
                return result
            elif lower_value in ("false", "no", "0", "of", "disabled"):
                return result
            else:
                result.add_error(
                    self._create_error(
                        value,
                        "must be true/false, yes/no, 1/0, or on/of",
                        "TYPE_ERROR",
                    )
                )
                return result

        # Accept numeric 0/1
        if isinstance(value, int) and value in (0, 1):
            return result

        result.add_error(
            self._create_error(value, "must be a boolean value", "TYPE_ERROR")
        )
        return result


class ChoiceValidator(BaseValidator):
    """
    Validates that a value is one of a predefined set of choices.

    Supports case-insensitive matching and provides helpful error
    messages listing valid options.
    """

    def __init__(
        self, field_name: str, choices: List[Any], case_sensitive: bool = True, **kwargs
    ):
        super().__init__(field_name, **kwargs)
        self.choices = choices
        self.case_sensitive = case_sensitive

        # Create normalized choices for case-insensitive comparison
        if not case_sensitive:
            self.normalized_choices = [
                str(choice).lower() if choice is not None else None
                for choice in choices
            ]

    def validate(self, value: Any) -> ValidationResult:
        """Validate that value is in the allowed choices."""
        result = ValidationResult()

        if value is None:
            if self.required:
                result.add_error(
                    self._create_error(value, "is required", "REQUIRED_ERROR")
                )
            return result

        # Check if value is in choices
        if self.case_sensitive:
            if value in self.choices:
                return result
        else:
            # Case-insensitive comparison
            normalized_value = str(value).lower()
            if normalized_value in self.normalized_choices:
                return result

        # Format choices for error message
        choices_str = ", ".join(str(choice) for choice in self.choices)
        result.add_error(
            self._create_error(value, f"must be one of: {choices_str}", "CHOICE_ERROR")
        )

        return result
