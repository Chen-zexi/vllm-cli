#!/usr/bin/env python3
"""
Validator factory functions for common validation patterns.

Provides convenient factory functions to create commonly used
validator combinations and pre-configured validators.
"""
import logging
from typing import Any, List, Optional

from .base import CompositeValidator, DependencyValidator
from .types import (
    BooleanValidator,
    ChoiceValidator,
    FloatValidator,
    IntegerValidator,
    StringValidator,
)

logger = logging.getLogger(__name__)


def create_integer_validator(
    field_name: str,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    required: bool = False,
) -> CompositeValidator:
    """
    Create a composite integer validator.

    Args:
        field_name: Name of the field
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        required: Whether the field is required

    Returns:
        CompositeValidator with integer validation
    """
    composite = CompositeValidator(field_name)
    validator = IntegerValidator(
        field_name, min_value=min_value, max_value=max_value, required=required
    )
    composite.add_validator(validator)
    return composite


def create_float_validator(
    field_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    required: bool = False,
) -> CompositeValidator:
    """
    Create a composite float validator.

    Args:
        field_name: Name of the field
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        required: Whether the field is required

    Returns:
        CompositeValidator with float validation
    """
    composite = CompositeValidator(field_name)
    validator = FloatValidator(
        field_name, min_value=min_value, max_value=max_value, required=required
    )
    composite.add_validator(validator)
    return composite


def create_string_validator(
    field_name: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    required: bool = False,
) -> CompositeValidator:
    """
    Create a composite string validator.

    Args:
        field_name: Name of the field
        min_length: Minimum string length
        max_length: Maximum string length
        pattern: Regex pattern to match
        required: Whether the field is required

    Returns:
        CompositeValidator with string validation
    """
    composite = CompositeValidator(field_name)
    validator = StringValidator(
        field_name,
        min_length=min_length,
        max_length=max_length,
        pattern=pattern,
        required=required,
    )
    composite.add_validator(validator)
    return composite


def create_boolean_validator(
    field_name: str, required: bool = False
) -> CompositeValidator:
    """
    Create a composite boolean validator.

    Args:
        field_name: Name of the field
        required: Whether the field is required

    Returns:
        CompositeValidator with boolean validation
    """
    composite = CompositeValidator(field_name)
    validator = BooleanValidator(field_name, required=required)
    composite.add_validator(validator)
    return composite


def create_choice_validator(
    field_name: str,
    choices: List[Any],
    case_sensitive: bool = True,
    required: bool = False,
) -> CompositeValidator:
    """
    Create a composite choice validator.

    Args:
        field_name: Name of the field
        choices: List of valid choices
        case_sensitive: Whether string comparison is case sensitive
        required: Whether the field is required

    Returns:
        CompositeValidator with choice validation
    """
    composite = CompositeValidator(field_name)
    validator = ChoiceValidator(
        field_name, choices=choices, case_sensitive=case_sensitive, required=required
    )
    composite.add_validator(validator)
    return composite


# Specialized validators for common patterns


def validate_positive_integer(field_name: str) -> CompositeValidator:
    """
    Create validator for positive integers (> 0).

    Args:
        field_name: Name of the field

    Returns:
        CompositeValidator that ensures value is a positive integer
    """
    return create_integer_validator(field_name, min_value=1)


def validate_non_negative_integer(field_name: str) -> CompositeValidator:
    """
    Create validator for non-negative integers (>= 0).

    Args:
        field_name: Name of the field

    Returns:
        CompositeValidator that ensures value is non-negative
    """
    return create_integer_validator(field_name, min_value=0)


def validate_probability(field_name: str) -> CompositeValidator:
    """
    Create validator for probability values (0.0 to 1.0).

    Args:
        field_name: Name of the field

    Returns:
        CompositeValidator that ensures value is a valid probability
    """
    return create_float_validator(field_name, min_value=0.0, max_value=1.0)


def validate_port_number(field_name: str) -> CompositeValidator:
    """
    Create validator for network port numbers (1-65535).

    Args:
        field_name: Name of the field

    Returns:
        CompositeValidator that ensures value is a valid port number
    """
    return create_integer_validator(field_name, min_value=1, max_value=65535)


def validate_percentage(field_name: str) -> CompositeValidator:
    """
    Create validator for percentage values (0-100).

    Args:
        field_name: Name of the field

    Returns:
        CompositeValidator that ensures value is a valid percentage
    """
    return create_float_validator(field_name, min_value=0.0, max_value=100.0)


def validate_file_path(field_name: str, must_exist: bool = False) -> CompositeValidator:
    """
    Create validator for file paths.

    Args:
        field_name: Name of the field
        must_exist: Whether the file must exist (not implemented yet)

    Returns:
        CompositeValidator for file path validation
    """
    # Basic string validation for now
    # Could be extended to check file existence, permissions, etc.
    return create_string_validator(field_name, min_length=1)


def validate_directory_path(
    field_name: str, must_exist: bool = False
) -> CompositeValidator:
    """
    Create validator for directory paths.

    Args:
        field_name: Name of the field
        must_exist: Whether the directory must exist (not implemented yet)

    Returns:
        CompositeValidator for directory path validation
    """
    # Basic string validation for now
    # Could be extended to check directory existence, permissions, etc.
    return create_string_validator(field_name, min_length=1)


def validate_url(field_name: str) -> CompositeValidator:
    """
    Create validator for URL format.

    Args:
        field_name: Name of the field

    Returns:
        CompositeValidator for URL validation
    """
    # Basic URL pattern - could be more sophisticated
    url_pattern = r"^https?:\/\/.+"
    return create_string_validator(field_name, pattern=url_pattern)


def validate_email(field_name: str) -> CompositeValidator:
    """
    Create validator for email format.

    Args:
        field_name: Name of the field

    Returns:
        CompositeValidator for email validation
    """
    # Basic email pattern - RFC compliant regex would be much more complex
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return create_string_validator(field_name, pattern=email_pattern)


def create_dependent_validator(
    field_name: str, base_validator: CompositeValidator, dependencies: List[str]
) -> CompositeValidator:
    """
    Create a validator that requires other fields to be present.

    Args:
        field_name: Name of the field
        base_validator: Base validator to use for the field
        dependencies: List of field names that must be present

    Returns:
        CompositeValidator with dependency checking
    """
    # Add dependency validator to the base validator
    dep_validator = DependencyValidator(field_name, dependencies)
    base_validator.add_validator(dep_validator)
    return base_validator
