#!/usr/bin/env python3
"""
Validation Framework Package

This package provides a comprehensive validation framework for vLLM CLI
with type-specific validators, composition patterns, and dependency checking.

Main Components:
- Base validators: Abstract classes and validation result containers
- Type validators: Specific validators for integers, floats, strings, booleans, choices
- Registry: Central management of validation rules
- Factory functions: Convenient creation of common validator patterns

The validation framework follows the Strategy pattern to allow easy extension
and composition of validation rules.
"""

# Base validation classes
from .base import (
    ValidationError,
    ValidationResult,
    BaseValidator,
    DependencyValidator,
    CompositeValidator,
)

# Type-specific validators
from .types import (
    IntegerValidator,
    FloatValidator,
    StringValidator,
    BooleanValidator,
    ChoiceValidator,
)

# Validation registry
from .registry import ValidationRegistry

# Factory functions for common patterns
from .factory import (
    create_integer_validator,
    create_float_validator,
    create_string_validator,
    create_boolean_validator,
    create_choice_validator,
    validate_positive_integer,
    validate_non_negative_integer,
    validate_probability,
    validate_port_number,
    validate_percentage,
    validate_file_path,
    validate_directory_path,
    validate_url,
    validate_email,
    create_dependent_validator,
)

# Schema building functions
from .schema import (
    create_vllm_validation_registry,
    create_compatibility_validator,
    CompatibilityValidator,
    load_validation_schema_from_file,
)

__all__ = [
    # Base classes
    "ValidationError",
    "ValidationResult",
    "BaseValidator",
    "DependencyValidator",
    "CompositeValidator",
    # Type validators
    "IntegerValidator",
    "FloatValidator",
    "StringValidator",
    "BooleanValidator",
    "ChoiceValidator",
    # Registry
    "ValidationRegistry",
    # Factory functions
    "create_integer_validator",
    "create_float_validator",
    "create_string_validator",
    "create_boolean_validator",
    "create_choice_validator",
    "validate_positive_integer",
    "validate_non_negative_integer",
    "validate_probability",
    "validate_port_number",
    "validate_percentage",
    "validate_file_path",
    "validate_directory_path",
    "validate_url",
    "validate_email",
    "create_dependent_validator",
    # Schema functions
    "create_vllm_validation_registry",
    "create_compatibility_validator",
    "CompatibilityValidator",
    "load_validation_schema_from_file",
]
