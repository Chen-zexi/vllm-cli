#!/usr/bin/env python3
"""
Base validator classes and validation result containers.

Defines the core validation infrastructure including base classes,
result containers, and common validation patterns.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """
    Base exception for validation errors.

    Attributes:
        field: The field name that failed validation
        value: The value that failed validation
        message: Human-readable error message
        code: Error code for programmatic handling
    """

    def __init__(
        self, field: str, value: Any, message: str, code: str = "VALIDATION_ERROR"
    ):
        self.field = field
        self.value = value
        self.message = message
        self.code = code
        super().__init__(f"{field}: {message}")


class ValidationResult:
    """
    Container for validation results.

    Allows collecting multiple validation errors and warnings
    while maintaining performance for success cases.
    """

    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[str] = []

    def add_error(self, error: ValidationError) -> None:
        """Add a validation error."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)

    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    def get_error_messages(self) -> List[str]:
        """Get list of error messages."""
        return [str(error) for error in self.errors]

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


class BaseValidator(ABC):
    """
    Abstract base class for all validators.

    Validators follow the Strategy pattern to allow easy extension
    and composition of validation rules.
    """

    def __init__(self, field_name: str, required: bool = False):
        self.field_name = field_name
        self.required = required

    @abstractmethod
    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a value and return validation result.

        Args:
            value: Value to validate

        Returns:
            ValidationResult containing any errors or warnings
        """

    def _create_error(
        self, value: Any, message: str, code: str = "VALIDATION_ERROR"
    ) -> ValidationError:
        """Helper to create validation errors."""
        return ValidationError(self.field_name, value, message, code)


class DependencyValidator(BaseValidator):
    """
    Validates that required dependencies are present when this field is set.

    This validator checks that certain fields are present in the configuration
    when this field is being validated. Useful for conditional requirements.
    """

    def __init__(self, field_name: str, required_fields: List[str]):
        super().__init__(field_name)
        self.required_fields = required_fields

    def validate(self, value: Any) -> ValidationResult:
        """
        Validate dependencies. Note: This requires the full config context.

        This validator needs access to the full configuration to check
        for dependent fields. It should be used within a CompositeValidator
        that provides the necessary context.
        """
        result = ValidationResult()

        # This validator requires context from the registry
        # Individual validation is not meaningful
        # The actual dependency checking happens in ValidationRegistry

        return result


class CompositeValidator:
    """
    Combines multiple validators for a single field.

    Allows building complex validation rules by composing simpler validators.
    Validates using all contained validators and collects all results.
    """

    def __init__(self, field_name: str):
        self.field_name = field_name
        self.validators: List[BaseValidator] = []

    def add_validator(self, validator: BaseValidator) -> None:
        """Add a validator to the composition."""
        self.validators.append(validator)

    def validate(
        self, value: Any, context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate using all contained validators.

        Args:
            value: Value to validate
            context: Optional context (full config) for dependency validation

        Returns:
            ValidationResult with combined results from all validators
        """
        result = ValidationResult()

        for validator in self.validators:
            if isinstance(validator, DependencyValidator) and context:
                # Special handling for dependency validation
                dep_result = self._validate_dependencies(validator, value, context)
                result.merge(dep_result)
            else:
                validator_result = validator.validate(value)
                result.merge(validator_result)

        return result

    def _validate_dependencies(
        self, dep_validator: DependencyValidator, value: Any, context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate field dependencies."""
        result = ValidationResult()

        if value is not None:  # Only check dependencies if field is set
            for required_field in dep_validator.required_fields:
                if required_field not in context or context[required_field] is None:
                    error = dep_validator._create_error(
                        value,
                        f"requires {required_field} to be set",
                        "DEPENDENCY_ERROR",
                    )
                    result.add_error(error)

        return result
