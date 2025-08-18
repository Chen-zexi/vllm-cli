#!/usr/bin/env python3
"""
Validation registry for managing and executing validation rules.

Provides centralized management of validation rules and orchestrates
validation across multiple fields with dependency checking.
"""
import logging
from typing import Any, Dict, List, Optional

from .base import BaseValidator, CompositeValidator, ValidationResult

logger = logging.getLogger(__name__)


class ValidationRegistry:
    """
    Central registry for field validators.

    Manages validation rules for different fields and orchestrates
    validation across the entire configuration with dependency handling.
    """

    def __init__(self):
        self.validators: Dict[str, CompositeValidator] = {}

    def register(self, field_name: str, validator: BaseValidator) -> None:
        """
        Register a validator for a field.

        Args:
            field_name: Name of the field to validate
            validator: Validator instance to register
        """
        if field_name not in self.validators:
            self.validators[field_name] = CompositeValidator(field_name)

        self.validators[field_name].add_validator(validator)
        logger.debug(f"Registered validator for field: {field_name}")

    def get_validator(self, field_name: str) -> Optional[CompositeValidator]:
        """
        Get the composite validator for a field.

        Args:
            field_name: Name of the field

        Returns:
            CompositeValidator if registered, None otherwise
        """
        return self.validators.get(field_name)

    def validate_field(
        self, field_name: str, value: Any, context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate a single field.

        Args:
            field_name: Name of the field to validate
            value: Value to validate
            context: Full configuration context for dependency validation

        Returns:
            ValidationResult for the field
        """
        validator = self.validators.get(field_name)
        if not validator:
            # No validation rules for this field - consider it valid
            return ValidationResult()

        return validator.validate(value, context)

    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate an entire configuration.

        Validates all fields present in the config and checks dependencies.

        Args:
            config: Configuration dictionary to validate

        Returns:
            ValidationResult with all validation errors and warnings
        """
        result = ValidationResult()

        # Validate each field in the config
        for field_name, value in config.items():
            field_result = self.validate_field(field_name, value, config)
            result.merge(field_result)

        return result

    def get_registered_fields(self) -> List[str]:
        """
        Get list of all registered field names.

        Returns:
            List of field names that have validation rules
        """
        return list(self.validators.keys())

    def has_validator(self, field_name: str) -> bool:
        """
        Check if a field has validation rules.

        Args:
            field_name: Name of the field to check

        Returns:
            True if field has validators, False otherwise
        """
        return field_name in self.validators

    def remove_validator(self, field_name: str) -> bool:
        """
        Remove all validators for a field.

        Args:
            field_name: Name of the field to remove validators for

        Returns:
            True if validators were removed, False if field wasn't registered
        """
        if field_name in self.validators:
            del self.validators[field_name]
            logger.debug(f"Removed validators for field: {field_name}")
            return True
        return False

    def clear(self) -> None:
        """Remove all registered validators."""
        self.validators.clear()
        logger.debug("Cleared all validators from registry")

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary information about registered validators.

        Returns:
            Dictionary with validation registry statistics
        """
        summary = {
            "total_fields": len(self.validators),
            "fields": list(self.validators.keys()),
        }

        # Count validators by type
        validator_counts = {}
        for field_name, composite_validator in self.validators.items():
            for validator in composite_validator.validators:
                validator_type = type(validator).__name__
                validator_counts[validator_type] = (
                    validator_counts.get(validator_type, 0) + 1
                )

        summary["validator_counts"] = validator_counts
        return summary
