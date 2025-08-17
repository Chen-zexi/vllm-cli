"""Simplified validation tests to ensure basic compatibility."""

from vllm_cli.validation.base import ValidationError, ValidationResult
from vllm_cli.validation.registry import ValidationRegistry
from vllm_cli.validation.types import (
    BooleanValidator,
    ChoiceValidator,
    FloatValidator,
    IntegerValidator,
    StringValidator,
)


class TestValidationBasics:
    """Basic validation tests."""

    def test_validation_result(self):
        """Test ValidationResult basics."""
        result = ValidationResult()
        assert result.is_valid() is True

        # Add error
        result.add_error(ValidationError("field", "value", "error"))
        assert result.is_valid() is False

    def test_validators_exist(self):
        """Test that validator classes exist."""
        assert IntegerValidator is not None
        assert FloatValidator is not None
        assert StringValidator is not None
        assert BooleanValidator is not None
        assert ChoiceValidator is not None

    def test_validation_registry_exists(self):
        """Test that ValidationRegistry exists."""
        assert ValidationRegistry is not None
        registry = ValidationRegistry()
        assert registry is not None
