#!/usr/bin/env python3
"""
Model-related exception classes.

Defines exceptions for model operations including loading,
validation, and discovery errors.
"""
from typing import Optional

from .base import VLLMCLIError


class ModelError(VLLMCLIError):
    """Base class for model-related errors."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="MODEL_ERROR", **kwargs)
        if model_name:
            self.add_context("model_name", model_name)
        if model_path:
            self.add_context("model_path", model_path)

    def _generate_user_message(self) -> str:
        return "Model operation failed. Check model availability and permissions."


class ModelNotFoundError(ModelError):
    """Model not found in expected locations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="MODEL_NOT_FOUND", **kwargs)

    def _generate_user_message(self) -> str:
        model_name = self.get_context("model_name", "Model")
        return f"{model_name} not found. Use 'vllm-cli models' to see available models."


class ModelLoadError(ModelError):
    """Error loading model files or metadata."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="MODEL_LOAD_ERROR", **kwargs)

    def _generate_user_message(self) -> str:
        return "Failed to load model. Check model files and format compatibility."


class ModelValidationError(ModelError):
    """Model failed validation checks."""

    def __init__(self, message: str, validation_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="MODEL_VALIDATION_ERROR", **kwargs)
        if validation_type:
            self.add_context("validation_type", validation_type)

    def _generate_user_message(self) -> str:
        return "Model validation failed. Model may be corrupted or incompatible."


class ModelFormatError(ModelError):
    """Unsupported or invalid model format."""

    def __init__(
        self,
        message: str,
        format_type: Optional[str] = None,
        supported_formats: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="MODEL_FORMAT_ERROR", **kwargs)
        if format_type:
            self.add_context("format_type", format_type)
        if supported_formats:
            self.add_context("supported_formats", supported_formats)

    def _generate_user_message(self) -> str:
        format_type = self.get_context("format_type")
        if format_type:
            return f"Unsupported model format: {format_type}. Convert to a supported format."
        return "Unsupported model format. Check format compatibility."


class ModelSizeError(ModelError):
    """Model size exceeds available resources."""

    def __init__(
        self,
        message: str,
        model_size: Optional[str] = None,
        available_memory: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="MODEL_SIZE_ERROR", **kwargs)
        if model_size:
            self.add_context("model_size", model_size)
        if available_memory:
            self.add_context("available_memory", available_memory)

    def _generate_user_message(self) -> str:
        return "Model is too large for available resources. Use quantization or upgrade hardware."


class ModelPermissionError(ModelError):
    """Insufficient permissions to access model files."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="MODEL_PERMISSION_ERROR", **kwargs)

    def _generate_user_message(self) -> str:
        return "Permission denied accessing model files. Check file permissions."


class ModelDownloadError(ModelError):
    """Error downloading model from repository."""

    def __init__(
        self,
        message: str,
        repository: Optional[str] = None,
        download_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="MODEL_DOWNLOAD_ERROR", **kwargs)
        if repository:
            self.add_context("repository", repository)
        if download_url:
            self.add_context("download_url", download_url)

    def _generate_user_message(self) -> str:
        return "Failed to download model. Check network connection and repository availability."


class ModelCompatibilityError(ModelError):
    """Model incompatible with current vLLM version."""

    def __init__(
        self,
        message: str,
        model_version: Optional[str] = None,
        vllm_version: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="MODEL_COMPATIBILITY_ERROR", **kwargs)
        if model_version:
            self.add_context("model_version", model_version)
        if vllm_version:
            self.add_context("vllm_version", vllm_version)

    def _generate_user_message(self) -> str:
        return "Model incompatible with current vLLM version. Update vLLM or use compatible model."


class ModelCacheError(ModelError):
    """Error with model caching operations."""

    def __init__(
        self,
        message: str,
        cache_path: Optional[str] = None,
        cache_operation: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="MODEL_CACHE_ERROR", **kwargs)
        if cache_path:
            self.add_context("cache_path", cache_path)
        if cache_operation:
            self.add_context("cache_operation", cache_operation)

    def _generate_user_message(self) -> str:
        return "Model cache error. Clear cache and try again."


class ModelRequirementsError(ModelError):
    """Model requirements not met (dependencies, hardware, etc.)."""

    def __init__(
        self,
        message: str,
        requirement_type: Optional[str] = None,
        missing_requirement: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="MODEL_REQUIREMENTS_ERROR", **kwargs)
        if requirement_type:
            self.add_context("requirement_type", requirement_type)
        if missing_requirement:
            self.add_context("missing_requirement", missing_requirement)

    def _generate_user_message(self) -> str:
        requirement_type = self.get_context("requirement_type", "requirement")
        missing = self.get_context("missing_requirement")
        if missing:
            return (
                f"Missing {requirement_type}: {missing}. Install required dependencies."
            )
        return f"Model {requirement_type} not met. Check system requirements."
