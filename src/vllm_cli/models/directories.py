#!/usr/bin/env python3
"""
Simplified directory utilities for vLLM CLI models.

Provides default search paths for model discovery when hf-model-tool
is not available. All directory management is delegated to hf-model-tool.
"""
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class DirectoryManager:
    """
    Simplified directory manager that provides default search paths.

    All actual directory management is delegated to hf-model-tool.
    This class only provides fallback search paths for model discovery.
    """

    def __init__(self):
        """Initialize with default paths only."""

    def get_model_directories(self) -> List[Path]:
        """
        Get default model directories for fallback searching.

        Returns:
            List of Path objects for default model directories
        """
        return self._get_default_directories()

    def get_lora_directories(self) -> List[Path]:
        """
        Get default LoRA directories for fallback searching.

        Returns:
            List of Path objects for default LoRA directories
        """
        # Use same paths as models for LoRA search
        return self._get_default_directories()

    def _get_default_directories(self) -> List[Path]:
        """
        Get default directories to search when hf-model-tool is not available.

        Returns:
            List of existing default directories
        """
        defaults = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "huggingface" / "transformers",
            Path("/data/models"),
            Path("/models"),
            Path.home() / "models",
        ]

        return [d for d in defaults if d.exists()]

    # Legacy methods kept for compatibility but simplified
    def list_directories(self) -> List:
        """Legacy method - returns empty list as management is in hf-model-tool."""
        return []

    def add_directory(self, directory: str, path_type: str = "auto") -> bool:
        """Legacy method - directory management is in hf-model-tool."""
        logger.info("Directory management has moved to hf-model-tool")
        return False

    def remove_directory(self, directory: str) -> bool:
        """Legacy method - directory management is in hf-model-tool."""
        logger.info("Directory management has moved to hf-model-tool")
        return False

    def validate_directory(self, directory: str) -> bool:
        """
        Check if a directory exists and might contain models.

        Args:
            directory: Path to check

        Returns:
            True if directory exists and is accessible
        """
        path = Path(directory)
        return path.exists() and path.is_dir()
