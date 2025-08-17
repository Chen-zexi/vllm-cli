#!/usr/bin/env python3
"""
Model management module for vLLM CLI.

Integrates with hf-model-tool to list and manage downloaded models
with caching for performance optimization.
"""
import logging
from typing import Any, Dict, List, Optional

from .cache import ModelCache
from .discovery import build_model_dict, scan_for_models

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model discovery, caching, and metadata operations.

    Provides a high-level interface for model operations including
    listing, searching, and retrieving model information with
    integrated caching for performance.
    """

    def __init__(self):
        self.cache = ModelCache()

    def list_available_models(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """
        List all available downloaded models with caching.

        Model scanning can be expensive, so results are cached for a short time
        to improve performance when multiple UI components need model data.

        Args:
            refresh: Force refresh the model cache, ignoring TTL

        Returns:
            List of model dictionaries with keys:
            - name: Full model name (publisher/model)
            - size: Model size in bytes
            - path: Path to model directory
            - type: Model type (model, custom_model)
            - publisher: Model publisher/organization
            - display_name: Human-readable model name
            - metadata: Additional model metadata
        """
        # Check cache first (unless refresh is forced)
        if not refresh:
            cached_models = self.cache.get_cached_models()
            if cached_models is not None:
                return cached_models

        # Fetch fresh model data
        models = scan_for_models()

        # Process and normalize model data
        processed_models = []
        for item in models:
            if item.get("type") in ["model", "custom_model"]:
                model_dict = build_model_dict(item)
                processed_models.append(model_dict)

        # Sort models by name
        processed_models.sort(key=lambda x: x["name"])

        # Update cache
        self.cache.cache_models(processed_models)

        logger.info(f"Found {len(processed_models)} models")
        return processed_models

    def get_model_details(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Name of the model to get details for

        Returns:
            Dictionary with model details or None if not found
        """
        # First, try to find from cached models
        models = self.list_available_models()
        model_info = None

        for model in models:
            if model["name"] == model_name or model.get("display_name") == model_name:
                model_info = model
                break

        if not model_info:
            logger.warning(f"Model {model_name} not found")
            return None

        # Build detailed information
        details = {
            "name": model_info["name"],
            "full_name": model_info["name"],
            "path": model_info.get("path", ""),
            "size": model_info.get("size", 0),
            "type": model_info.get("type", "model"),
            "publisher": model_info.get("publisher", "unknown"),
            "display_name": model_info.get("display_name", model_name),
        }

        # Extract metadata if available
        metadata = model_info.get("metadata", {})
        if metadata:
            details["architecture"] = (
                metadata.get("architectures", ["unknown"])[0]
                if metadata.get("architectures")
                else "unknown"
            )
            details["model_type"] = metadata.get("model_type", "unknown")
            details["torch_dtype"] = metadata.get("torch_dtype", "unknown")
            details["vocab_size"] = metadata.get("vocab_size", "unknown")

            # Add to main details
            details["metadata"] = metadata

        # Try to read config.json for more details if path exists
        if details["path"]:
            from .metadata import extract_model_config

            config_data = extract_model_config(details["path"])
            if config_data:
                details.update(config_data)

        return details

    def search_models(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for models matching a query.

        Args:
            query: Search query string

        Returns:
            List of matching models
        """
        models = self.list_available_models()
        query_lower = query.lower()

        # Filter models matching the query
        matches = []
        for model in models:
            if (
                query_lower in model["name"].lower()
                or query_lower in model.get("display_name", "").lower()
                or query_lower in model.get("publisher", "").lower()
            ):
                matches.append(model)

        return matches

    def get_model_count(self) -> int:
        """
        Get the total number of available models.

        Returns:
            Number of available models
        """
        return len(self.list_available_models())

    def get_models_by_publisher(self, publisher: str) -> List[Dict[str, Any]]:
        """
        Get all models from a specific publisher.

        Args:
            publisher: Publisher/organization name

        Returns:
            List of models from the publisher
        """
        models = self.list_available_models()
        return [
            model
            for model in models
            if model.get("publisher", "").lower() == publisher.lower()
        ]

    def get_models_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """
        Get all models of a specific type.

        Args:
            model_type: Type of model (e.g., 'model', 'custom_model')

        Returns:
            List of models of the specified type
        """
        models = self.list_available_models()
        return [model for model in models if model.get("type") == model_type]

    def refresh_cache(self) -> None:
        """Force refresh of the model cache."""
        self.cache.clear_cache()
        self.list_available_models(refresh=True)
        logger.info("Model cache refreshed")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_cache_stats()


# Global model manager instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# Convenience functions for backward compatibility
def list_available_models(refresh: bool = False) -> List[Dict[str, Any]]:
    """List all available models (convenience function)."""
    return get_model_manager().list_available_models(refresh)


def get_model_details(model_name: str) -> Optional[Dict[str, Any]]:
    """Get model details (convenience function)."""
    return get_model_manager().get_model_details(model_name)


def search_models(query: str) -> List[Dict[str, Any]]:
    """Search for models (convenience function)."""
    return get_model_manager().search_models(query)
