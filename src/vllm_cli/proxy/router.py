#!/usr/bin/env python3
"""
Request routing logic for the proxy server.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RequestRouter:
    """
    Routes incoming requests to appropriate vLLM backend servers based on model name.

    This router focuses solely on request routing. Model state and health
    are managed by the ModelRegistry.
    """

    def __init__(self):
        """Initialize the request router."""
        # Map model names to backend URLs
        self.backends: Dict[str, Dict[str, Any]] = {}

    def add_backend(self, model_name: str, backend_url: str, config: Dict[str, Any]):
        """
        Add a backend server for a model.

        Args:
            model_name: Name of the model (used in API requests)
            backend_url: Base URL of the vLLM server (e.g., http://localhost:8001)
            config: Additional configuration for the backend
        """
        self.backends[model_name] = {
            "url": backend_url,
            "port": config.get("port"),
            **config,
        }
        logger.info(f"Added backend route for model '{model_name}' at {backend_url}")

    def remove_backend(self, model_name: str):
        """
        Remove a backend server route.

        Args:
            model_name: Name of the model to remove

        Raises:
            KeyError: If model doesn't exist
        """
        if model_name not in self.backends:
            raise KeyError(f"Model '{model_name}' not found in backends")

        del self.backends[model_name]
        logger.info(f"Removed backend route for model '{model_name}'")

    def route_request(self, model_name: str) -> Optional[str]:
        """
        Route a request to the appropriate backend.

        Args:
            model_name: Name of the model requested

        Returns:
            Backend URL if available, None otherwise
        """
        # Direct lookup - registry handles actual names
        if model_name in self.backends:
            return self.backends[model_name]["url"]

        # Check for wildcard/default backend
        if "*" in self.backends:
            return self.backends["*"]["url"]

        logger.warning(f"No backend route found for model '{model_name}'")
        return None

    def get_active_models(self) -> List[str]:
        """
        Get list of registered model names.

        Returns:
            List of model names with routes
        """
        return list(self.backends.keys())

    def get_backends(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all backend configurations.

        Returns:
            Dictionary of backend configurations
        """
        return self.backends.copy()
