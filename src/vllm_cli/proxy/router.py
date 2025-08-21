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
    """

    def __init__(self):
        """Initialize the request router."""
        # Map model names to backend configurations
        self.backends: Dict[str, Dict[str, Any]] = {}
        # Track backend health status
        self.backend_health: Dict[str, bool] = {}

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
            "gpu_ids": config.get("gpu_ids", []),
            "model_path": config.get("model_path"),
            **config,
        }
        self.backend_health[model_name] = True
        logger.info(f"Added backend for model '{model_name}' at {backend_url}")

    def remove_backend(self, model_name: str):
        """
        Remove a backend server.

        Args:
            model_name: Name of the model to remove

        Raises:
            KeyError: If model doesn't exist
        """
        if model_name not in self.backends:
            raise KeyError(f"Model '{model_name}' not found in backends")

        del self.backends[model_name]
        del self.backend_health[model_name]
        logger.info(f"Removed backend for model '{model_name}'")

    def route_request(self, model_name: str) -> Optional[str]:
        """
        Route a request to the appropriate backend.

        Args:
            model_name: Name of the model requested

        Returns:
            Backend URL if available, None otherwise
        """
        # Check for exact match first
        if model_name in self.backends:
            if self.backend_health.get(model_name, False):
                return self.backends[model_name]["url"]
            else:
                logger.warning(f"Backend for model '{model_name}' is unhealthy")
                return None

        # Check for partial matches or aliases
        for backend_model, backend_config in self.backends.items():
            # Check if model_name is an alias
            aliases = backend_config.get("aliases", [])
            if model_name in aliases:
                if self.backend_health.get(backend_model, False):
                    return backend_config["url"]

        # Check for wildcard/default backend
        if "*" in self.backends and self.backend_health.get("*", False):
            return self.backends["*"]["url"]

        logger.warning(f"No backend found for model '{model_name}'")
        return None

    def get_active_models(self) -> List[str]:
        """
        Get list of active model names.

        Returns:
            List of model names that have healthy backends
        """
        return [model for model, healthy in self.backend_health.items() if healthy]

    def get_backends(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all backend configurations.

        Returns:
            Dictionary of backend configurations
        """
        return self.backends.copy()

    def mark_backend_health(self, model_name: str, healthy: bool):
        """
        Update health status of a backend.

        Args:
            model_name: Name of the model
            healthy: Health status
        """
        if model_name in self.backend_health:
            self.backend_health[model_name] = healthy
            status = "healthy" if healthy else "unhealthy"
            logger.info(f"Marked backend for '{model_name}' as {status}")

    def get_backend_for_gpu(self, gpu_id: int) -> Optional[str]:
        """
        Find which model is using a specific GPU.

        Args:
            gpu_id: GPU device ID

        Returns:
            Model name if found, None otherwise
        """
        for model_name, config in self.backends.items():
            gpu_ids = config.get("gpu_ids", [])
            if gpu_id in gpu_ids:
                return model_name
        return None

    def load_balancing_route(self, model_name: str) -> Optional[str]:
        """
        Route with load balancing if multiple backends serve the same model.

        This is for future enhancement when we support multiple instances
        of the same model for load balancing.

        Args:
            model_name: Name of the model

        Returns:
            Backend URL if available, None otherwise
        """
        # For now, just use simple routing
        # In future, this could implement round-robin, least-connections, etc.
        return self.route_request(model_name)
