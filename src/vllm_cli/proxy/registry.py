#!/usr/bin/env python3
"""
Dynamic runtime registry for managing models in the proxy server.

This module provides in-memory tracking of models and their states
without GPU utilization tracking.
"""
import logging
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelState(Enum):
    """Model state enumeration."""

    RUNNING = "running"
    SLEEPING = "sleeping"
    STOPPED = "stopped"
    STARTING = "starting"


class RegistrationStatus(Enum):
    """Model registration status."""

    PENDING = "pending"  # Pre-registered, waiting for model to be ready
    AVAILABLE = "available"  # Model is ready and serving
    ERROR = "error"  # Registration failed or model is down


class ModelEntry:
    """Represents a single model in the registry."""

    def __init__(
        self, port: int, gpu_ids: List[int] = None, config_name: Optional[str] = None
    ):
        self.port = port
        self.gpu_ids = gpu_ids or []
        self.actual_name: Optional[str] = None  # From /v1/models
        self.config_name: Optional[str] = config_name  # Original config name
        self.state: ModelState = ModelState.STARTING
        self.status: RegistrationStatus = RegistrationStatus.PENDING
        self.sleep_level: int = 0  # 0=awake, 1=CPU offload, 2=discarded
        self.last_activity: datetime = datetime.now()
        self.base_url: str = f"http://localhost:{port}"
        self.verification_attempts: int = 0
        self.error_message: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Get the best available name for display."""
        return self.actual_name or self.config_name or f"port_{self.port}"

    def update_state(self, state: ModelState):
        """Update model state and activity timestamp."""
        self.state = state
        self.last_activity = datetime.now()

    def mark_verified(self, actual_name: str = None):
        """Mark model as successfully verified."""
        self.status = RegistrationStatus.AVAILABLE
        if actual_name:
            self.actual_name = actual_name
        self.error_message = None
        self.last_activity = datetime.now()

    def mark_error(self, error_msg: str = None):
        """Mark model as having an error."""
        self.status = RegistrationStatus.ERROR
        self.error_message = error_msg
        self.last_activity = datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "port": self.port,
            "name": self.display_name,
            "actual_name": self.actual_name,
            "config_name": self.config_name,
            "gpu_ids": self.gpu_ids,
            "state": self.state.value,
            "registration_status": self.status.value,  # UI expects this key
            "sleep_level": self.sleep_level,
            "base_url": self.base_url,
            "last_activity": self.last_activity.isoformat(),
            "error_message": self.error_message,
        }


class ModelRegistry:
    """
    Runtime registry for managing model instances.

    This registry tracks models and their states without
    managing GPU utilization.
    """

    def __init__(self):
        """Initialize the model registry."""
        self.models: Dict[int, ModelEntry] = {}  # port -> ModelEntry
        self.lock = threading.RLock()
        self._shutdown = False

    def shutdown(self):
        """Shutdown the registry and clean up."""
        with self.lock:
            self._shutdown = True
            self.models.clear()
            logger.info("Model registry shutdown complete")

    def pre_register(
        self, port: int, gpu_ids: List[int] = None, config_name: Optional[str] = None
    ) -> bool:
        """
        Pre-register a model before it's ready.

        Args:
            port: Port number the model will run on
            gpu_ids: List of GPU IDs assigned to the model
            config_name: Configuration name for the model

        Returns:
            True if pre-registration successful
        """
        with self.lock:
            if port in self.models:
                logger.warning(f"Model already registered on port {port}")
                return False

            entry = ModelEntry(port, gpu_ids, config_name)
            self.models[port] = entry

            logger.info(
                f"Pre-registered model '{config_name}' on port {port} "
                f"with GPUs {gpu_ids}"
            )
            return True

    def verify_and_activate(self, port: int, actual_name: str = None) -> bool:
        """
        Verify a pre-registered model and activate it.

        Args:
            port: Port number of the model
            actual_name: Actual model name from the server

        Returns:
            True if verification successful
        """
        with self.lock:
            if port not in self.models:
                # Model wasn't pre-registered, register it now
                entry = ModelEntry(port)
                self.models[port] = entry
                logger.info(f"Registered new model on port {port}")
            else:
                entry = self.models[port]

            entry.mark_verified(actual_name)

            # Only update state to RUNNING if the model is in a transitional state
            # This preserves SLEEPING state during refresh while allowing
            # STARTING → RUNNING transition during initial startup
            if entry.state in [ModelState.STARTING, ModelState.STOPPED]:
                entry.update_state(ModelState.RUNNING)

            logger.info(f"Model '{entry.display_name}' on port {port} is now AVAILABLE")
            return True

    def mark_model_error(self, port: int, error_msg: str = None):
        """Mark a model as having an error."""
        with self.lock:
            if port in self.models:
                self.models[port].mark_error(error_msg)
                logger.warning(f"Model on port {port} marked as ERROR: {error_msg}")

    def remove_model(self, port: int) -> bool:
        """
        Remove a model from the registry.

        Args:
            port: Port number of the model to remove

        Returns:
            True if model was removed
        """
        with self.lock:
            if port in self.models:
                entry = self.models[port]
                del self.models[port]
                logger.info(f"Removed model '{entry.display_name}' from port {port}")
                return True
            return False

    def update_model_state(self, port: int, state: ModelState) -> bool:
        """
        Update the state of a model.

        Args:
            port: Port number of the model
            state: New state for the model

        Returns:
            True if state was updated
        """
        with self.lock:
            if port in self.models:
                self.models[port].update_state(state)
                return True
            return False

    def get_model(self, port: int) -> Optional[ModelEntry]:
        """Get a model entry by port."""
        with self.lock:
            return self.models.get(port)

    def get_all_models(self) -> Dict[int, ModelEntry]:
        """Get all registered models."""
        with self.lock:
            return self.models.copy()

    def get_available_models(self) -> List[ModelEntry]:
        """Get all available (ready) models."""
        with self.lock:
            return [
                entry
                for entry in self.models.values()
                if entry.status == RegistrationStatus.AVAILABLE
            ]

    def get_models_on_gpu(self, gpu_id: int) -> List[ModelEntry]:
        """Get all models assigned to a specific GPU."""
        with self.lock:
            return [entry for entry in self.models.values() if gpu_id in entry.gpu_ids]

    def get_status_summary(self) -> Dict:
        """Get a summary of registry status."""
        with self.lock:
            available_models = [
                e
                for e in self.models.values()
                if e.status == RegistrationStatus.AVAILABLE
            ]
            pending_models = [
                e
                for e in self.models.values()
                if e.status == RegistrationStatus.PENDING
            ]
            error_models = [
                e for e in self.models.values() if e.status == RegistrationStatus.ERROR
            ]

            # Group models by GPU
            gpu_usage = {}
            for entry in self.models.values():
                for gpu_id in entry.gpu_ids:
                    if gpu_id not in gpu_usage:
                        gpu_usage[gpu_id] = []
                    gpu_usage[gpu_id].append(entry.display_name)

            return {
                "total_models": len(self.models),
                "available": len(available_models),
                "pending": len(pending_models),
                "errors": len(error_models),
                "models": [e.to_dict() for e in self.models.values()],
                "gpu_usage": gpu_usage,
            }

    def cleanup_stale_entries(self, timeout_seconds: int = 300):
        """
        Remove stale pending entries that haven't been verified.

        Args:
            timeout_seconds: Time after which pending entries are considered stale
        """
        with self.lock:
            current_time = datetime.now()
            stale_ports = []

            for port, entry in self.models.items():
                if entry.status == RegistrationStatus.PENDING:
                    age = (current_time - entry.last_activity).total_seconds()
                    if age > timeout_seconds:
                        stale_ports.append(port)

            for port in stale_ports:
                entry = self.models[port]
                logger.warning(
                    f"Removing stale pending entry for '{entry.display_name}' "
                    f"on port {port}"
                )
                del self.models[port]

            return len(stale_ports)
