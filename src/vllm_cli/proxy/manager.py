#!/usr/bin/env python3
"""
Manager for coordinating multiple vLLM server instances and the proxy server.
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import ConfigManager
from ..server import VLLMServer
from ..server.utils import get_next_available_port
from .models import ModelConfig, ProxyConfig
from .server_process import ProxyServerProcess

logger = logging.getLogger(__name__)


class ProxyManager:
    """
    Manages the lifecycle of multiple vLLM servers and the proxy server.
    """

    def __init__(self, config: Optional[ProxyConfig] = None):
        """
        Initialize the proxy manager.

        Args:
            config: Proxy configuration (uses defaults if not provided)
        """
        self.proxy_config = config or ProxyConfig()
        self.proxy_process: Optional[ProxyServerProcess] = None
        self.vllm_servers: Dict[str, VLLMServer] = {}
        self.config_manager = ConfigManager()

        # Compatibility properties for monitoring code that may still reference old structure

    def start_proxy(self) -> bool:
        """
        Start the proxy server.

        Returns:
            True if proxy started successfully
        """
        try:
            # Create proxy server process instance
            self.proxy_process = ProxyServerProcess(self.proxy_config)

            # Start proxy as a subprocess
            if not self.proxy_process.start():
                logger.error("Failed to start proxy server process")
                return False

            # Give it a moment to fully initialize
            time.sleep(2)

            # Register all running models with the proxy
            # Note: We'll need to update this to work with subprocess
            # For now, models will register when they start

            logger.info(
                f"Proxy server started on "
                f"{self.proxy_config.host}:{self.proxy_config.port}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start proxy server: {e}")
            return False

    def stop_proxy(self):
        """Stop the proxy server and all vLLM instances."""
        logger.info("Stopping proxy server and all model servers...")

        # Stop all vLLM servers
        for model_name in list(self.vllm_servers.keys()):
            self.stop_model(model_name)

        # Stop proxy server process
        if self.proxy_process:
            self.proxy_process.stop()
            self.proxy_process = None
            logger.info("Proxy server stopped")

    def start_model(self, model_config: ModelConfig) -> bool:
        """
        Start a vLLM server for a specific model.

        Args:
            model_config: Configuration for the model

        Returns:
            True if server started successfully
        """
        try:
            # Check if model is already running
            if model_config.name in self.vllm_servers:
                logger.warning(f"Model '{model_config.name}' is already running")
                return False

            # Build vLLM server configuration
            vllm_config = self._build_vllm_config(model_config)

            # Create and start vLLM server
            server = VLLMServer(vllm_config)
            if not server.start():
                logger.error(f"Failed to start vLLM server for '{model_config.name}'")
                return False

            # Store server reference
            self.vllm_servers[model_config.name] = server

            # Register with proxy if it's running
            if self.proxy_server:
                backend_url = f"http://localhost:{model_config.port}"
                # Register with the actual model name
                self.proxy_server.router.add_backend(
                    model_config.name, backend_url, model_config.dict()
                )

                # Also register any aliases
                aliases = model_config.config_overrides.get("aliases", [])
                for alias in aliases:
                    self.proxy_server.router.add_backend(
                        alias, backend_url, model_config.dict()
                    )

            logger.info(
                f"Started vLLM server for '{model_config.name}' "
                f"on port {model_config.port} using GPUs {model_config.gpu_ids}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start model '{model_config.name}': {e}")
            return False

    def stop_model(self, model_name: str) -> bool:
        """
        Stop a vLLM server for a specific model.

        Args:
            model_name: Name of the model to stop

        Returns:
            True if server stopped successfully
        """
        if model_name not in self.vllm_servers:
            logger.warning(f"Model '{model_name}' is not running")
            return False

        try:
            # Stop the vLLM server
            server = self.vllm_servers[model_name]
            server.stop()

            # Remove from tracking
            del self.vllm_servers[model_name]

            # Remove from proxy if it's running
            if self.proxy_server:
                self.proxy_server.router.remove_backend(model_name)

            logger.info(f"Stopped vLLM server for '{model_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to stop model '{model_name}': {e}")
            return False

    def start_all_models(self) -> int:
        """
        Start all models defined in the proxy configuration.

        Returns:
            Number of models successfully started
        """
        started = 0
        for model_config in self.proxy_config.models:
            if model_config.enabled and self.start_model(model_config):
                started += 1
                # Add a small delay between starts to avoid resource conflicts
                time.sleep(2)
        return started

    def _build_vllm_config(self, model_config: ModelConfig) -> Dict[str, Any]:
        """
        Build vLLM server configuration from model configuration.

        Args:
            model_config: Model configuration

        Returns:
            vLLM server configuration dictionary
        """
        # Start with profile configuration if specified
        config = {}
        if model_config.profile:
            profile = self.config_manager.get_profile(model_config.profile)
            if profile:
                config = profile.get("config", {}).copy()

        # Set model and port
        config["model"] = model_config.model_path
        config["port"] = model_config.port

        # Handle GPU assignment
        if model_config.gpu_ids:
            # Set CUDA_VISIBLE_DEVICES via device field
            config["device"] = ",".join(str(gpu) for gpu in model_config.gpu_ids)

            num_gpus = len(model_config.gpu_ids)

            # For single GPU assignment, override any parallel settings from profile
            if num_gpus == 1:
                # Remove parallel configuration that conflicts with single GPU
                conflicting_settings = [
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "distributed_executor_backend",
                ]

                removed_settings = []
                for setting in conflicting_settings:
                    if setting in config:
                        removed_settings.append(f"{setting}={config[setting]}")
                        del config[setting]

                # Disable expert parallelism for single GPU
                if config.get("enable_expert_parallel"):
                    removed_settings.append("enable_expert_parallel=True")
                    config["enable_expert_parallel"] = False

                if removed_settings:
                    logger.warning(
                        f"Model '{model_config.name}' assigned single GPU. "
                        f"Overriding profile settings: {', '.join(removed_settings)}"
                    )

            elif num_gpus > 1:
                # For multi-GPU, set tensor_parallel_size if not already set
                if "tensor_parallel_size" not in config:
                    config["tensor_parallel_size"] = num_gpus
                elif config["tensor_parallel_size"] > num_gpus:
                    # Adjust if profile expects more GPUs than assigned
                    logger.warning(
                        f"Model '{model_config.name}': Adjusting tensor_parallel_size "
                        f"from {config['tensor_parallel_size']} to {num_gpus} "
                        f"(assigned GPUs)"
                    )
                    config["tensor_parallel_size"] = num_gpus

        # Apply any config overrides
        config.update(model_config.config_overrides)

        return config

    def get_status(self) -> Dict[str, Any]:
        """
        Get status of proxy and all model servers.

        Returns:
            Status dictionary
        """
        status = {
            "proxy_running": self.proxy_server is not None,
            "proxy_host": self.proxy_config.host,
            "proxy_port": self.proxy_config.port,
            "models": [],
        }

        for model_name, server in self.vllm_servers.items():
            model_status = {
                "name": model_name,
                "running": server.is_running(),
                "port": server.port,
                "uptime": None,
            }

            if server.is_running() and server.start_time:
                uptime = time.time() - server.start_time.timestamp()
                model_status["uptime"] = uptime

            status["models"].append(model_status)

        return status

    def reload_model(self, model_name: str) -> bool:
        """
        Reload a model (stop and start again).

        Args:
            model_name: Name of the model to reload

        Returns:
            True if reload successful
        """
        # Find the model config
        model_config = None
        for config in self.proxy_config.models:
            if config.name == model_name:
                model_config = config
                break

        if not model_config:
            logger.error(f"Model '{model_name}' not found in configuration")
            return False

        # Stop if running
        if model_name in self.vllm_servers:
            self.stop_model(model_name)
            time.sleep(2)  # Wait before restarting

        # Start again
        return self.start_model(model_config)

    def allocate_gpus_automatically(self) -> List[ModelConfig]:
        """
        Automatically allocate GPUs to models based on available resources.

        Returns:
            List of model configurations with GPU allocations
        """
        from ..system import get_gpu_info

        # Get available GPUs
        gpu_info = get_gpu_info()
        if not gpu_info:
            logger.warning("No GPUs available for allocation")
            return []

        num_gpus = len(gpu_info)
        models = self.proxy_config.models

        # Simple allocation strategy: distribute GPUs evenly
        allocated_configs = []

        if len(models) <= num_gpus:
            # Each model gets at least one GPU
            gpus_per_model = num_gpus // len(models)
            remaining_gpus = num_gpus % len(models)

            gpu_index = 0
            for i, model in enumerate(models):
                num_gpus_for_model = gpus_per_model
                if i < remaining_gpus:
                    num_gpus_for_model += 1

                model.gpu_ids = list(range(gpu_index, gpu_index + num_gpus_for_model))
                gpu_index += num_gpus_for_model

                # Allocate port if not specified
                if not model.port:
                    model.port = get_next_available_port(8001 + i)

                allocated_configs.append(model)
        else:
            # More models than GPUs - some models won't be allocated
            logger.warning(
                f"More models ({len(models)}) than GPUs ({num_gpus}). "
                f"Only first {num_gpus} models will be allocated."
            )
            for i in range(num_gpus):
                models[i].gpu_ids = [i]
                if not models[i].port:
                    models[i].port = get_next_available_port(8001 + i)
                allocated_configs.append(models[i])

        return allocated_configs

    def _register_running_models(self):
        """
        Register all running models with the proxy router.

        This method is called after the proxy server starts to ensure
        all previously started models are registered with the router.
        """
        if not self.proxy_server:
            return

        for model_name, server in self.vllm_servers.items():
            if server.is_running():
                # Find the model config for this model
                model_config = self._get_model_config_by_name(model_name)
                if model_config:
                    backend_url = f"http://localhost:{model_config.port}"
                    # Register the main model name
                    self.proxy_server.router.add_backend(
                        model_name, backend_url, model_config.dict()
                    )
                    logger.info(f"Registered model '{model_name}' with proxy router")

                    # Also register any aliases
                    aliases = model_config.config_overrides.get("aliases", [])
                    for alias in aliases:
                        self.proxy_server.router.add_backend(
                            alias, backend_url, model_config.dict()
                        )
                        logger.info(
                            f"Registered alias '{alias}' for model '{model_name}'"
                        )

    def _get_model_config_by_name(self, model_name: str) -> Optional[ModelConfig]:
        """
        Get model configuration by model name.

        Args:
            model_name: Name of the model

        Returns:
            ModelConfig if found, None otherwise
        """
        for model_config in self.proxy_config.models:
            if model_config.name == model_name:
                return model_config
        return None

    @classmethod
    def from_config_file(cls, config_path: Path) -> "ProxyManager":
        """
        Create ProxyManager from a configuration file.

        Args:
            config_path: Path to configuration file (YAML or JSON)

        Returns:
            ProxyManager instance
        """
        import json

        import yaml

        with open(config_path, "r") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)

        # Parse proxy configuration
        proxy_config = ProxyConfig(**config_dict.get("proxy", {}))

        # Parse model configurations
        models = []
        for model_dict in config_dict.get("models", []):
            models.append(ModelConfig(**model_dict))
        proxy_config.models = models

        return cls(proxy_config)

    @property
    def proxy_server(self):
        """Backward compatibility property for code that references proxy_server."""
        # Return a mock object with the attributes that monitoring code expects
        if self.proxy_process and self.proxy_process.is_running():
            manager = self

            class ProxyServerCompat:
                def __init__(self, process):
                    self.process = process
                    self.start_time = process.start_time
                    self.total_requests = 0
                    self.model_requests = {}

                @property
                def router(self):
                    # Create a mock router with backends from vllm_servers
                    class RouterCompat:
                        @property
                        def backends(self):
                            result = {}
                            for model_name, server in manager.vllm_servers.items():
                                result[model_name] = {
                                    "url": f"http://localhost:{server.port}",
                                    "port": server.port,
                                    "model_path": model_name,
                                }
                            return result

                    return RouterCompat()

                def get_recent_logs(self, n=50):
                    return self.process.get_recent_logs(n)

            return ProxyServerCompat(self.proxy_process)
        return None

    @property
    def proxy_thread(self):
        """Backward compatibility for proxy_thread checks."""
        if self.proxy_process:
            process = self.proxy_process

            class ThreadCompat:
                def is_alive(self):
                    return process.is_running()

            return ThreadCompat() if self.proxy_process.is_running() else None
        return None
