#!/usr/bin/env python3
"""
Configuration management for the proxy server.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .models import ModelConfig, ProxyConfig

logger = logging.getLogger(__name__)


class ProxyConfigManager:
    """
    Manages proxy server configuration including persistence and validation.
    """

    def __init__(self):
        """Initialize the proxy configuration manager."""
        self.config_dir = Path.home() / ".config" / "vllm-cli"
        # Default configuration file path - used as fallback when no path specified
        # Also maintains backward compatibility with existing configurations
        self.proxy_config_file = self.config_dir / "proxy_config.yaml"
        self.proxy_configs_dir = self.config_dir / "proxy_configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.proxy_configs_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_path: Optional[Path] = None) -> ProxyConfig:
        """
        Load proxy configuration from file.

        Args:
            config_path: Path to configuration file (uses default if not provided)

        Returns:
            ProxyConfig instance
        """
        config_file = config_path or self.proxy_config_file

        if not config_file.exists():
            logger.info(f"No config file found at {config_file}, using defaults")
            return self.get_default_config()

        try:
            with open(config_file, "r") as f:
                if config_file.suffix in [".yaml", ".yml"]:
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)

            return self._parse_config(config_dict)

        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            return self.get_default_config()

    def save_config(self, config: ProxyConfig, config_path: Optional[Path] = None):
        """
        Save proxy configuration to file.

        Args:
            config: ProxyConfig to save
            config_path: Path to save to (uses default if not provided)
        """
        config_file = config_path or self.proxy_config_file

        try:
            config_dict = {
                "proxy": {
                    "host": config.host,
                    "port": config.port,
                    "enable_cors": config.enable_cors,
                    "enable_metrics": config.enable_metrics,
                    "log_requests": config.log_requests,
                },
                "models": [
                    {
                        "name": model.name,
                        "model_path": model.model_path,
                        "gpu_ids": model.gpu_ids,
                        "port": model.port,
                        "profile": model.profile,
                        "config_overrides": model.config_overrides,
                        "enabled": model.enabled,
                    }
                    for model in config.models
                ],
            }

            with open(config_file, "w") as f:
                if config_file.suffix in [".yaml", ".yml"]:
                    yaml.safe_dump(config_dict, f, default_flow_style=False)
                else:
                    json.dump(config_dict, f, indent=2)

            logger.info(f"Saved proxy configuration to {config_file}")

        except Exception as e:
            logger.error(f"Failed to save config to {config_file}: {e}")

    def save_named_config(self, config: ProxyConfig, name: str):
        """
        Save proxy configuration with a specific name.

        Args:
            config: ProxyConfig to save
            name: Name for the configuration
        """
        # Sanitize name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        config_file = self.proxy_configs_dir / f"{safe_name}.yaml"

        self.save_config(config, config_file)
        logger.info(f"Saved proxy configuration as '{name}'")

    def load_named_config(self, name: str) -> Optional[ProxyConfig]:
        """
        Load a named proxy configuration.

        Args:
            name: Name of the configuration to load

        Returns:
            ProxyConfig if found, None otherwise
        """
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        config_file = self.proxy_configs_dir / f"{safe_name}.yaml"

        if not config_file.exists():
            logger.error(f"Configuration '{name}' not found")
            return None

        return self.load_config(config_file)

    def list_saved_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        List all saved proxy configurations.

        Returns:
            Dictionary mapping config names to their summaries
        """
        configs = {}

        # Check new named configs directory
        for config_file in self.proxy_configs_dir.glob("*.yaml"):
            name = config_file.stem
            try:
                config = self.load_config(config_file)
                if config:
                    configs[name] = {
                        "file": str(config_file),
                        "models": len(config.models),
                        "port": config.port,
                        "model_names": [m.name for m in config.models][
                            :3
                        ],  # First 3 models
                    }
            except Exception as e:
                logger.warning(f"Failed to load config {name}: {e}")

        # Also check legacy default location
        if self.proxy_config_file.exists():
            try:
                config = self.load_config(self.proxy_config_file)
                if config:
                    configs["default"] = {
                        "file": str(self.proxy_config_file),
                        "models": len(config.models),
                        "port": config.port,
                        "model_names": [m.name for m in config.models][:3],
                    }
            except Exception:
                pass

        return configs

    def delete_named_config(self, name: str) -> bool:
        """
        Delete a named proxy configuration.

        Args:
            name: Name of the configuration to delete

        Returns:
            True if deleted successfully
        """
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        config_file = self.proxy_configs_dir / f"{safe_name}.yaml"

        if config_file.exists():
            try:
                config_file.unlink()
                logger.info(f"Deleted configuration '{name}'")
                return True
            except Exception as e:
                logger.error(f"Failed to delete configuration '{name}': {e}")
                return False
        else:
            logger.warning(f"Configuration '{name}' not found")
            return False

    def get_default_config(self) -> ProxyConfig:
        """
        Get default proxy configuration.

        Returns:
            Default ProxyConfig instance
        """
        return ProxyConfig(
            host="0.0.0.0",  # nosec B104
            port=8000,
            models=[],
            enable_cors=True,
            enable_metrics=True,
            log_requests=False,
        )

    def _parse_config(self, config_dict: Dict[str, Any]) -> ProxyConfig:
        """
        Parse configuration dictionary into ProxyConfig.

        Args:
            config_dict: Configuration dictionary

        Returns:
            ProxyConfig instance
        """
        proxy_settings = config_dict.get("proxy", {})
        models_list = config_dict.get("models", [])

        models = []
        for model_dict in models_list:
            try:
                models.append(ModelConfig(**model_dict))
            except Exception as e:
                logger.warning(f"Failed to parse model config: {e}")

        return ProxyConfig(
            host=proxy_settings.get("host", "0.0.0.0"),  # nosec B104
            port=proxy_settings.get("port", 8000),
            models=models,
            enable_cors=proxy_settings.get("enable_cors", True),
            enable_metrics=proxy_settings.get("enable_metrics", True),
            log_requests=proxy_settings.get("log_requests", False),
        )

    def validate_config(self, config: ProxyConfig) -> List[str]:
        """
        Validate proxy configuration.

        Args:
            config: ProxyConfig to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check proxy port
        if not 1 <= config.port <= 65535:
            errors.append(f"Invalid proxy port: {config.port}")

        # Check for port conflicts
        used_ports = {config.port}
        for model in config.models:
            if model.port in used_ports:
                errors.append(f"Port {model.port} is used by multiple services")
            used_ports.add(model.port)

        # Check model names are unique
        model_names = [m.name for m in config.models]
        if len(model_names) != len(set(model_names)):
            errors.append("Model names must be unique")

        return errors

    def create_example_config(self) -> ProxyConfig:
        """
        Create an example proxy configuration.

        Returns:
            Example ProxyConfig instance
        """
        return ProxyConfig(
            host="0.0.0.0",  # nosec B104
            port=8000,
            models=[
                ModelConfig(
                    name="llama-3-8b",
                    model_path="meta-llama/Meta-Llama-3-8B-Instruct",
                    gpu_ids=[0],
                    port=8001,
                    profile="standard",
                    enabled=True,
                ),
                ModelConfig(
                    name="mistral-7b",
                    model_path="mistralai/Mistral-7B-Instruct-v0.2",
                    gpu_ids=[1],
                    port=8002,
                    profile="performance",
                    enabled=True,
                ),
                ModelConfig(
                    name="gemma-2b",
                    model_path="google/gemma-2b",
                    gpu_ids=[2],
                    port=8003,
                    profile="memory-optimized",
                    enabled=False,  # Disabled by default
                ),
            ],
            enable_cors=True,
            enable_metrics=True,
            log_requests=False,
        )

    def export_config(self, config: ProxyConfig, export_path: Path):
        """
        Export configuration to a file.

        Args:
            config: ProxyConfig to export
            export_path: Path to export to
        """
        self.save_config(config, export_path)
        logger.info(f"Exported configuration to {export_path}")

    def import_config(self, import_path: Path) -> ProxyConfig:
        """
        Import configuration from a file.

        Args:
            import_path: Path to import from

        Returns:
            Imported ProxyConfig instance
        """
        config = self.load_config(import_path)
        logger.info(f"Imported configuration from {import_path}")
        return config
