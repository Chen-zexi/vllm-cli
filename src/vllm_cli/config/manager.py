#!/usr/bin/env python3
"""
Configuration manager for vLLM CLI.

Handles configuration file management and validation using
the new validation framework.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from copy import deepcopy

import yaml

from ..validation import ValidationRegistry
from ..validation import create_vllm_validation_registry, create_compatibility_validator
from ..errors import ConfigurationError, ValidationError

from .profiles import ProfileManager
from .schemas import SchemaManager
from .persistence import PersistenceManager

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Enhanced configuration manager with comprehensive validation support.

    This class manages configuration files, profiles, and validation using
    the new validation framework. It provides a clean API for configuration
    operations while ensuring data integrity through comprehensive validation.

    Attributes:
        config_dir: Directory for configuration files
        validation_registry: Registry of validation rules
        compatibility_validator: Validator for parameter combinations
    """

    def __init__(self):
        # Paths for configuration files
        self.config_dir = Path.home() / ".config" / "vllm-cli"
        self.config_file = self.config_dir / "config.yaml"

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-managers
        self.schema_manager = SchemaManager()
        self.profile_manager = ProfileManager(self.config_dir)
        self.persistence_manager = PersistenceManager()

        # Initialize validation system
        self.validation_registry = create_vllm_validation_registry()
        self.compatibility_validator = create_compatibility_validator(
            self.validation_registry
        )

        # Load main configuration
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load main configuration file (legacy YAML support)."""
        return self.persistence_manager.load_yaml_file(self.config_file)

    def _save_config(self) -> None:
        """Save main configuration file."""
        self.persistence_manager.save_yaml_file(self.config_file, self.config)

    def get_argument_info(self, arg_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific argument."""
        return self.schema_manager.get_argument_info(arg_name)

    def get_arguments_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all arguments belonging to a specific category."""
        return self.schema_manager.get_arguments_by_category(category)

    def get_category_info(self, category: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific category."""
        return self.schema_manager.get_category_info(category)

    def get_ordered_categories(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get categories ordered by their display order."""
        return self.schema_manager.get_ordered_categories()

    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all profiles (default and user-created)."""
        return self.profile_manager.get_all_profiles()

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific profile by name with dynamic defaults applied."""
        profile = self.profile_manager.get_profile(name)
        if profile and "config" in profile:
            # Apply dynamic defaults to the profile configuration
            profile["config"] = self.profile_manager.apply_dynamic_defaults(
                profile["config"]
            )
        return profile

    def save_user_profile(self, name: str, profile: Dict[str, Any]) -> bool:
        """Save or update a user profile with validation."""
        # Validate profile configuration if present
        if "config" in profile:
            is_valid, errors = self.validate_config(profile["config"])
            if not is_valid:
                from ..errors import ProfileError

                raise ProfileError(
                    f"Profile configuration validation failed: {'; '.join(errors)}",
                    profile_name=name,
                    error_code="PROFILE_CONFIG_INVALID",
                )

        return self.profile_manager.save_user_profile(name, profile)

    def delete_user_profile(self, name: str) -> bool:
        """Delete a user profile."""
        return self.profile_manager.delete_user_profile(name)

    @property
    def default_profiles(self) -> Dict[str, Any]:
        """Get default profiles."""
        return self.profile_manager.default_profiles

    @property
    def user_profiles(self) -> Dict[str, Any]:
        """Get user profiles."""
        return self.profile_manager.user_profiles

    def import_profile(self, filepath: Path, new_name: Optional[str] = None) -> bool:
        """Import a profile from file."""
        return self.profile_manager.import_profile(filepath, new_name)

    def export_profile(self, profile_name: str, filepath: Path) -> bool:
        """Export a profile to file."""
        return self.profile_manager.export_profile(profile_name, filepath)

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a configuration using the new validation framework.

        This method uses the comprehensive validation system to check
        configuration values, types, ranges, and dependencies.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_error_messages)

        Raises:
            ConfigurationError: If validation system fails
        """
        try:
            # Use new validation system
            result = self.validation_registry.validate_config(config)

            # Convert validation errors to string messages for backward compatibility
            error_messages = result.get_error_messages()

            # Add warnings if any
            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"Configuration warning: {warning}")

            return result.is_valid(), error_messages

        except Exception as e:
            logger.error(f"Error during configuration validation: {e}")
            raise ConfigurationError(
                f"Validation system error: {e}", error_code="VALIDATION_SYSTEM_ERROR"
            ) from e

    def validate_argument_combination(
        self, config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that argument combinations are compatible using the new compatibility validator.

        Args:
            config: Configuration dictionary to check for compatibility

        Returns:
            Tuple of (is_valid, list_of_warnings_or_errors)
        """
        try:
            issues = self.compatibility_validator.validate_compatibility(config)

            warnings = []
            errors = []

            for issue in issues:
                message = issue["message"]
                if issue["severity"] == "error":
                    errors.append(message)
                else:
                    warnings.append(message)

            # Log warnings
            for warning in warnings:
                logger.warning(f"Configuration compatibility warning: {warning}")

            # Return all issues as warnings for backward compatibility
            # (errors are treated as warnings in the legacy interface)
            all_issues = errors + warnings

            return len(errors) == 0, all_issues

        except Exception as e:
            logger.error(f"Error during compatibility validation: {e}")
            return False, [f"Compatibility validation error: {e}"]

    def build_cli_args(self, config: Dict[str, Any]) -> List[str]:
        """
        Build command-line arguments from a configuration dictionary.

        Only includes arguments that are explicitly set in the config.
        """
        args = []

        # Special handling for model (positional argument)
        if "model" in config:
            args.extend(["serve", config["model"]])

        for arg_name, value in config.items():
            # Skip special keys and None values
            if arg_name in ["model", "name", "description", "icon"] or value is None:
                continue

            arg_info = self.schema_manager.get_argument_info(arg_name)
            if not arg_info:
                logger.warning(f"Unknown argument in config: {arg_name}")
                continue

            cli_flag = arg_info.get("cli_flag")
            if not cli_flag:
                continue

            arg_type = arg_info.get("type")

            if arg_type == "boolean":
                if value:  # Only add flag if True
                    args.append(cli_flag)
            elif arg_type in ["integer", "float", "string", "choice"]:
                args.extend([cli_flag, str(value)])
            else:
                logger.warning(f"Unknown argument type for {arg_name}: {arg_type}")

        return args

    def export_profile(self, profile_name: str, filepath: Path) -> bool:
        """Export a profile to a JSON file."""
        return self.profile_manager.export_profile(profile_name, filepath)

    def import_profile(self, filepath: Path, new_name: Optional[str] = None) -> bool:
        """Import a profile from a JSON file with comprehensive validation."""
        return self.profile_manager.import_profile(
            filepath, new_name, self.validate_config
        )

    def get_last_config(self) -> Optional[Dict[str, Any]]:
        """Get the last used server configuration."""
        return self.config.get("last_config")

    def save_last_config(self, config: Dict[str, Any]) -> None:
        """Save the last used server configuration."""
        self.config["last_config"] = config
        self._save_config()

    def get_recent_models(self, limit: int = 5) -> List[str]:
        """Get recently used models."""
        return self.config.get("recent_models", [])[:limit]

    def add_recent_model(self, model: str) -> None:
        """Add a model to recent models list."""
        recent = self.config.get("recent_models", [])
        if model in recent:
            recent.remove(model)
        recent.insert(0, model)
        self.config["recent_models"] = recent[:10]  # Keep only last 10
        self._save_config()

    def clear_cache(self) -> None:
        """Clear cached data."""
        cache_file = self.config_dir / "cache.json"
        if cache_file.exists():
            cache_file.unlink()
        self.config.pop("last_config", None)
        self.config.pop("recent_models", None)
        self._save_config()
        logger.info("Cache cleared")

    def get_server_defaults(self) -> Dict[str, Any]:
        """Get default server settings."""
        return self.config.get(
            "server_defaults",
            {
                "default_port": 8000,
                "auto_restart": False,
                "log_level": "info",
            },
        )

    def save_server_defaults(self, defaults: Dict[str, Any]) -> None:
        """Save default server settings."""
        self.config["server_defaults"] = defaults
        self._save_config()

    def get_ui_preferences(self) -> Dict[str, Any]:
        """Get UI preferences including progress bar style and log display settings."""
        return self.config.get(
            "ui_preferences",
            {
                "progress_bar_style": "blocks",  # Default style
                "show_gpu_in_monitor": True,
                "auto_refresh_interval": 1.0,
                "log_lines_startup": 50,  # Number of log lines to show during startup
                "log_lines_monitor": 50,  # Number of log lines to show in server monitor
                "startup_refresh_rate": 4.0,  # Hz for startup log refresh
                "monitor_refresh_rate": 1.0,  # Hz for monitor log refresh
            },
        )

    def save_ui_preferences(self, preferences: Dict[str, Any]) -> None:
        """Save UI preferences."""
        self.config["ui_preferences"] = preferences
        self._save_config()
