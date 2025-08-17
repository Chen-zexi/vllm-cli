#!/usr/bin/env python3
"""
Profile management for vLLM CLI configurations.

Handles user profile CRUD operations including validation,
import/export, and persistence.
"""
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..errors import ProfileError
from ..system.gpu import get_gpu_info

logger = logging.getLogger(__name__)


class ProfileManager:
    """
    Manages user profiles for vLLM configurations.

    Handles creation, retrieval, updating, and deletion of user profiles
    with validation and persistence.
    """

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.user_profiles_file = config_dir / "user_profiles.json"

        # Paths for schema files (packaged with application)
        schema_dir = Path(__file__).parent.parent / "schemas"
        self.default_profiles_file = schema_dir / "default_profiles.json"

        # Load profiles
        self.default_profiles = self._load_default_profiles()
        self.user_profiles = self._load_user_profiles()

    def _load_json_file(self, filepath: Path) -> Dict[str, Any]:
        """Load a JSON file safely."""
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        return {}

    def _save_json_file(self, filepath: Path, data: Dict[str, Any]) -> bool:
        """Save data to a JSON file."""
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")
            return False

    def _load_default_profiles(self) -> Dict[str, Any]:
        """Load default built-in profiles."""
        profiles = self._load_json_file(self.default_profiles_file)
        return profiles.get("profiles", {})

    def _load_user_profiles(self) -> Dict[str, Any]:
        """Load user-created custom profiles."""
        return self._load_json_file(self.user_profiles_file)

    def _save_user_profiles(self) -> bool:
        """Save user profiles to file."""
        return self._save_json_file(self.user_profiles_file, self.user_profiles)

    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all profiles (default and user-created)."""
        all_profiles = deepcopy(self.default_profiles)
        all_profiles.update(self.user_profiles)
        return all_profiles

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific profile by name."""
        # Check user profiles first, then default profiles
        if name in self.user_profiles:
            return deepcopy(self.user_profiles[name])
        elif name in self.default_profiles:
            return deepcopy(self.default_profiles[name])
        return None

    def save_user_profile(self, name: str, profile: Dict[str, Any]) -> bool:
        """
        Save or update a user profile with validation.

        Args:
            name: Profile name
            profile: Profile data to save

        Returns:
            True if saved successfully

        Raises:
            ProfileError: If profile validation fails or save fails
        """
        try:
            # Validate profile structure
            if not isinstance(profile, dict):
                raise ProfileError(
                    "Profile must be a dictionary",
                    profile_name=name,
                    error_code="INVALID_PROFILE_TYPE",
                )

            # Process LoRA configuration if present
            if "config" in profile:
                config = profile["config"]

                # Handle LoRA adapters in config
                if "lora_adapters" in config:
                    # Validate LoRA adapter entries
                    lora_adapters = config["lora_adapters"]
                    if isinstance(lora_adapters, list):
                        # Ensure each adapter has required fields
                        for adapter in lora_adapters:
                            if not isinstance(adapter, dict):
                                continue
                            if "path" not in adapter and "name" in adapter:
                                # Try to resolve adapter by name
                                adapter["path"] = self._resolve_lora_path(
                                    adapter["name"]
                                )

                    # Convert to lora_modules format for vLLM
                    if lora_adapters:
                        config["enable_lora"] = True
                        lora_modules = []
                        for adapter in lora_adapters:
                            if isinstance(adapter, dict):
                                name = adapter.get("name", "adapter")
                                path = adapter.get("path", "")
                                if path:
                                    lora_modules.append(f"{name}={path}")
                        if lora_modules:
                            config["lora_modules"] = " ".join(lora_modules)

            # Save the profile
            self.user_profiles[name] = deepcopy(profile)

            if not self._save_user_profiles():
                raise ProfileError(
                    "Failed to save profile to disk",
                    profile_name=name,
                    error_code="PROFILE_SAVE_FAILED",
                )

            logger.info(f"Successfully saved user profile: {name}")
            return True

        except ProfileError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving profile {name}: {e}")
            raise ProfileError(
                f"Unexpected error saving profile: {e}",
                profile_name=name,
                error_code="PROFILE_SAVE_ERROR",
            ) from e

    def _resolve_lora_path(self, adapter_name: str) -> str:
        """
        Try to resolve a LoRA adapter path by name.

        Args:
            adapter_name: Name of the LoRA adapter

        Returns:
            Path to the adapter if found, empty string otherwise
        """
        try:
            from ..models import scan_for_lora_adapters

            adapters = scan_for_lora_adapters()
            for adapter in adapters:
                if (
                    adapter.get("name") == adapter_name
                    or adapter.get("display_name") == adapter_name
                ):
                    return adapter.get("path", "")
        except Exception as e:
            logger.debug(f"Could not resolve LoRA path for {adapter_name}: {e}")
        return ""

    def delete_user_profile(self, name: str) -> bool:
        """Delete a user profile."""
        if name in self.user_profiles:
            del self.user_profiles[name]
            return self._save_user_profiles()
        return False

    def export_profile(self, profile_name: str, filepath: Path) -> bool:
        """Export a profile to a JSON file."""
        profile = self.get_profile(profile_name)
        if not profile:
            logger.error(f"Profile '{profile_name}' not found")
            return False

        export_data = {"version": "1.0", "profile": profile}

        try:
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Profile '{profile_name}' exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting profile: {e}")
            return False

    def import_profile(
        self,
        filepath: Path,
        new_name: Optional[str] = None,
        validate_config_func: Optional[Callable] = None,
    ) -> bool:
        """
        Import a profile from a JSON file with comprehensive validation.

        Args:
            filepath: Path to the profile file
            new_name: Optional new name for the profile
            validate_config_func: Optional function to validate profile config

        Returns:
            True if imported successfully

        Raises:
            ProfileError: If import fails for any reason
        """
        try:
            # Check file exists
            if not filepath.exists():
                raise ProfileError(
                    f"Profile file not found: {filepath}",
                    error_code="PROFILE_FILE_NOT_FOUND",
                )

            # Load and parse file
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                raise ProfileError(
                    f"Invalid JSON in profile file: {e}",
                    error_code="PROFILE_INVALID_JSON",
                ) from e

            # Validate file format
            if not isinstance(data, dict) or "profile" not in data:
                raise ProfileError(
                    "Invalid profile file format - missing 'profile' key",
                    error_code="PROFILE_INVALID_FORMAT",
                )

            profile = data["profile"]
            name = new_name or profile.get("name", filepath.stem)

            # Validate profile configuration if function provided
            if validate_config_func and "config" in profile:
                is_valid, errors = validate_config_func(profile["config"])
                if not is_valid:
                    raise ProfileError(
                        f"Profile configuration validation failed: {'; '.join(errors)}",
                        profile_name=name,
                        error_code="PROFILE_CONFIG_INVALID",
                    )

            # Save the profile
            return self.save_user_profile(name, profile)

        except ProfileError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error importing profile from {filepath}: {e}")
            raise ProfileError(
                f"Failed to import profile: {e}", error_code="PROFILE_IMPORT_ERROR"
            ) from e

    def list_profile_names(self) -> Dict[str, List[str]]:
        """
        List all profile names categorized by type.

        Returns:
            Dictionary with 'default' and 'user' lists of profile names
        """
        return {
            "default": list(self.default_profiles.keys()),
            "user": list(self.user_profiles.keys()),
        }

    def profile_exists(self, name: str) -> bool:
        """Check if a profile exists (in either default or user profiles)."""
        return name in self.default_profiles or name in self.user_profiles

    def is_user_profile(self, name: str) -> bool:
        """Check if a profile is a user-created profile."""
        return name in self.user_profiles

    def has_user_override(self, name: str) -> bool:
        """
        Check if a built-in profile has been overridden by a user profile.

        Args:
            name: Profile name to check

        Returns:
            True if this is a built-in profile that has a user override
        """
        return name in self.default_profiles and name in self.user_profiles

    def get_original_default_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get the original default profile without any user overrides.

        Args:
            name: Profile name

        Returns:
            Original default profile if it exists, None otherwise
        """
        if name in self.default_profiles:
            return deepcopy(self.default_profiles[name])
        return None

    def reset_to_default(self, name: str) -> bool:
        """
        Reset a customized built-in profile to its default settings.

        This removes the user override for a built-in profile.

        Args:
            name: Profile name to reset

        Returns:
            True if reset successfully, False otherwise
        """
        # Only reset if this is a built-in profile with a user override
        if self.has_user_override(name):
            return self.delete_user_profile(name)
        return False

    def get_profile_count(self) -> Dict[str, int]:
        """Get count of profiles by type."""
        return {
            "default": len(self.default_profiles),
            "user": len(self.user_profiles),
            "total": len(self.default_profiles) + len(self.user_profiles),
        }

    def apply_dynamic_defaults(self, profile_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply dynamic defaults to a profile configuration.

        This function intelligently sets tensor_parallel_size for multi-GPU systems.
        For single GPU systems, it allows vLLM to use its default behavior.

        Note: vLLM defaults to using only 1 GPU unless tensor_parallel_size is explicitly set.

        Args:
            profile_config: The profile configuration to enhance

        Returns:
            Enhanced profile configuration with dynamic defaults applied
        """
        config = deepcopy(profile_config)

        # Apply dynamic tensor_parallel_size if not specified
        # Note: vLLM defaults to single GPU, so we only set this for multi-GPU systems
        # where tensor parallelism would be beneficial
        if "tensor_parallel_size" not in config:
            gpu_count = self._get_gpu_count()
            if gpu_count > 1:
                config["tensor_parallel_size"] = gpu_count
                logger.debug(
                    f"Set tensor_parallel_size to {gpu_count} for multi-GPU system"
                )
            # For single GPU, let vLLM use its default (no tensor_parallel_size needed)

        return config

    def _get_gpu_count(self) -> int:
        """
        Get the number of available GPUs for tensor parallelism.

        Returns:
            Number of GPUs available, defaults to 1 if detection fails
        """
        try:
            gpus = get_gpu_info()
            gpu_count = len(gpus) if gpus else 1
            logger.debug(f"Detected {gpu_count} GPU(s)")
            return gpu_count
        except Exception as e:
            logger.warning(f"Failed to detect GPU count, defaulting to 1: {e}")
            return 1
