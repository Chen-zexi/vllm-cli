#!/usr/bin/env python3
"""
Shortcut management for vLLM CLI.

Handles shortcuts which are saved combinations of model + profile configurations
for quick launching.
"""
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import ConfigurationError

logger = logging.getLogger(__name__)


class ShortcutManager:
    """
    Manages shortcuts for vLLM configurations.

    Shortcuts are saved combinations of model + profile that allow users
    to quickly launch frequently used configurations.
    """

    def __init__(self, config_dir: Path):
        """
        Initialize the ShortcutManager.

        Args:
            config_dir: Directory for configuration files
        """
        self.config_dir = config_dir
        self.shortcuts_file = config_dir / "shortcuts.json"

        # Load existing shortcuts
        self.shortcuts = self._load_shortcuts()

    def _load_shortcuts(self) -> Dict[str, Dict[str, Any]]:
        """Load shortcuts from file."""
        if self.shortcuts_file.exists():
            try:
                with open(self.shortcuts_file, "r") as f:
                    data = json.load(f)
                    # Handle both old and new format
                    if isinstance(data, dict):
                        if "shortcuts" in data:
                            return data["shortcuts"]
                        else:
                            # Assume it's already in the right format
                            return data
                    return {}
            except Exception as e:
                logger.error(f"Error loading shortcuts: {e}")
                return {}
        return {}

    def _save_shortcuts(self) -> bool:
        """Save shortcuts to file."""
        try:
            # Save with metadata
            data = {
                "version": "1.0",
                "description": "User shortcuts for model + profile combinations",
                "shortcuts": self.shortcuts,
            }
            with open(self.shortcuts_file, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving shortcuts: {e}")
            return False

    def get_all_shortcuts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all shortcuts.

        Returns:
            Dictionary of shortcut_name -> shortcut_data
        """
        # Reload shortcuts from disk to ensure we have the latest data
        self.shortcuts = self._load_shortcuts()
        return deepcopy(self.shortcuts)

    def get_shortcut(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific shortcut by name.

        Args:
            name: Shortcut name

        Returns:
            Shortcut data if found, None otherwise
        """
        if name in self.shortcuts:
            return deepcopy(self.shortcuts[name])
        return None

    def save_shortcut(self, name: str, shortcut_data: Dict[str, Any]) -> bool:
        """
        Save or update a shortcut.

        Args:
            name: Shortcut name
            shortcut_data: Shortcut configuration containing:
                - model: Model name or path
                - profile: Profile name to use
                - description: Optional description
                - config_overrides: Optional config overrides

        Returns:
            True if saved successfully

        Raises:
            ConfigurationError: If shortcut data is invalid
        """
        try:
            # Validate required fields
            if not isinstance(shortcut_data, dict):
                raise ConfigurationError(
                    "Shortcut data must be a dictionary",
                    error_code="INVALID_SHORTCUT_TYPE",
                )

            if "model" not in shortcut_data:
                raise ConfigurationError(
                    "Shortcut must specify a model", error_code="SHORTCUT_MISSING_MODEL"
                )

            if "profile" not in shortcut_data:
                raise ConfigurationError(
                    "Shortcut must specify a profile",
                    error_code="SHORTCUT_MISSING_PROFILE",
                )

            # Prepare shortcut data
            clean_data = {
                "name": name,
                "model": shortcut_data["model"],
                "profile": shortcut_data["profile"],
                "description": shortcut_data.get(
                    "description", f"Shortcut for {shortcut_data['model']}"
                ),
                "created_at": shortcut_data.get(
                    "created_at", None
                ),  # Preserve creation time if exists
                "last_used": shortcut_data.get("last_used", None),
            }

            # Add optional fields if present
            if "config_overrides" in shortcut_data:
                clean_data["config_overrides"] = shortcut_data["config_overrides"]

            if "icon" in shortcut_data:
                clean_data["icon"] = shortcut_data["icon"]

            # If this is a new shortcut, add creation timestamp
            if name not in self.shortcuts:
                from datetime import datetime

                clean_data["created_at"] = datetime.now().isoformat()

            # Save the shortcut
            self.shortcuts[name] = clean_data

            if not self._save_shortcuts():
                raise ConfigurationError(
                    "Failed to save shortcut to disk", error_code="SHORTCUT_SAVE_FAILED"
                )

            logger.info(f"Successfully saved shortcut: {name}")
            return True

        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving shortcut {name}: {e}")
            raise ConfigurationError(
                f"Unexpected error saving shortcut: {e}",
                error_code="SHORTCUT_SAVE_ERROR",
            ) from e

    def delete_shortcut(self, name: str) -> bool:
        """
        Delete a shortcut.

        Args:
            name: Shortcut name to delete

        Returns:
            True if deleted successfully, False if not found
        """
        if name in self.shortcuts:
            del self.shortcuts[name]
            return self._save_shortcuts()
        return False

    def update_last_used(self, name: str) -> None:
        """
        Update the last used timestamp for a shortcut.

        Args:
            name: Shortcut name
        """
        if name in self.shortcuts:
            from datetime import datetime

            self.shortcuts[name]["last_used"] = datetime.now().isoformat()
            self._save_shortcuts()

    def list_shortcuts(self) -> List[Dict[str, Any]]:
        """
        List all shortcuts with summary information.

        Returns:
            List of shortcut summaries sorted by name
        """
        # Reload shortcuts from disk to ensure we have the latest data
        self.shortcuts = self._load_shortcuts()

        shortcuts_list = []
        for name, data in self.shortcuts.items():
            summary = {
                "name": name,
                "model": data.get("model", "unknown"),
                "profile": data.get("profile", "unknown"),
                "description": data.get("description", ""),
                "last_used": data.get("last_used", None),
            }
            shortcuts_list.append(summary)

        # Sort by name
        shortcuts_list.sort(key=lambda x: x["name"].lower())
        return shortcuts_list

    def get_recent_shortcuts(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recently used shortcuts.

        Args:
            limit: Maximum number of shortcuts to return

        Returns:
            List of recently used shortcuts
        """
        # Reload shortcuts from disk to ensure we have the latest data
        self.shortcuts = self._load_shortcuts()

        shortcuts_with_time = []
        for name, data in self.shortcuts.items():
            if "last_used" in data and data["last_used"]:
                shortcuts_with_time.append(
                    {
                        "name": name,
                        "model": data.get("model", "unknown"),
                        "profile": data.get("profile", "unknown"),
                        "description": data.get("description", ""),
                        "last_used": data["last_used"],
                    }
                )

        # Sort by last_used timestamp (most recent first)
        shortcuts_with_time.sort(key=lambda x: x["last_used"], reverse=True)

        return shortcuts_with_time[:limit]

    def export_shortcut(self, name: str, filepath: Path) -> bool:
        """
        Export a shortcut to a JSON file.

        Args:
            name: Shortcut name to export
            filepath: Path to export file

        Returns:
            True if exported successfully
        """
        shortcut = self.get_shortcut(name)
        if not shortcut:
            logger.error(f"Shortcut '{name}' not found")
            return False

        export_data = {
            "version": "1.0",
            "type": "vllm-cli-shortcut",
            "shortcut": shortcut,
        }

        try:
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Shortcut '{name}' exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting shortcut: {e}")
            return False

    def import_shortcut(self, filepath: Path, new_name: Optional[str] = None) -> bool:
        """
        Import a shortcut from a JSON file.

        Args:
            filepath: Path to the shortcut file
            new_name: Optional new name for the shortcut

        Returns:
            True if imported successfully

        Raises:
            ConfigurationError: If import fails
        """
        try:
            if not filepath.exists():
                raise ConfigurationError(
                    f"Shortcut file not found: {filepath}",
                    error_code="SHORTCUT_FILE_NOT_FOUND",
                )

            with open(filepath, "r") as f:
                data = json.load(f)

            # Validate file format
            if not isinstance(data, dict) or "shortcut" not in data:
                raise ConfigurationError(
                    "Invalid shortcut file format", error_code="SHORTCUT_INVALID_FORMAT"
                )

            shortcut = data["shortcut"]
            name = new_name or shortcut.get("name", filepath.stem)

            # Save the imported shortcut
            return self.save_shortcut(name, shortcut)

        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Error importing shortcut from {filepath}: {e}")
            raise ConfigurationError(
                f"Failed to import shortcut: {e}", error_code="SHORTCUT_IMPORT_ERROR"
            ) from e

    def rename_shortcut(self, old_name: str, new_name: str) -> bool:
        """
        Rename a shortcut.

        Args:
            old_name: Current shortcut name
            new_name: New shortcut name

        Returns:
            True if renamed successfully
        """
        if old_name not in self.shortcuts:
            logger.error(f"Shortcut '{old_name}' not found")
            return False

        if new_name in self.shortcuts:
            logger.error(f"Shortcut '{new_name}' already exists")
            return False

        # Copy the shortcut with new name
        self.shortcuts[new_name] = self.shortcuts[old_name]
        self.shortcuts[new_name]["name"] = new_name

        # Delete old shortcut
        del self.shortcuts[old_name]

        return self._save_shortcuts()

    def clear_all_shortcuts(self) -> bool:
        """
        Clear all shortcuts (with confirmation in UI).

        Returns:
            True if cleared successfully
        """
        self.shortcuts = {}
        return self._save_shortcuts()
