#!/usr/bin/env python3
"""
Configuration persistence utilities.

Handles file I/O operations for configuration data including
JSON and YAML formats with error handling.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class PersistenceManager:
    """
    Manages configuration file persistence operations.

    Handles reading and writing configuration data in various formats
    with proper error handling and validation.
    """

    def load_json_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Load a JSON file safely.

        Args:
            filepath: Path to the JSON file

        Returns:
            Dictionary with loaded data, empty dict if file doesn't exist or fails
        """
        if not filepath.exists():
            logger.debug(f"JSON file does not exist: {filepath}")
            return {}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.debug(f"Successfully loaded JSON file: {filepath}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error loading JSON file {filepath}: {e}")

        return {}

    def save_json_file(
        self, filepath: Path, data: Dict[str, Any], indent: int = 2
    ) -> bool:
        """
        Save data to a JSON file.

        Args:
            filepath: Path where to save the file
            data: Data dictionary to save
            indent: JSON indentation level

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)

            logger.debug(f"Successfully saved JSON file: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving JSON file {filepath}: {e}")
            return False

    def load_yaml_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Load a YAML file safely.

        Args:
            filepath: Path to the YAML file

        Returns:
            Dictionary with loaded data, empty dict if file doesn't exist or fails
        """
        if not filepath.exists():
            logger.debug(f"YAML file does not exist: {filepath}")
            return {}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                # Handle case where YAML file is empty
                if data is None:
                    data = {}
                logger.debug(f"Successfully loaded YAML file: {filepath}")
                return data
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in file {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error loading YAML file {filepath}: {e}")

        return {}

    def save_yaml_file(self, filepath: Path, data: Dict[str, Any]) -> bool:
        """
        Save data to a YAML file.

        Args:
            filepath: Path where to save the file
            data: Data dictionary to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

            logger.debug(f"Successfully saved YAML file: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving YAML file {filepath}: {e}")
            return False

    def backup_file(self, filepath: Path, backup_suffix: str = ".backup") -> bool:
        """
        Create a backup copy of a file.

        Args:
            filepath: Path to the file to backup
            backup_suffix: Suffix to add to backup filename

        Returns:
            True if backup created successfully, False otherwise
        """
        if not filepath.exists():
            logger.warning(f"Cannot backup non-existent file: {filepath}")
            return False

        try:
            backup_path = filepath.with_suffix(filepath.suffix + backup_suffix)

            # Copy file contents
            with open(filepath, "rb") as src, open(backup_path, "wb") as dst:
                dst.write(src.read())

            logger.info(f"Created backup: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating backup of {filepath}: {e}")
            return False

    def restore_from_backup(
        self, filepath: Path, backup_suffix: str = ".backup"
    ) -> bool:
        """
        Restore a file from its backup.

        Args:
            filepath: Path to the file to restore
            backup_suffix: Suffix of the backup file

        Returns:
            True if restored successfully, False otherwise
        """
        backup_path = filepath.with_suffix(filepath.suffix + backup_suffix)

        if not backup_path.exists():
            logger.error(f"Backup file does not exist: {backup_path}")
            return False

        try:
            # Copy backup contents to original file
            with open(backup_path, "rb") as src, open(filepath, "wb") as dst:
                dst.write(src.read())

            logger.info(f"Restored from backup: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error restoring from backup {backup_path}: {e}")
            return False

    def file_exists_and_readable(self, filepath: Path) -> bool:
        """
        Check if a file exists and is readable.

        Args:
            filepath: Path to check

        Returns:
            True if file exists and is readable, False otherwise
        """
        try:
            return (
                filepath.exists()
                and filepath.is_file()
                and filepath.stat().st_size >= 0
            )
        except Exception as e:
            logger.debug(f"Error checking file {filepath}: {e}")
            return False

    def get_file_size(self, filepath: Path) -> int:
        """
        Get the size of a file in bytes.

        Args:
            filepath: Path to the file

        Returns:
            File size in bytes, -1 if error
        """
        try:
            if filepath.exists():
                return filepath.stat().st_size
        except Exception as e:
            logger.debug(f"Error getting file size for {filepath}: {e}")

        return -1

    def cleanup_temp_files(self, directory: Path, pattern: str = "*.tmp") -> int:
        """
        Clean up temporary files in a directory.

        Args:
            directory: Directory to clean
            pattern: File pattern to match (glob)

        Returns:
            Number of files deleted
        """
        if not directory.exists():
            return 0

        deleted = 0
        try:
            for temp_file in directory.glob(pattern):
                try:
                    temp_file.unlink()
                    deleted += 1
                    logger.debug(f"Deleted temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning temp files in {directory}: {e}")

        return deleted
