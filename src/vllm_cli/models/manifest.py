#!/usr/bin/env python3
"""
Manifest support for vLLM CLI.

Provides functionality to read model manifests for custom directories.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "models_manifest.json"


def load_manifest(directory: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load manifest from a directory if it exists.

    Args:
        directory: Path to directory containing manifest

    Returns:
        Manifest data or None if not found/invalid
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILENAME

    if not manifest_path.exists():
        logger.debug(f"No manifest found at {manifest_path}")
        return None

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        logger.info(f"Loaded manifest from {manifest_path}")
        return manifest

    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Error loading manifest from {manifest_path}: {e}")
        return None


def apply_manifest_to_models(
    models: List[Dict[str, Any]], directory: Path
) -> List[Dict[str, Any]]:
    """
    Apply manifest data to discovered models if manifest exists.

    Args:
        models: List of discovered models
        directory: Directory to check for manifest

    Returns:
        Updated models list with manifest data applied
    """
    manifest = load_manifest(directory)
    if not manifest:
        return models

    manifest_models = manifest.get("models", [])
    manifest_by_path = {}

    # Build lookup by path
    for entry in manifest_models:
        if entry.get("enabled", True):
            # Resolve path
            model_path = directory / entry["path"]
            manifest_by_path[str(model_path)] = entry

    # Apply manifest data to models
    updated_models = []
    for model in models:
        model_path = model.get("path", "")

        if model_path in manifest_by_path:
            manifest_entry = manifest_by_path[model_path]
            # Override with manifest data
            model["name"] = manifest_entry.get("name", model["name"])
            model["display_name"] = manifest_entry.get("name", model["display_name"])
            model["publisher"] = manifest_entry.get(
                "publisher", model.get("publisher", "unknown")
            )
            model["notes"] = manifest_entry.get("notes", "")
            model["from_manifest"] = True

        updated_models.append(model)

    return updated_models
