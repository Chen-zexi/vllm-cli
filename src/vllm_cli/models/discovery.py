#!/usr/bin/env python3
"""
Model discovery utilities for vLLM CLI.

Handles detection and enumeration of available models from various sources
including hf-model-tool and fallback directory scanning.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def scan_for_models() -> List[Dict[str, Any]]:
    """
    Scan for available models using hf-model-tool or fallback methods.

    Attempts to use hf-model-tool for comprehensive model discovery,
    falling back to manual directory scanning if not available.

    Returns:
        List of model information dictionaries
    """
    try:
        # Try using hf-model-tool first
        models = _scan_with_hf_model_tool()
        if models:
            logger.info(f"Found {len(models)} models via hf-model-tool")
            return models
    except ImportError:
        logger.warning("hf-model-tool not available, using fallback search")
    except Exception as e:
        logger.error(f"Error listing models with hf-model-tool: {e}")
        logger.info("Falling back to manual search")

    # Fallback to manual search
    return _fallback_model_search()


def _scan_with_hf_model_tool() -> List[Dict[str, Any]]:
    """
    Scan for models using hf-model-tool.

    Returns:
        List of model dictionaries from hf-model-tool
    """
    from hf_model_tool.cache import scan_all_directories

    logger.debug("Scanning for models using hf-model-tool...")

    # Get all items with full details
    all_items = scan_all_directories()

    if not all_items:
        return []

    # Filter for models only (not datasets or other assets)
    models = []
    for item in all_items:
        if item.get("type") in ["model", "custom_model"]:
            models.append(item)

    return models


def _fallback_model_search() -> List[Dict[str, Any]]:
    """
    Fallback method to search for models when hf-model-tool is not available.

    Searches common model directories for model files and attempts to
    extract basic information.

    Returns:
        List of found models
    """
    models = []

    # Common model directories
    search_paths = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface" / "transformers",
        Path("/data/models"),  # Common custom location
        Path("/models"),  # Another common location
        Path.home() / "models",  # User home models
    ]

    for base_path in search_paths:
        if not base_path.exists():
            continue

        try:
            models.extend(_scan_directory_for_models(base_path))
        except Exception as e:
            logger.warning(f"Error searching {base_path}: {e}")
            # Continue with other search paths

    logger.info(f"Found {len(models)} models via fallback search")
    return models


def _scan_directory_for_models(base_path: Path) -> List[Dict[str, Any]]:
    """
    Scan a directory for model files.

    Args:
        base_path: Base directory to scan

    Returns:
        List of model dictionaries found in the directory
    """
    models = []

    # Look for model directories
    for model_dir in base_path.glob("*"):
        if not model_dir.is_dir():
            continue

        # Check if it looks like a model directory
        if _is_model_directory(model_dir):
            model_info = _extract_basic_model_info(model_dir)
            if model_info:
                models.append(model_info)

    return models


def _is_model_directory(model_dir: Path) -> bool:
    """
    Check if a directory contains a model.

    Args:
        model_dir: Directory to check

    Returns:
        True if directory appears to contain a model
    """
    # Check for config.json (required for most models)
    config_file = model_dir / "config.json"
    if not config_file.exists():
        return False

    # Check for model weights
    weight_patterns = ["*.bin", "*.safetensors", "*.pt", "*.pth"]
    has_weights = any(list(model_dir.glob(pattern)) for pattern in weight_patterns)

    return has_weights


def _extract_basic_model_info(model_dir: Path) -> Dict[str, Any]:
    """
    Extract basic information from a model directory.

    Args:
        model_dir: Path to model directory

    Returns:
        Dictionary with basic model information
    """
    try:
        # Calculate directory size
        size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())

        # Extract model name from directory
        model_name = model_dir.name

        # Try to extract publisher from path structure
        publisher = "unknown"
        if len(model_dir.parts) >= 2:
            # Common pattern: .../publisher/model_name
            potential_publisher = model_dir.parent.name
            if potential_publisher not in ["hub", "transformers", "models"]:
                publisher = potential_publisher

        return {
            "name": model_name,
            "size": size,
            "path": str(model_dir),
            "type": "model",
            "publisher": publisher,
            "display_name": model_name,
            "metadata": {},
        }

    except Exception as e:
        logger.debug(f"Error extracting info from {model_dir}: {e}")
        return {}


def build_model_dict(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a standardized model dictionary from model data.

    Takes raw model information from various sources and normalizes
    it into a consistent format.

    Args:
        item: Model item data

    Returns:
        Standardized model dictionary
    """
    publisher = item.get("publisher", "unknown")
    display_name = item.get("display_name", item.get("name", "unknown"))

    # Create proper model name
    if publisher and publisher != "unknown":
        model_name = f"{publisher}/{display_name}"
    else:
        model_name = display_name

    return {
        "name": model_name,
        "size": item.get("size", 0),
        "path": item.get("path", ""),
        "type": item.get("type", "model"),
        "publisher": publisher,
        "display_name": display_name,
        "metadata": item.get("metadata", {}),
    }


def validate_model_path(model_path: str) -> bool:
    """
    Validate if a model path exists and contains a valid model.

    Args:
        model_path: Path to the model directory

    Returns:
        True if valid model path, False otherwise
    """
    path = Path(model_path)

    if not path.exists() or not path.is_dir():
        return False

    return _is_model_directory(path)


def find_model_by_name(
    model_name: str, search_paths: List[Path] = None
) -> Optional[Path]:
    """
    Find a model directory by name.

    Args:
        model_name: Name of the model to find
        search_paths: Optional list of paths to search

    Returns:
        Path to model directory if found, None otherwise
    """
    if search_paths is None:
        search_paths = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "huggingface" / "transformers",
            Path("/data/models"),
            Path("/models"),
            Path.home() / "models",
        ]

    for base_path in search_paths:
        if not base_path.exists():
            continue

        # Try direct match
        model_path = base_path / model_name
        if model_path.exists() and _is_model_directory(model_path):
            return model_path

        # Try searching subdirectories
        for model_dir in base_path.glob("*"):
            if model_dir.is_dir() and model_dir.name == model_name:
                if _is_model_directory(model_dir):
                    return model_dir

    return None
