#!/usr/bin/env python3
"""
Model discovery utilities for vLLM CLI.

Handles detection and enumeration of available models from various sources
including hf-model-tool and fallback directory scanning.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .manifest import apply_manifest_to_models

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
        # Try using shared registry first for best performance
        from hf_model_tool import get_registry
        from hf_model_tool.config import ConfigManager as HFConfigManager

        registry = get_registry()
        registry.scan_all()  # Ensure data is current

        # Get all models (including custom)
        models = list(registry.models.values())
        models.extend(registry.custom_models.values())

        # Check if Ollama scanning is enabled in hf-model-tool config
        hf_config_manager = HFConfigManager()
        hf_config = hf_config_manager.load_config()
        scan_ollama = hf_config.get("scan_ollama", False)
        has_ollama_dirs = bool(hf_config.get("ollama_directories", []))

        # Only include Ollama models if scanning is enabled or custom dirs exist
        if scan_ollama or has_ollama_dirs:
            models.extend(registry.ollama_models.values())
            models.extend(registry.gguf_models.values())
            logger.info(
                f"Including Ollama models (scan_ollama={scan_ollama}, custom_dirs={has_ollama_dirs})"
            )
        else:
            logger.info("Excluding Ollama models (scanning disabled in hf-model-tool)")

        if models:
            logger.info(f"Found {len(models)} models via ModelRegistry")
            return models
    except ImportError:
        pass  # Try legacy method
    except Exception as e:
        logger.debug(f"Registry not available: {e}")

    try:
        # Try using hf-model-tool API or legacy method
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
    try:
        from hf_model_tool import HFModelAPI

        # Use new API if available
        api = HFModelAPI()
        logger.debug("Scanning for models using HFModelAPI...")

        # Get all models (not LoRA or datasets)
        models = api.list_assets(
            asset_type="model", include_lora=False, include_datasets=False
        )

        # Also get custom models
        custom_models = api.list_assets(
            asset_type="custom_model", include_lora=False, include_datasets=False
        )
        models.extend(custom_models)

        return models
    except ImportError:
        # Fallback to old method
        from hf_model_tool.cache import scan_all_directories

        logger.debug("Scanning for models using hf-model-tool cache (legacy)...")

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


def scan_for_lora_adapters() -> List[Dict[str, Any]]:
    """
    Scan for available LoRA adapters.

    Returns:
        List of LoRA adapter information dictionaries
    """
    try:
        # Try using shared registry first for best performance
        from hf_model_tool import get_registry

        registry = get_registry()
        registry.scan_all()  # Ensure data is current

        adapters = list(registry.lora_adapters.values())
        if adapters:
            # Process adapters to ensure correct path and rank
            for adapter in adapters:
                # Extract metadata fields to top level
                if "metadata" in adapter:
                    metadata = adapter["metadata"]
                    if isinstance(metadata, dict):
                        adapter["base_model"] = metadata.get("base_model", "unknown")
                        adapter["rank"] = metadata.get(
                            "lora_rank", metadata.get("rank", metadata.get("r", 16))
                        )

                # Use lora_path if available
                if "lora_path" in adapter and adapter["lora_path"]:
                    adapter["path"] = adapter["lora_path"]

            logger.info(f"Found {len(adapters)} LoRA adapters via ModelRegistry")
            return adapters
    except ImportError:
        pass  # Try legacy method
    except Exception as e:
        logger.debug(f"Registry not available: {e}")

    try:
        # Try using hf-model-tool API or legacy method
        adapters = _scan_lora_with_hf_model_tool()
        if adapters:
            logger.info(f"Found {len(adapters)} LoRA adapters via hf-model-tool")
            return adapters
    except ImportError:
        logger.warning("hf-model-tool not available for LoRA scanning")
    except Exception as e:
        logger.error(f"Error scanning LoRA adapters with hf-model-tool: {e}")

    # Fallback to manual search
    return _fallback_lora_search()


def _scan_lora_with_hf_model_tool() -> List[Dict[str, Any]]:
    """
    Scan for LoRA adapters using hf-model-tool.

    Returns:
        List of LoRA adapter dictionaries
    """
    try:
        from hf_model_tool import HFModelAPI

        # Use new API if available
        api = HFModelAPI()
        logger.debug("Scanning for LoRA adapters using HFModelAPI...")

        # Get all LoRA adapters with enriched details
        adapters = api.list_lora_adapters()

        # Ensure all adapters have the expected fields
        for adapter in adapters:
            # Extract metadata fields to top level for compatibility
            if "metadata" in adapter:
                metadata = adapter["metadata"]
                if isinstance(metadata, dict):
                    adapter["base_model"] = metadata.get("base_model", "unknown")
                    adapter["trained_on"] = metadata.get("trained_on", "unknown")
                    adapter["rank"] = metadata.get(
                        "lora_rank", metadata.get("rank", metadata.get("r", 16))
                    )

            # Ensure rank is at top level
            if "rank" not in adapter and "metadata" in adapter:
                adapter["rank"] = adapter["metadata"].get(
                    "lora_rank",
                    adapter["metadata"].get("rank", adapter["metadata"].get("r", 16)),
                )

            # Use lora_path if available (the actual adapter directory)
            # This is important as hf-model-tool returns both path (parent) and lora_path (actual adapter)
            if "lora_path" in adapter and adapter["lora_path"]:
                adapter["path"] = adapter["lora_path"]

        return adapters

    except ImportError:
        # Fallback to old method
        from hf_model_tool.cache import scan_all_directories

        logger.debug("Scanning for LoRA adapters using hf-model-tool cache (legacy)...")

        # Get all items with full details
        all_items = scan_all_directories()

        if not all_items:
            return []

        # Filter for LoRA adapters
        adapters = []
        for item in all_items:
            if item.get("type") == "lora_adapter":
                # Add additional LoRA-specific information
                adapter_info = item.copy()

                # Extract base model requirement if available
                if "metadata" in adapter_info:
                    metadata = adapter_info["metadata"]
                    if isinstance(metadata, dict):
                        adapter_info["base_model"] = metadata.get(
                            "base_model", "unknown"
                        )
                        adapter_info["trained_on"] = metadata.get(
                            "trained_on", "unknown"
                        )
                        # Look for rank in both 'rank' and 'lora_rank' keys
                        adapter_info["rank"] = metadata.get(
                            "lora_rank", metadata.get("rank", metadata.get("r", 16))
                        )

                # Use lora_path if available (from AssetDetector)
                if "lora_path" in item:
                    adapter_info["path"] = item["lora_path"]

                # Ensure rank is available at top level
                if "rank" not in adapter_info and "metadata" in adapter_info:
                    # Look for rank in both 'lora_rank' and 'rank' keys
                    adapter_info["rank"] = adapter_info["metadata"].get(
                        "lora_rank",
                        adapter_info["metadata"].get(
                            "rank", adapter_info["metadata"].get("r", 16)
                        ),
                    )

                adapters.append(adapter_info)

        return adapters


def _fallback_lora_search() -> List[Dict[str, Any]]:
    """
    Fallback method to search for LoRA adapters.

    Returns:
        List of found LoRA adapters
    """
    from .directories import DirectoryManager

    adapters = []
    dir_manager = DirectoryManager()

    # Get directories to search
    search_paths = dir_manager.get_lora_directories()
    if not search_paths:
        search_paths = dir_manager.get_model_directories()

    for base_path in search_paths:
        if not base_path.exists():
            continue

        try:
            adapters.extend(_scan_directory_for_lora(base_path))
        except Exception as e:
            logger.warning(f"Error searching {base_path} for LoRA: {e}")

    logger.info(f"Found {len(adapters)} LoRA adapters via fallback search")
    return adapters


def _scan_directory_for_lora(base_path: Path) -> List[Dict[str, Any]]:
    """
    Scan a directory for LoRA adapter files.

    Args:
        base_path: Base directory to scan

    Returns:
        List of LoRA adapter dictionaries
    """
    adapters = []

    # Look for adapter directories
    for adapter_dir in base_path.rglob("*"):
        if not adapter_dir.is_dir():
            continue

        # Check if it contains LoRA adapter files
        if _is_lora_directory(adapter_dir):
            adapter_info = _extract_lora_info(adapter_dir)
            if adapter_info:
                adapters.append(adapter_info)

    return adapters


def _is_lora_directory(directory: Path) -> bool:
    """
    Check if a directory contains a LoRA adapter.

    Args:
        directory: Directory to check

    Returns:
        True if directory contains a LoRA adapter
    """
    # Check for adapter configuration file
    adapter_config = directory / "adapter_config.json"
    if not adapter_config.exists():
        return False

    # Check for adapter weights
    adapter_patterns = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_adapter.bin",
    ]

    for pattern in adapter_patterns:
        if (directory / pattern).exists():
            return True

    return False


def _extract_lora_info(adapter_dir: Path) -> Dict[str, Any]:
    """
    Extract information from a LoRA adapter directory.

    Args:
        adapter_dir: Path to LoRA adapter directory

    Returns:
        Dictionary with LoRA adapter information
    """
    import json

    try:
        # Read adapter config
        config_file = adapter_dir / "adapter_config.json"
        base_model = "unknown"
        task_type = "unknown"
        rank = 16  # Default LoRA rank

        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                    base_model = config.get("base_model_name_or_path", "unknown")
                    task_type = config.get("task_type", "unknown")
                    # Extract LoRA rank (r parameter)
                    rank = config.get("r", 16)
            except Exception as e:
                logger.debug(f"Error reading adapter config: {e}")

        # Calculate adapter size
        size = sum(f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file())

        # Extract adapter name from directory
        adapter_name = adapter_dir.name

        return {
            "name": adapter_name,
            "type": "lora_adapter",
            "path": str(adapter_dir),
            "size": size,
            "base_model": base_model,
            "task_type": task_type,
            "display_name": adapter_name,
            "metadata": {
                "base_model": base_model,
                "task_type": task_type,
                "rank": rank,
            },
            "rank": rank,  # Also store at top level for easy access
        }

    except Exception as e:
        logger.debug(f"Error extracting LoRA info from {adapter_dir}: {e}")
        return {}


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

    # Also add user-configured directories from vLLM CLI config
    try:
        from ..config import ConfigManager

        config_manager = ConfigManager()
        user_dirs = config_manager.config.get("model_directories", [])
        for dir_path in user_dirs:
            p = Path(dir_path)
            if p not in search_paths:
                search_paths.append(p)
    except Exception as e:
        logger.debug(f"Could not load user directories: {e}")

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

    # Apply manifest data if available
    models = apply_manifest_to_models(models, base_path)

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
