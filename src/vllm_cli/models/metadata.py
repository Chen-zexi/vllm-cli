#!/usr/bin/env python3
"""
Model metadata extraction utilities.

Extracts detailed metadata from model files including configuration,
architecture information, and resource requirements.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def extract_model_config(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract configuration from a model directory.

    Reads config.json and other metadata files to build a comprehensive
    configuration dictionary for the model.

    Args:
        model_path: Path to the model directory

    Returns:
        Dictionary with extracted configuration or None if extraction fails
    """
    try:
        model_dir = Path(model_path)
        config_path = model_dir / "config.json"

        if not config_path.exists():
            return None

        # Read main config.json
        with open(config_path, "r") as f:
            config = json.load(f)

        # Extract key parameters
        extracted_config = {
            "config": config,
            "hidden_size": config.get("hidden_size", "unknown"),
            "num_layers": config.get("num_hidden_layers", "unknown"),
            "max_position_embeddings": config.get("max_position_embeddings", "unknown"),
            "vocab_size": config.get("vocab_size", "unknown"),
            "model_type": config.get("model_type", "unknown"),
            "torch_dtype": config.get("torch_dtype", "unknown"),
        }

        # Extract architecture information
        architectures = config.get("architectures")
        if architectures and isinstance(architectures, list):
            extracted_config["architecture"] = architectures[0]
        else:
            extracted_config["architecture"] = "unknown"

        # Try to extract additional metadata from other files
        additional_metadata = _extract_additional_metadata(model_dir)
        if additional_metadata:
            extracted_config.update(additional_metadata)

        return extracted_config

    except Exception as e:
        logger.warning(f"Could not read config.json for {model_path}: {e}")
        return None


def _extract_additional_metadata(model_dir: Path) -> Dict[str, Any]:
    """
    Extract additional metadata from various model files.

    Args:
        model_dir: Path to model directory

    Returns:
        Dictionary with additional metadata
    """
    metadata = {}

    # Try to read tokenizer config
    tokenizer_config = _read_tokenizer_config(model_dir)
    if tokenizer_config:
        metadata["tokenizer_config"] = tokenizer_config

    # Try to read generation config
    generation_config = _read_generation_config(model_dir)
    if generation_config:
        metadata["generation_config"] = generation_config

    # Get file information
    file_info = _get_model_file_info(model_dir)
    if file_info:
        metadata["file_info"] = file_info

    return metadata


def _read_tokenizer_config(model_dir: Path) -> Optional[Dict[str, Any]]:
    """Read tokenizer configuration if available."""
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        try:
            with open(tokenizer_config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Error reading tokenizer config: {e}")
    return None


def _read_generation_config(model_dir: Path) -> Optional[Dict[str, Any]]:
    """Read generation configuration if available."""
    generation_config_path = model_dir / "generation_config.json"
    if generation_config_path.exists():
        try:
            with open(generation_config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Error reading generation config: {e}")
    return None


def _get_model_file_info(model_dir: Path) -> Dict[str, Any]:
    """Get information about model files."""
    file_info = {"weight_files": [], "total_size": 0, "file_types": set()}

    # Weight file patterns
    weight_patterns = ["*.bin", "*.safetensors", "*.pt", "*.pth"]

    for pattern in weight_patterns:
        for weight_file in model_dir.glob(pattern):
            if weight_file.is_file():
                size = weight_file.stat().st_size
                file_info["weight_files"].append(
                    {"name": weight_file.name, "size": size, "type": weight_file.suffix}
                )
                file_info["total_size"] += size
                file_info["file_types"].add(weight_file.suffix)

    # Convert set to list for JSON serialization
    file_info["file_types"] = list(file_info["file_types"])

    return file_info


def get_model_requirements(
    model_name: str, model_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Estimate resource requirements for a model.

    Args:
        model_name: Name of the model
        model_data: Optional model data dictionary (from list_available_models)

    Returns:
        Dictionary with estimated requirements
    """
    # If no model data provided, try to get it
    if model_data is None:
        from .manager import get_model_details

        model_data = get_model_details(model_name)

    if not model_data:
        return {
            "min_gpu_memory": "unknown",
            "recommended_gpu_memory": "unknown",
            "recommended_dtype": "auto",
            "supports_tp": False,
        }

    # Estimate based on model size
    size_gb = model_data.get("size", 0) / (1024**3)

    # Basic heuristics for requirements
    requirements = {
        "model_size_gb": round(size_gb, 2),
        "min_gpu_memory": f"{int(size_gb * 1.2)} GB",  # Minimum with quantization
        "recommended_gpu_memory": f"{int(size_gb * 2)} GB",  # Comfortable margin
        "recommended_dtype": "bfloat16" if size_gb > 20 else "float16",
        "supports_tp": size_gb > 40,  # Large models benefit from tensor parallelism
        "recommended_tp_size": 2 if size_gb > 40 else 1,
    }

    # Adjust for specific architectures
    arch = model_data.get("architecture", "").lower()
    if "llama" in arch or "qwen" in arch:
        requirements["supports_flash_attention"] = True
    if "mixtral" in arch or "moe" in arch:
        requirements["supports_expert_parallel"] = True

    # Add model-specific recommendations
    requirements.update(_get_architecture_specific_requirements(arch, size_gb))

    return requirements


def _get_architecture_specific_requirements(
    architecture: str, size_gb: float
) -> Dict[str, Any]:
    """
    Get architecture-specific requirements and recommendations.

    Args:
        architecture: Model architecture name
        size_gb: Model size in GB

    Returns:
        Dictionary with architecture-specific requirements
    """
    arch_requirements = {}

    arch_lower = architecture.lower()

    # LLaMA-based models
    if "llama" in arch_lower:
        arch_requirements.update(
            {
                "supports_flash_attention": True,
                "supports_grouped_query_attention": True,
                "recommended_quantization": (
                    ["awq", "gptq"] if size_gb > 10 else ["bitsandbytes"]
                ),
            }
        )

    # Qwen models
    elif "qwen" in arch_lower:
        arch_requirements.update(
            {
                "supports_flash_attention": True,
                "supports_sliding_window": True,
                "recommended_quantization": ["awq", "gptq"],
            }
        )

    # Mixtral models (MoE)
    elif "mixtral" in arch_lower:
        arch_requirements.update(
            {
                "supports_expert_parallel": True,
                "supports_flash_attention": True,
                "recommended_tp_size": 2 if size_gb > 20 else 1,
                "notes": "MoE models benefit from expert parallelism",
            }
        )

    # Mistral models
    elif "mistral" in arch_lower:
        arch_requirements.update(
            {
                "supports_flash_attention": True,
                "supports_sliding_window": True,
                "recommended_quantization": ["awq", "gptq"],
            }
        )

    return arch_requirements


def analyze_model_compatibility(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze model compatibility with vLLM.

    Args:
        model_data: Model information dictionary

    Returns:
        Dictionary with compatibility analysis
    """
    compatibility = {
        "is_compatible": True,
        "warnings": [],
        "recommendations": [],
        "unsupported_features": [],
    }

    architecture = model_data.get("architecture", "").lower()

    # Check for known compatible architectures
    compatible_archs = [
        "llama",
        "qwen",
        "mixtral",
        "mistral",
        "phi",
        "gemma",
        "bloom",
        "falcon",
        "gpt_neox",
        "opt",
        "chatglm",
    ]

    is_known_arch = any(arch in architecture for arch in compatible_archs)

    if not is_known_arch:
        compatibility["warnings"].append(
            f"Architecture '{architecture}' may not be fully supported"
        )
        compatibility["recommendations"].append(
            "Check vLLM documentation for architecture support"
        )

    # Check model size
    size_gb = model_data.get("size", 0) / (1024**3)
    if size_gb > 100:
        compatibility["warnings"].append(
            "Very large model may require significant GPU memory"
        )
        compatibility["recommendations"].append(
            "Consider using tensor parallelism or quantization"
        )

    # Check for custom code requirements
    config = model_data.get("config", {})
    if config.get("auto_map") or config.get("custom_pipelines"):
        compatibility["warnings"].append("Model uses custom code")
        compatibility["recommendations"].append(
            "Use trust_remote_code=true when loading"
        )

    return compatibility
