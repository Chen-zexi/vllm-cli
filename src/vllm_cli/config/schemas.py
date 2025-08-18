#!/usr/bin/env python3
"""
Schema management for vLLM CLI configuration.

Handles argument schema definitions, categories, and metadata
for configuration validation and UI generation.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SchemaManager:
    """
    Manages configuration schemas and argument definitions.

    Handles loading and providing access to argument schemas,
    categories, and metadata used for configuration validation
    and UI generation.
    """

    def __init__(self):
        # Paths for schema files (packaged with application)
        self.schema_dir = Path(__file__).parent.parent / "schemas"
        self.argument_schema_file = self.schema_dir / "argument_schema.json"
        self.categories_file = self.schema_dir / "categories.json"

        # Load schemas
        self.argument_schema = self._load_argument_schema()
        self.categories = self._load_categories()

    def _load_json_file(self, filepath: Path) -> Dict[str, Any]:
        """Load a JSON file safely."""
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        return {}

    def _load_argument_schema(self) -> Dict[str, Any]:
        """Load the argument schema definitions."""
        schema = self._load_json_file(self.argument_schema_file)
        return schema.get("arguments", {})

    def _load_categories(self) -> Dict[str, Any]:
        """Load category definitions."""
        categories = self._load_json_file(self.categories_file)
        return categories.get("categories", {})

    def get_argument_info(self, arg_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific argument."""
        return self.argument_schema.get(arg_name)

    def get_arguments_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all arguments belonging to a specific category."""
        args = []
        for arg_name, arg_info in self.argument_schema.items():
            if arg_info.get("category") == category:
                args.append({"name": arg_name, **arg_info})
        # Sort by importance (critical > high > medium > low)
        importance_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        args.sort(key=lambda x: importance_order.get(x.get("importance", "low"), 4))
        return args

    def get_category_info(self, category: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific category."""
        return self.categories.get(category)

    def get_ordered_categories(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get categories ordered by their display order."""
        items = list(self.categories.items())
        items.sort(key=lambda x: x[1].get("order", 999))
        return items

    def list_all_arguments(self) -> List[str]:
        """Get a list of all argument names."""
        return list(self.argument_schema.keys())

    def get_arguments_by_type(self, arg_type: str) -> List[Dict[str, Any]]:
        """Get all arguments of a specific type."""
        args = []
        for arg_name, arg_info in self.argument_schema.items():
            if arg_info.get("type") == arg_type:
                args.append({"name": arg_name, **arg_info})
        return args

    def get_critical_arguments(self) -> List[Dict[str, Any]]:
        """Get all arguments marked as critical importance."""
        return self._get_arguments_by_importance("critical")

    def get_high_priority_arguments(self) -> List[Dict[str, Any]]:
        """Get all arguments marked as high importance."""
        return self._get_arguments_by_importance("high")

    def _get_arguments_by_importance(self, importance: str) -> List[Dict[str, Any]]:
        """Get arguments by importance level."""
        args = []
        for arg_name, arg_info in self.argument_schema.items():
            if arg_info.get("importance") == importance:
                args.append({"name": arg_name, **arg_info})
        return args

    def get_category_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all categories with argument counts.

        Returns:
            Dictionary mapping category names to summary info
        """
        summary = {}

        for category_id, category_info in self.categories.items():
            args = self.get_arguments_by_category(category_id)
            summary[category_id] = {
                "name": category_info.get("name", category_id),
                "description": category_info.get("description", ""),
                "order": category_info.get("order", 999),
                "icon": category_info.get("icon", ""),
                "show_by_default": category_info.get("show_by_default", False),
                "argument_count": len(args),
                "critical_count": len(
                    [a for a in args if a.get("importance") == "critical"]
                ),
                "high_count": len([a for a in args if a.get("importance") == "high"]),
            }

        return summary

    def validate_argument_name(self, arg_name: str) -> bool:
        """Check if an argument name is valid (exists in schema)."""
        return arg_name in self.argument_schema

    def get_argument_cli_flag(self, arg_name: str) -> Optional[str]:
        """Get the CLI flag for an argument."""
        arg_info = self.get_argument_info(arg_name)
        return arg_info.get("cli_flag") if arg_info else None

    def get_argument_type(self, arg_name: str) -> Optional[str]:
        """Get the type of an argument."""
        arg_info = self.get_argument_info(arg_name)
        return arg_info.get("type") if arg_info else None

    def get_argument_description(self, arg_name: str) -> Optional[str]:
        """Get the description of an argument."""
        arg_info = self.get_argument_info(arg_name)
        return arg_info.get("description") if arg_info else None

    def get_boolean_arguments(self) -> List[str]:
        """Get list of all boolean argument names."""
        return [
            arg_name
            for arg_name, arg_info in self.argument_schema.items()
            if arg_info.get("type") == "boolean"
        ]

    def get_choice_arguments(self) -> Dict[str, List[str]]:
        """Get dictionary of choice arguments and their valid choices."""
        choice_args = {}
        for arg_name, arg_info in self.argument_schema.items():
            if arg_info.get("type") == "choice":
                choice_args[arg_name] = arg_info.get("choices", [])
        return choice_args
