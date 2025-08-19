#!/usr/bin/env python3
"""
Configuration Management Package

This package provides comprehensive configuration management for vLLM CLI
including profiles, schemas, validation, persistence, and shortcuts.

Main Components:
- ConfigManager: Main configuration management class
- ProfileManager: User profile CRUD operations
- SchemaManager: Argument schema and category management
- PersistenceManager: File I/O operations
- ShortcutManager: Model + profile combination shortcuts

The main ConfigManager class integrates all sub-components and provides
a unified interface for configuration operations.
"""

# Main configuration manager
from .manager import ConfigManager
from .persistence import PersistenceManager

# Sub-managers for direct access if needed
from .profiles import ProfileManager
from .schemas import SchemaManager
from .shortcuts import ShortcutManager

__all__ = [
    "ConfigManager",
    "ProfileManager",
    "SchemaManager",
    "PersistenceManager",
    "ShortcutManager",
]
