#!/usr/bin/env python3
"""
Command Line Interface Package

This package provides the command line interface for vLLM CLI including
argument parsing, command handlers, and utility functions.

Main Components:
- Parser: Command line argument parsing and validation
- Handlers: Implementation of individual CLI commands (serve, info, models, status, stop)
- Utils: CLI utilities, formatting, and coordination functions

The CLI supports both direct command execution and interactive mode,
with rich formatting and comprehensive error handling.
"""

# Individual command handlers
from .handlers import (
    handle_info,
    handle_models,
    handle_serve,
    handle_status,
    handle_stop,
)

# Parser functions
from .parser import create_parser, parse_args

# CLI utilities
# Main CLI functions
from .utils import (
    format_cli_output,
    get_user_confirmation,
    handle_cli_command,
    main,
    print_cli_error,
    print_cli_info,
    print_cli_success,
    print_cli_warning,
    validate_cli_args,
)

__all__ = [
    # Main CLI functions
    "handle_cli_command",
    "main",
    # Parser
    "create_parser",
    "parse_args",
    # Command handlers
    "handle_serve",
    "handle_info",
    "handle_models",
    "handle_status",
    "handle_stop",
    # Utilities
    "validate_cli_args",
    "format_cli_output",
    "get_user_confirmation",
    "print_cli_error",
    "print_cli_warning",
    "print_cli_success",
    "print_cli_info",
]
