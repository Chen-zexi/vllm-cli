#!/usr/bin/env python3
"""
Command line argument parser for vLLM CLI.

Defines the argument parser structure and command definitions
for the vLLM CLI interface.
"""
import argparse
import logging

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser for vLLM CLI.

    Sets up the main parser with all subcommands and their arguments.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="vllm-cli",
        description="vLLM CLI - Convenient LLM serving tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  vllm-cli                                      # Interactive mode
  vllm-cli serve MODEL --profile standard       # Serve with profile
  vllm-cli serve MODEL --quantization awq       # Serve with AWQ quantization
  vllm-cli serve MODEL --extra-args "--seed 42" # With custom vLLM args
  vllm-cli info                                 # System information
  vllm-cli models                               # List available models
  vllm-cli status                               # Show active servers""",
    )

    # Global options
    from .. import __version__

    parser.add_argument(
        "--version", action="version", version=f"vLLM CLI {__version__}"
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="{serve,info,models,status,stop}",
    )

    # Add individual command parsers
    _add_serve_parser(subparsers)
    _add_info_parser(subparsers)
    _add_models_parser(subparsers)
    _add_status_parser(subparsers)
    _add_stop_parser(subparsers)
    _add_dirs_parser(subparsers)

    return parser


def _add_serve_parser(subparsers) -> None:
    """Add the 'serve' subcommand parser."""
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start a vLLM server",
        description="Start a vLLM server with the specified model and configuration",
    )

    # Required positional argument
    serve_parser.add_argument("model", help="Model name or path to serve")

    # Configuration options
    serve_parser.add_argument(
        "--profile",
        help="Configuration profile to use (e.g., standard, performance, memory-optimized)",
    )

    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to serve on (default: 8000)"
    )

    serve_parser.add_argument(
        "--host", default="localhost", help="Host to bind to (default: localhost)"
    )

    # vLLM-specific options
    serve_parser.add_argument(
        "--quantization",
        choices=["awq", "gptq", "bitsandbytes", "fp8"],
        help="Quantization method to use",
    )

    serve_parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Tensor parallel size for multi-GPU serving (auto-detected if not specified)",
    )

    serve_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0, default: 0.9)",
    )

    serve_parser.add_argument(
        "--max-model-len", type=int, help="Maximum model sequence length"
    )

    serve_parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model data type (default: auto)",
    )

    # LoRA adapter options
    serve_parser.add_argument(
        "--lora",
        action="append",
        help="LoRA adapter to load (can specify multiple times). Format: name=path or just path",
    )

    serve_parser.add_argument(
        "--enable-lora",
        action="store_true",
        help="Enable LoRA adapter support (auto-enabled if --lora is used)",
    )

    # Authentication options
    serve_parser.add_argument(
        "--hf-token",
        help="HuggingFace token for accessing gated models (can also set via Settings menu)",
    )

    # Advanced options
    serve_parser.add_argument(
        "--extra-args",
        help='Additional vLLM arguments as string (e.g., "--seed 42 --temperature 0.8")',
    )

    serve_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start in interactive configuration mode",
    )

    serve_parser.add_argument(
        "--save-profile", help="Save current configuration as a profile with this name"
    )


def _add_info_parser(subparsers) -> None:
    """Add the 'info' subcommand parser."""
    info_parser = subparsers.add_parser(
        "info",
        help="Show system information",
        description="Display system information including GPU status, memory, and software versions",
    )

    info_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )


def _add_models_parser(subparsers) -> None:
    """Add the 'models' subcommand parser."""
    models_parser = subparsers.add_parser(
        "models",
        help="List available models",
        description="List all available models that can be served",
    )

    models_parser.add_argument(
        "--search", help="Search for models matching the given pattern"
    )

    models_parser.add_argument(
        "--details", action="store_true", help="Show detailed model information"
    )

    models_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )


def _add_status_parser(subparsers) -> None:
    """Add the 'status' subcommand parser."""
    status_parser = subparsers.add_parser(
        "status",
        help="Show active servers",
        description="Display status of all running vLLM servers",
    )

    status_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )

    status_parser.add_argument(
        "--monitor", action="store_true", help="Continuously monitor server status"
    )


def _add_stop_parser(subparsers) -> None:
    """Add the 'stop' subcommand parser."""
    stop_parser = subparsers.add_parser(
        "stop", help="Stop a vLLM server", description="Stop one or more vLLM servers"
    )

    # Make model and options mutually exclusive
    stop_group = stop_parser.add_mutually_exclusive_group(required=True)

    stop_group.add_argument(
        "model", nargs="?", help="Model name or port number of server to stop"
    )

    stop_group.add_argument(
        "--all", action="store_true", help="Stop all running servers"
    )

    stop_group.add_argument(
        "--port", type=int, help="Stop server running on specific port"
    )


def _add_dirs_parser(subparsers) -> None:
    """Add the 'dirs' subcommand parser for directory management."""
    dirs_parser = subparsers.add_parser(
        "dirs",
        help="Manage model directories",
        description="Manage directories for model and LoRA adapter scanning",
    )

    # Create subcommands for directory operations
    dirs_subparsers = dirs_parser.add_subparsers(
        dest="dirs_command",
        help="Directory management commands",
        metavar="{add,remove,list}",
    )

    # Add directory
    add_parser = dirs_subparsers.add_parser(
        "add",
        help="Add a directory for model/LoRA scanning",
        description="Add a new directory to scan for models and LoRA adapters",
    )
    add_parser.add_argument("path", help="Path to directory to add")
    add_parser.add_argument(
        "--type",
        choices=["huggingface", "custom", "lora", "auto"],
        default="auto",
        help="Type of directory (default: auto-detect)",
    )

    # Remove directory
    remove_parser = dirs_subparsers.add_parser(
        "remove",
        help="Remove a directory from scanning",
        description="Remove a directory from model/LoRA scanning",
    )
    remove_parser.add_argument("path", help="Path to directory to remove")

    # List directories
    dirs_subparsers.add_parser(
        "list",
        help="List all configured directories",
        description="Show all directories configured for model/LoRA scanning",
    )


def parse_args(args=None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        args: Optional list of arguments (uses sys.argv if None)

    Returns:
        Parsed arguments namespace
    """
    parser = create_parser()
    return parser.parse_args(args)
