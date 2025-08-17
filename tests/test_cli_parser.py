"""Tests for CLI argument parsing."""

import pytest

from vllm_cli.cli.parser import create_parser


class TestCLIParser:
    """Test CLI argument parser."""

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "vllm-cli"

    def test_no_args_interactive_mode(self):
        """Test that no arguments defaults to interactive mode."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_serve_command_basic(self):
        """Test basic serve command."""
        parser = create_parser()
        args = parser.parse_args(["serve", "meta-llama/Llama-2-7b"])

        assert args.command == "serve"
        assert args.model == "meta-llama/Llama-2-7b"

    def test_serve_command_with_profile(self):
        """Test serve command with profile."""
        parser = create_parser()
        args = parser.parse_args(
            ["serve", "test-model", "--profile", "high_throughput"]
        )

        assert args.command == "serve"
        assert args.model == "test-model"
        assert args.profile == "high_throughput"

    def test_serve_command_with_port(self):
        """Test serve command with custom port."""
        parser = create_parser()
        args = parser.parse_args(["serve", "test-model", "--port", "9000"])

        assert args.command == "serve"
        assert args.port == 9000

    def test_serve_command_with_lora(self):
        """Test serve command with LoRA adapters."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "serve",
                "test-model",
                "--lora",
                "adapter1=/path/to/adapter1",
                "--lora",
                "adapter2=/path/to/adapter2",
            ]
        )

        assert args.command == "serve"
        assert len(args.lora) == 2
        assert "adapter1=/path/to/adapter1" in args.lora

    def test_serve_command_with_extra_args(self):
        """Test serve command with extra vLLM arguments."""
        parser = create_parser()
        args = parser.parse_args(
            ["serve", "test-model", "--extra-args", "--seed 42 --enable-prefix-caching"]
        )

        assert args.command == "serve"
        assert args.extra_args == "--seed 42 --enable-prefix-caching"

    def test_info_command(self):
        """Test info command."""
        parser = create_parser()
        args = parser.parse_args(["info"])

        assert args.command == "info"

    def test_models_command(self):
        """Test models command."""
        parser = create_parser()
        args = parser.parse_args(["models"])

        assert args.command == "models"

    def test_models_command_with_details(self):
        """Test models command with details flag."""
        parser = create_parser()
        args = parser.parse_args(["models", "--details"])

        assert args.command == "models"
        assert args.details is True

    def test_status_command(self):
        """Test status command."""
        parser = create_parser()
        args = parser.parse_args(["status"])

        assert args.command == "status"

    def test_stop_command_with_port(self):
        """Test stop command with port."""
        parser = create_parser()
        args = parser.parse_args(["stop", "--port", "8000"])

        assert args.command == "stop"
        assert args.port == 8000

    def test_stop_command_with_all(self):
        """Test stop command with all flag."""
        parser = create_parser()
        args = parser.parse_args(["stop", "--all"])

        assert args.command == "stop"
        assert args.all is True

    def test_version_flag(self):
        """Test version flag."""
        parser = create_parser()

        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--version"])

        # Version flag causes sys.exit(0)
        assert excinfo.value.code == 0

    def test_invalid_command(self):
        """Test invalid command handling."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["invalid-command"])

    def test_serve_requires_model(self):
        """Test that serve command requires model argument."""
        parser = create_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["serve"])  # Missing model
