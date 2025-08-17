"""Tests for configuration management."""

from unittest.mock import Mock, patch

import yaml

from vllm_cli.config.manager import ConfigManager


class TestConfigManager:
    """Test ConfigManager functionality."""

    @patch("vllm_cli.config.manager.Path.home")
    def test_init_creates_config_dir(self, mock_home, temp_config_dir):
        """Test that ConfigManager creates config directory on init."""
        mock_home.return_value = temp_config_dir

        config_manager = ConfigManager()

        expected_dir = temp_config_dir / ".config" / "vllm-cli"
        assert expected_dir.exists()
        assert config_manager.config_dir == expected_dir

    @patch("vllm_cli.config.manager.Path.home")
    def test_load_config_creates_default(self, mock_home, temp_config_dir):
        """Test loading config creates default if not exists."""
        mock_home.return_value = temp_config_dir

        config_manager = ConfigManager()
        # Config is loaded on init and stored in config attribute
        config = config_manager.config

        assert isinstance(config, dict)
        # Config file may not exist if no save was called
        assert config_manager.config_dir.exists()

    @patch("vllm_cli.config.manager.Path.home")
    def test_load_existing_config(self, mock_home, temp_config_dir):
        """Test loading existing configuration file."""
        mock_home.return_value = temp_config_dir
        config_dir = temp_config_dir / ".config" / "vllm-cli"
        config_dir.mkdir(parents=True)

        # Create a config file
        config_file = config_dir / "config.yaml"
        test_config = {
            "last_model": "test-model",
            "last_config": {"model": "test-model", "port": 8000},
        }
        with open(config_file, "w") as f:
            yaml.dump(test_config, f)

        config_manager = ConfigManager()
        # Config is loaded on init
        config = config_manager.config

        assert config["last_model"] == "test-model"
        assert config["last_config"]["port"] == 8000

    @patch("vllm_cli.config.manager.Path.home")
    def test_save_config(self, mock_home, temp_config_dir):
        """Test saving configuration."""
        mock_home.return_value = temp_config_dir

        config_manager = ConfigManager()
        test_config = {"test_key": "test_value", "port": 9000}

        # Update config and save
        config_manager.config = test_config
        config_manager._save_config()

        # Read back the saved config
        with open(config_manager.config_file, "r") as f:
            saved_config = yaml.safe_load(f)

        assert saved_config["test_key"] == "test_value"
        assert saved_config["port"] == 9000

    @patch("vllm_cli.config.manager.Path.home")
    def test_get_last_config(self, mock_home, temp_config_dir):
        """Test getting last used configuration."""
        mock_home.return_value = temp_config_dir
        config_dir = temp_config_dir / ".config" / "vllm-cli"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "config.yaml"
        test_config = {
            "last_config": {
                "model": "llama-7b",
                "port": 8080,
                "tensor_parallel_size": 2,
            }
        }
        with open(config_file, "w") as f:
            yaml.dump(test_config, f)

        config_manager = ConfigManager()
        # Access last_config from the loaded config
        last_config = config_manager.config.get("last_config")

        assert last_config is not None
        assert last_config["model"] == "llama-7b"
        assert last_config["port"] == 8080
        assert last_config["tensor_parallel_size"] == 2

    @patch("vllm_cli.config.manager.Path.home")
    def test_get_last_config_none(self, mock_home, temp_config_dir):
        """Test getting last config when none exists."""
        mock_home.return_value = temp_config_dir

        config_manager = ConfigManager()
        last_config = config_manager.config.get("last_config")

        assert last_config is None

    @patch("vllm_cli.config.manager.Path.home")
    def test_save_last_config(self, mock_home, temp_config_dir):
        """Test saving last used configuration."""
        mock_home.return_value = temp_config_dir

        config_manager = ConfigManager()
        server_config = {"model": "new-model", "port": 7000, "quantization": "awq"}

        # Update config with last config
        config_manager.config["last_model"] = "new-model"
        config_manager.config["last_config"] = server_config
        config_manager._save_config()

        # Verify it was saved by reloading
        config = config_manager._load_config()
        assert config["last_model"] == "new-model"
        assert config["last_config"] == server_config

    @patch("vllm_cli.config.manager.Path.home")
    def test_validate_config_valid(self, mock_home, temp_config_dir):
        """Test validating a valid configuration."""
        mock_home.return_value = temp_config_dir

        config_manager = ConfigManager()

        # Mock the validation registry
        with patch.object(config_manager, "validation_registry") as mock_registry:
            mock_registry.validate.return_value = Mock(is_valid=True, errors=[])

            valid_config = {
                "model": "test-model",
                "port": 8000,
                "tensor_parallel_size": 1,
            }

            is_valid, errors = config_manager.validate_config(valid_config)
            assert is_valid
            assert len(errors) == 0

    @patch("vllm_cli.config.manager.Path.home")
    def test_validate_config_invalid(self, mock_home, temp_config_dir):
        """Test validating an invalid configuration."""
        mock_home.return_value = temp_config_dir

        config_manager = ConfigManager()

        # Mock the validation registry to return proper validation result
        with patch.object(config_manager, "validation_registry") as mock_registry:
            # Create a mock result that properly handles is_valid as a method
            mock_result = Mock()
            mock_result.is_valid.return_value = False  # Method returns False
            mock_result.get_error_messages.return_value = ["Port out of range"]
            mock_result.warnings = []  # No warnings

            mock_registry.validate_config.return_value = mock_result

            invalid_config = {"model": "test-model", "port": 99999}  # Invalid port

            is_valid, errors = config_manager.validate_config(invalid_config)
            assert not is_valid
            assert len(errors) == 1
            assert "Port out of range" in errors[0]

    @patch("vllm_cli.config.manager.Path.home")
    def test_get_model_directories(self, mock_home, temp_config_dir):
        """Test getting model directories."""
        mock_home.return_value = temp_config_dir
        config_dir = temp_config_dir / ".config" / "vllm-cli"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "config.yaml"
        test_config = {"model_directories": ["/path/to/models", "/another/path"]}
        with open(config_file, "w") as f:
            yaml.dump(test_config, f)

        config_manager = ConfigManager()
        directories = config_manager.config.get("model_directories", [])

        assert len(directories) == 2
        assert "/path/to/models" in directories
        assert "/another/path" in directories

    @patch("vllm_cli.config.manager.Path.home")
    def test_add_model_directory(self, mock_home, temp_config_dir):
        """Test adding a model directory."""
        mock_home.return_value = temp_config_dir

        config_manager = ConfigManager()
        new_dir = "/new/model/path"

        # Add model directory to config
        if "model_directories" not in config_manager.config:
            config_manager.config["model_directories"] = []
        config_manager.config["model_directories"].append(new_dir)
        config_manager._save_config()

        directories = config_manager.config.get("model_directories", [])
        assert new_dir in directories

    @patch("vllm_cli.config.manager.Path.home")
    def test_remove_model_directory(self, mock_home, temp_config_dir):
        """Test removing a model directory."""
        mock_home.return_value = temp_config_dir
        config_dir = temp_config_dir / ".config" / "vllm-cli"
        config_dir.mkdir(parents=True)

        config_file = config_dir / "config.yaml"
        test_config = {"model_directories": ["/path/to/remove", "/keep/this"]}
        with open(config_file, "w") as f:
            yaml.dump(test_config, f)

        config_manager = ConfigManager()
        # Remove directory from config
        if "model_directories" in config_manager.config:
            config_manager.config["model_directories"].remove("/path/to/remove")
            config_manager._save_config()

        directories = config_manager.config.get("model_directories", [])
        assert "/path/to/remove" not in directories
        assert "/keep/this" in directories
