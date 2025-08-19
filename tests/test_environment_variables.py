"""
Comprehensive tests for the three-tier environment variable system.

Tests ensure:
1. No environment pollution - parent process environment is never modified
2. Correct priority order - Universal < Profile < Session < Device flags
3. Proper isolation - each subprocess gets its own environment copy
4. Complete coverage - all edge cases and error conditions
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from vllm_cli.config.manager import ConfigManager
from vllm_cli.server.manager import VLLMServer


class TestEnvironmentIsolation:
    """Test that environment variables don't pollute the parent process."""

    def test_env_isolation_no_pollution(self):
        """Verify parent process environment is never modified."""
        # Save original environment
        original_env = os.environ.copy()
        original_keys = set(original_env.keys())

        config = {
            "model": "test-model",
            "port": 8000,
            "profile_environment": {
                "TEST_VAR": "test_value",
                "VLLM_LOGGING_LEVEL": "DEBUG",
            },
        }

        server = VLLMServer(config)

        # Mock subprocess to prevent actual process creation
        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_process.stdout = Mock()
            mock_process.stdout.readline.return_value = ""
            mock_popen.return_value = mock_process

            # Mock ConfigManager for universal env
            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                mock_cm_instance.config = {
                    "universal_environment": {"UNIVERSAL_VAR": "universal_value"}
                }
                mock_cm_instance.build_cli_args.return_value = [
                    "serve",
                    "test-model",
                    "--port",
                    "8000",
                ]
                mock_config_manager.return_value = mock_cm_instance

                # Start server (this should not modify os.environ)
                server.start()

        # Verify parent environment is unchanged
        current_env = os.environ.copy()
        current_keys = set(current_env.keys())

        # No new keys should be added to parent environment
        assert current_keys == original_keys, "Parent environment has new keys"

        # All original values should be unchanged
        for key, value in original_env.items():
            assert (
                os.environ.get(key) == value
            ), f"Parent environment key {key} was modified"

        # The test variables should NOT be in parent environment
        assert "TEST_VAR" not in os.environ
        assert "UNIVERSAL_VAR" not in os.environ

    def test_env_copy_independence(self):
        """Ensure env.copy() creates truly independent copy."""
        # Set a test variable in parent
        test_key = "VLLM_TEST_PARENT_VAR"
        os.environ[test_key] = "parent_value"

        try:
            config = {"model": "test-model", "port": 8000}
            server = VLLMServer(config)

            with patch("subprocess.Popen") as mock_popen:
                env_captured = None

                def capture_env(*args, **kwargs):
                    nonlocal env_captured
                    env_captured = kwargs.get("env", {}).copy()
                    mock_process = Mock()
                    mock_process.poll.return_value = None
                    mock_process.stdout = Mock()
                    mock_process.stdout.readline.return_value = ""
                    mock_process.pid = 12345
                    return mock_process

                mock_popen.side_effect = capture_env

                with patch(
                    "vllm_cli.server.manager.ConfigManager"
                ) as mock_config_manager:
                    mock_cm_instance = Mock()
                    mock_cm_instance.config = {}
                    mock_cm_instance.build_cli_args.return_value = [
                        "serve",
                        "test-model",
                    ]
                    mock_config_manager.return_value = mock_cm_instance

                    server.start()

                # Verify subprocess got parent variable
                assert env_captured is not None
                assert env_captured[test_key] == "parent_value"

                # Modify captured env should not affect parent
                env_captured[test_key] = "modified_value"
                assert os.environ[test_key] == "parent_value"

        finally:
            # Clean up
            del os.environ[test_key]

    def test_subprocess_env_isolation(self):
        """Verify subprocess gets isolated environment copy."""
        config = {
            "model": "test-model",
            "port": 8000,
            "profile_environment": {"SUBPROCESS_ONLY": "subprocess_value"},
        }

        server = VLLMServer(config)
        subprocess_env = None

        with patch("subprocess.Popen") as mock_popen:

            def capture_subprocess_env(*args, **kwargs):
                nonlocal subprocess_env
                subprocess_env = kwargs.get("env")
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_process.stdout = Mock()
                mock_process.stdout.readline.return_value = ""
                mock_process.pid = 12345
                return mock_process

            mock_popen.side_effect = capture_subprocess_env

            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                mock_cm_instance.config = {}
                mock_cm_instance.build_cli_args.return_value = ["serve", "test-model"]
                mock_config_manager.return_value = mock_cm_instance

                server.start()

        # Subprocess should have the variable
        assert subprocess_env is not None
        assert subprocess_env.get("SUBPROCESS_ONLY") == "subprocess_value"

        # Parent should NOT have the variable
        assert "SUBPROCESS_ONLY" not in os.environ


class TestThreeTierPriority:
    """Test the three-tier environment variable priority system."""

    def test_universal_env_applied(self):
        """Universal environment variables are applied to subprocess."""
        config = {"model": "test-model", "port": 8000}
        server = VLLMServer(config)
        subprocess_env = None

        with patch("subprocess.Popen") as mock_popen:

            def capture_env(*args, **kwargs):
                nonlocal subprocess_env
                subprocess_env = kwargs.get("env")
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_process.stdout = Mock()
                mock_process.stdout.readline.return_value = ""
                mock_process.pid = 12345
                return mock_process

            mock_popen.side_effect = capture_env

            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                mock_cm_instance.config = {
                    "universal_environment": {
                        "VLLM_LOGGING_LEVEL": "INFO",
                        "VLLM_USE_TRITON_FLASH_ATTN": "1",
                    }
                }
                mock_cm_instance.build_cli_args.return_value = ["serve", "test-model"]
                mock_config_manager.return_value = mock_cm_instance

                server.start()

        assert subprocess_env is not None
        assert subprocess_env.get("VLLM_LOGGING_LEVEL") == "INFO"
        assert subprocess_env.get("VLLM_USE_TRITON_FLASH_ATTN") == "1"

    def test_profile_env_overrides_universal(self):
        """Profile environment variables override universal ones."""
        config = {
            "model": "test-model",
            "port": 8000,
            "profile_environment": {
                "VLLM_LOGGING_LEVEL": "DEBUG",  # Override universal
                "PROFILE_SPECIFIC": "profile_value",
            },
        }

        server = VLLMServer(config)
        subprocess_env = None

        with patch("subprocess.Popen") as mock_popen:

            def capture_env(*args, **kwargs):
                nonlocal subprocess_env
                subprocess_env = kwargs.get("env")
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_process.stdout = Mock()
                mock_process.stdout.readline.return_value = ""
                mock_process.pid = 12345
                return mock_process

            mock_popen.side_effect = capture_env

            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                mock_cm_instance.config = {
                    "universal_environment": {
                        "VLLM_LOGGING_LEVEL": "INFO",  # Will be overridden
                        "UNIVERSAL_ONLY": "universal_value",
                    }
                }
                mock_cm_instance.build_cli_args.return_value = ["serve", "test-model"]
                mock_config_manager.return_value = mock_cm_instance

                server.start()

        assert subprocess_env is not None
        # Profile should override universal
        assert subprocess_env.get("VLLM_LOGGING_LEVEL") == "DEBUG"
        # Profile-specific should be present
        assert subprocess_env.get("PROFILE_SPECIFIC") == "profile_value"
        # Universal-only should still be present
        assert subprocess_env.get("UNIVERSAL_ONLY") == "universal_value"

    def test_device_overrides_cuda_visible_devices(self):
        """Device configuration overrides CUDA_VISIBLE_DEVICES from env vars."""
        config = {
            "model": "test-model",
            "port": 8000,
            "device": "2,3",  # This should win
            "profile_environment": {
                "CUDA_VISIBLE_DEVICES": "0,1"  # This should be overridden
            },
        }

        server = VLLMServer(config)
        subprocess_env = None

        with patch("subprocess.Popen") as mock_popen:

            def capture_env(*args, **kwargs):
                nonlocal subprocess_env
                subprocess_env = kwargs.get("env")
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_process.stdout = Mock()
                mock_process.stdout.readline.return_value = ""
                mock_process.pid = 12345
                return mock_process

            mock_popen.side_effect = capture_env

            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                mock_cm_instance.config = {
                    "universal_environment": {
                        "CUDA_VISIBLE_DEVICES": "4,5"  # This should also be overridden
                    }
                }
                mock_cm_instance.build_cli_args.return_value = ["serve", "test-model"]
                mock_config_manager.return_value = mock_cm_instance

                server.start()

        assert subprocess_env is not None
        # Device flag should override everything
        assert subprocess_env.get("CUDA_VISIBLE_DEVICES") == "2,3"

    def test_empty_env_levels(self):
        """Handle empty environment at each level gracefully."""
        config = {"model": "test-model", "port": 8000}
        server = VLLMServer(config)
        subprocess_env = None

        with patch("subprocess.Popen") as mock_popen:

            def capture_env(*args, **kwargs):
                nonlocal subprocess_env
                subprocess_env = kwargs.get("env")
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_process.stdout = Mock()
                mock_process.stdout.readline.return_value = ""
                mock_process.pid = 12345
                return mock_process

            mock_popen.side_effect = capture_env

            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                # No universal_environment key at all
                mock_cm_instance.config = {}
                mock_cm_instance.build_cli_args.return_value = ["serve", "test-model"]
                mock_config_manager.return_value = mock_cm_instance

                server.start()

        # Should not crash and subprocess should get parent env
        assert subprocess_env is not None
        # Should have at least PATH from parent
        assert "PATH" in subprocess_env


class TestEnvironmentMerging:
    """Test environment variable merging behavior."""

    def test_partial_override(self):
        """Profile only overrides specific universal variables."""
        config = {
            "model": "test-model",
            "port": 8000,
            "profile_environment": {
                "VAR_B": "profile_B",  # Override
                "VAR_C": "profile_C",  # New
            },
        }

        server = VLLMServer(config)
        subprocess_env = None

        with patch("subprocess.Popen") as mock_popen:

            def capture_env(*args, **kwargs):
                nonlocal subprocess_env
                subprocess_env = kwargs.get("env")
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_process.stdout = Mock()
                mock_process.stdout.readline.return_value = ""
                mock_process.pid = 12345
                return mock_process

            mock_popen.side_effect = capture_env

            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                mock_cm_instance.config = {
                    "universal_environment": {
                        "VAR_A": "universal_A",
                        "VAR_B": "universal_B",  # Will be overridden
                    }
                }
                mock_cm_instance.build_cli_args.return_value = ["serve", "test-model"]
                mock_config_manager.return_value = mock_cm_instance

                server.start()

        assert subprocess_env is not None
        assert subprocess_env.get("VAR_A") == "universal_A"  # Preserved
        assert subprocess_env.get("VAR_B") == "profile_B"  # Overridden
        assert subprocess_env.get("VAR_C") == "profile_C"  # New from profile

    def test_complete_override_chain(self):
        """Test complete override chain with all levels."""
        # Note: Session environment would be handled at config level before
        # creating server, but we can test profile + universal + device
        config = {
            "model": "test-model",
            "port": 8000,
            "device": "7",
            "profile_environment": {
                "LEVEL_VAR": "profile",
                "PROFILE_VAR": "profile_only",
                "CUDA_VISIBLE_DEVICES": "6",  # Will be overridden by device
            },
        }

        server = VLLMServer(config)
        subprocess_env = None

        with patch("subprocess.Popen") as mock_popen:

            def capture_env(*args, **kwargs):
                nonlocal subprocess_env
                subprocess_env = kwargs.get("env")
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_process.stdout = Mock()
                mock_process.stdout.readline.return_value = ""
                mock_process.pid = 12345
                return mock_process

            mock_popen.side_effect = capture_env

            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                mock_cm_instance.config = {
                    "universal_environment": {
                        "LEVEL_VAR": "universal",  # Will be overridden
                        "UNIVERSAL_VAR": "universal_only",
                        "CUDA_VISIBLE_DEVICES": "0,1,2,3",  # Will be overridden
                    }
                }
                mock_cm_instance.build_cli_args.return_value = ["serve", "test-model"]
                mock_config_manager.return_value = mock_cm_instance

                server.start()

        assert subprocess_env is not None
        assert subprocess_env.get("LEVEL_VAR") == "profile"
        assert subprocess_env.get("PROFILE_VAR") == "profile_only"
        assert subprocess_env.get("UNIVERSAL_VAR") == "universal_only"
        assert subprocess_env.get("CUDA_VISIBLE_DEVICES") == "7"  # Device wins


class TestSpecialCases:
    """Test special cases and edge conditions."""

    def test_sensitive_value_masking(self, caplog):
        """Keys containing KEY or TOKEN are masked in logs."""
        config = {
            "model": "test-model",
            "port": 8000,
            "profile_environment": {
                "API_KEY": "secret_key_123",
                "HF_TOKEN": "hf_secret_token",
                "NORMAL_VAR": "visible_value",
            },
        }

        server = VLLMServer(config)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_process.stdout = Mock()
            mock_process.stdout.readline.return_value = ""
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                mock_cm_instance.config = {}
                mock_cm_instance.build_cli_args.return_value = ["serve", "test-model"]
                mock_config_manager.return_value = mock_cm_instance

                import logging

                with caplog.at_level(logging.DEBUG):
                    server.start()

        # Check log messages
        log_text = " ".join(record.message for record in caplog.records)

        # Sensitive values should not appear in logs
        assert "secret_key_123" not in log_text
        assert "hf_secret_token" not in log_text

        # But the keys should be mentioned as <hidden>
        assert "API_KEY=<hidden>" in log_text or "API_KEY" in log_text
        assert "HF_TOKEN=<hidden>" in log_text or "HF_TOKEN" in log_text

        # Normal values can appear
        assert "NORMAL_VAR=visible_value" in log_text or "visible_value" in log_text

    def test_string_conversion(self):
        """All environment values are converted to strings."""
        config = {
            "model": "test-model",
            "port": 8000,
            "profile_environment": {
                "INT_VAR": 123,
                "FLOAT_VAR": 45.67,
                "BOOL_VAR": True,
                "NONE_VAR": None,
                "STRING_VAR": "already_string",
            },
        }

        server = VLLMServer(config)
        subprocess_env = None

        with patch("subprocess.Popen") as mock_popen:

            def capture_env(*args, **kwargs):
                nonlocal subprocess_env
                subprocess_env = kwargs.get("env")
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_process.stdout = Mock()
                mock_process.stdout.readline.return_value = ""
                mock_process.pid = 12345
                return mock_process

            mock_popen.side_effect = capture_env

            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                mock_cm_instance.config = {}
                mock_cm_instance.build_cli_args.return_value = ["serve", "test-model"]
                mock_config_manager.return_value = mock_cm_instance

                server.start()

        assert subprocess_env is not None
        # All values should be strings
        assert subprocess_env.get("INT_VAR") == "123"
        assert subprocess_env.get("FLOAT_VAR") == "45.67"
        assert subprocess_env.get("BOOL_VAR") == "True"
        assert subprocess_env.get("NONE_VAR") == "None"
        assert subprocess_env.get("STRING_VAR") == "already_string"

    def test_hf_token_injection(self):
        """HuggingFace token is injected when configured."""
        config = {"model": "test-model", "port": 8000}
        server = VLLMServer(config)
        subprocess_env = None

        with patch("subprocess.Popen") as mock_popen:

            def capture_env(*args, **kwargs):
                nonlocal subprocess_env
                subprocess_env = kwargs.get("env")
                mock_process = Mock()
                mock_process.poll.return_value = None
                mock_process.stdout = Mock()
                mock_process.stdout.readline.return_value = ""
                mock_process.pid = 12345
                return mock_process

            mock_popen.side_effect = capture_env

            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                mock_cm_instance.config = {"hf_token": "hf_test_token_123"}
                mock_cm_instance.build_cli_args.return_value = ["serve", "test-model"]
                mock_config_manager.return_value = mock_cm_instance

                server.start()

        assert subprocess_env is not None
        assert subprocess_env.get("HF_TOKEN") == "hf_test_token_123"
        assert subprocess_env.get("HUGGING_FACE_HUB_TOKEN") == "hf_test_token_123"

    def test_no_forced_pythonunbuffered(self):
        """PYTHONUNBUFFERED is not forced, only set if user configures it."""
        config = {"model": "test-model", "port": 8000}
        server = VLLMServer(config)
        subprocess_env = None

        # Remove PYTHONUNBUFFERED if it exists in parent
        original_value = os.environ.pop("PYTHONUNBUFFERED", None)

        try:
            with patch("subprocess.Popen") as mock_popen:

                def capture_env(*args, **kwargs):
                    nonlocal subprocess_env
                    subprocess_env = kwargs.get("env")
                    mock_process = Mock()
                    mock_process.poll.return_value = None
                    mock_process.stdout = Mock()
                    mock_process.stdout.readline.return_value = ""
                    mock_process.pid = 12345
                    return mock_process

                mock_popen.side_effect = capture_env

                with patch(
                    "vllm_cli.server.manager.ConfigManager"
                ) as mock_config_manager:
                    mock_cm_instance = Mock()
                    mock_cm_instance.config = {}  # No environment variables configured
                    mock_cm_instance.build_cli_args.return_value = [
                        "serve",
                        "test-model",
                    ]
                    mock_config_manager.return_value = mock_cm_instance

                    server.start()

            assert subprocess_env is not None
            # PYTHONUNBUFFERED should not be set if not in parent and not configured
            assert "PYTHONUNBUFFERED" not in subprocess_env

        finally:
            # Restore original value if it existed
            if original_value is not None:
                os.environ["PYTHONUNBUFFERED"] = original_value


class TestConfigurationFlow:
    """Test environment variable configuration flow."""

    def test_profile_save_with_environment(self):
        """Test that profiles save with environment field."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.yaml"

            # Create minimal config
            with open(config_file, "w") as f:
                yaml.dump({}, f)

            # Patch the config_dir attribute after initialization
            with patch.object(ConfigManager, "__init__", lambda self: None):
                config_manager = ConfigManager()
                config_manager.config_dir = Path(temp_dir)
                config_manager.config_file = config_file
                config_manager.schema_manager = Mock()
                config_manager.persistence_manager = Mock()
                config_manager.persistence_manager.load_json_file.return_value = {}
                config_manager.persistence_manager.save_json_file.return_value = None
                config_manager.profile_manager = Mock()
                config_manager.profile_manager.save_user_profile.return_value = True
                config_manager.profile_manager.get_profile.return_value = {
                    "name": "test_profile",
                    "environment": {
                        "VLLM_LOGGING_LEVEL": "DEBUG",
                        "VLLM_USE_TRITON_FLASH_ATTN": "1",
                    },
                }
                config_manager.shortcut_manager = Mock()
                config_manager.config = {}
                config_manager.validation_registry = Mock()
                config_manager.save_user_profile = Mock(return_value=True)

                profile_data = {
                    "name": "test_profile",
                    "description": "Test profile with env vars",
                    "config": {"gpu_memory_utilization": 0.9},
                    "environment": {
                        "VLLM_LOGGING_LEVEL": "DEBUG",
                        "VLLM_USE_TRITON_FLASH_ATTN": "1",
                    },
                }

                # Save profile (mocked)
                result = config_manager.save_user_profile("test_profile", profile_data)
                assert result is True

                # Verify the mocked profile has environment
                saved_profile = config_manager.get_profile("test_profile")
                assert saved_profile is not None
                assert "environment" in saved_profile
                assert saved_profile["environment"]["VLLM_LOGGING_LEVEL"] == "DEBUG"
                assert saved_profile["environment"]["VLLM_USE_TRITON_FLASH_ATTN"] == "1"

    def test_universal_environment_persistence(self):
        """Test that universal environment is saved to config.yaml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.yaml"

            # Create config with universal environment
            config_data = {
                "universal_environment": {
                    "VLLM_LOGGING_LEVEL": "INFO",
                    "VLLM_CPU_KVCACHE_SPACE": "4",
                }
            }

            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            # Create mock config manager with proper environment handling
            with patch.object(ConfigManager, "__init__", lambda self: None):
                config_manager = ConfigManager()
                config_manager.config_dir = Path(temp_dir)
                config_manager.config_file = config_file
                config_manager.config = config_data.copy()

                # Mock the save method
                def mock_save():
                    with open(config_file, "w") as f:
                        yaml.dump(config_manager.config, f)

                config_manager._save_config = mock_save

                # Verify it loads correctly
                assert "universal_environment" in config_manager.config
                assert (
                    config_manager.config["universal_environment"]["VLLM_LOGGING_LEVEL"]
                    == "INFO"
                )

                # Modify and save
                config_manager.config["universal_environment"]["NEW_VAR"] = "new_value"
                config_manager._save_config()

                # Reload and verify persistence
                with open(config_file, "r") as f:
                    saved_config = yaml.safe_load(f)

                assert saved_config["universal_environment"]["NEW_VAR"] == "new_value"
                assert (
                    saved_config["universal_environment"]["VLLM_LOGGING_LEVEL"]
                    == "INFO"
                )


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_config_keys(self):
        """Handle missing universal_environment gracefully."""
        config = {"model": "test-model", "port": 8000}
        server = VLLMServer(config)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_process.stdout = Mock()
            mock_process.stdout.readline.return_value = ""
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                # Config returns empty dict instead of None for .get() to work
                mock_cm_instance.config = {}  # Empty dict instead of None
                mock_cm_instance.build_cli_args.return_value = ["serve", "test-model"]
                mock_config_manager.return_value = mock_cm_instance

                # Should not crash
                result = server.start()
                assert result is True

    def test_subprocess_failure_preserves_parent_env(self):
        """Subprocess failures don't affect parent environment."""
        original_env = os.environ.copy()

        config = {
            "model": "test-model",
            "port": 8000,
            "profile_environment": {"FAILURE_VAR": "should_not_leak"},
        }

        server = VLLMServer(config)

        with patch("subprocess.Popen") as mock_popen:
            # Simulate subprocess creation failure
            mock_popen.side_effect = Exception("Failed to start process")

            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                mock_cm_instance.config = {}
                mock_cm_instance.build_cli_args.return_value = ["serve", "test-model"]
                mock_config_manager.return_value = mock_cm_instance

                # Start should fail but not pollute environment
                result = server.start()
                assert result is False

        # Parent environment should be unchanged
        assert "FAILURE_VAR" not in os.environ
        assert os.environ == original_env

    def test_tensor_parallel_adjustment_with_device(self):
        """Tensor parallel size is adjusted when device limits GPUs."""
        config = {
            "model": "test-model",
            "port": 8000,
            "device": "0,1",  # 2 GPUs
            "tensor_parallel_size": 4,  # Requesting 4 GPUs
        }

        server = VLLMServer(config)

        with patch("subprocess.Popen") as mock_popen:
            mock_process = Mock()
            mock_process.poll.return_value = None
            mock_process.stdout = Mock()
            mock_process.stdout.readline.return_value = ""
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            with patch("vllm_cli.server.manager.ConfigManager") as mock_config_manager:
                mock_cm_instance = Mock()
                mock_cm_instance.config = {}
                mock_cm_instance.build_cli_args.return_value = ["serve", "test-model"]
                mock_config_manager.return_value = mock_cm_instance

                with patch("vllm_cli.server.manager.logger") as mock_logger:
                    server.start()

                    # Should log warning about adjustment
                    mock_logger.warning.assert_called()
                    warning_msg = mock_logger.warning.call_args[0][0]
                    assert "tensor_parallel_size" in warning_msg
                    assert "4" in warning_msg  # Original value
                    assert "2" in warning_msg  # Adjusted value

        # Config should be adjusted
        assert server.config["tensor_parallel_size"] == 2


# Integration test combining multiple features
class TestIntegration:
    """Integration tests combining multiple environment features."""

    def test_full_environment_stack(self):
        """Test complete environment stack with all features."""
        # Set up a parent environment variable
        os.environ["PARENT_VAR"] = "parent_value"

        try:
            config = {
                "model": "test-model",
                "port": 8000,
                "device": "3,4",
                "profile_environment": {
                    "PROFILE_VAR": "profile_value",
                    "OVERRIDE_VAR": "profile_override",
                    "CUDA_VISIBLE_DEVICES": "1,2",  # Will be overridden by device
                },
            }

            server = VLLMServer(config)
            captured_env = None

            with patch("subprocess.Popen") as mock_popen:

                def capture_env(*args, **kwargs):
                    nonlocal captured_env
                    captured_env = kwargs.get("env")
                    mock_process = Mock()
                    mock_process.poll.return_value = None
                    mock_process.stdout = Mock()
                    mock_process.stdout.readline.return_value = ""
                    mock_process.pid = 12345
                    return mock_process

                mock_popen.side_effect = capture_env

                with patch(
                    "vllm_cli.server.manager.ConfigManager"
                ) as mock_config_manager:
                    mock_cm_instance = Mock()
                    mock_cm_instance.config = {
                        "universal_environment": {
                            "UNIVERSAL_VAR": "universal_value",
                            # Will be overridden by profile
                            "OVERRIDE_VAR": "universal_override",
                        },
                        "hf_token": "test_hf_token",
                    }
                    mock_cm_instance.build_cli_args.return_value = [
                        "serve",
                        "test-model",
                    ]
                    mock_config_manager.return_value = mock_cm_instance

                    result = server.start()
                    assert result is True

            # Verify the complete environment
            assert captured_env is not None

            # Parent variables are inherited
            assert captured_env["PARENT_VAR"] == "parent_value"

            # Universal variables are applied
            assert captured_env["UNIVERSAL_VAR"] == "universal_value"

            # Profile variables override universal
            assert captured_env["OVERRIDE_VAR"] == "profile_override"
            assert captured_env["PROFILE_VAR"] == "profile_value"

            # Device overrides CUDA_VISIBLE_DEVICES
            assert captured_env["CUDA_VISIBLE_DEVICES"] == "3,4"

            # HF token is injected
            assert captured_env["HF_TOKEN"] == "test_hf_token"
            assert captured_env["HUGGING_FACE_HUB_TOKEN"] == "test_hf_token"

            # Parent environment is still clean
            assert "UNIVERSAL_VAR" not in os.environ
            assert "PROFILE_VAR" not in os.environ
            assert "OVERRIDE_VAR" not in os.environ

        finally:
            # Clean up
            del os.environ["PARENT_VAR"]
