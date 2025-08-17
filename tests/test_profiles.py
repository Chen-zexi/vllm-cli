"""Tests for profile management."""

import json
from pathlib import Path
from unittest.mock import patch

from vllm_cli.config.profiles import ProfileManager


class TestProfileManager:
    """Test ProfileManager functionality."""

    def test_init(self, temp_config_dir):
        """Test ProfileManager initialization."""
        manager = ProfileManager(temp_config_dir)

        assert manager.config_dir == temp_config_dir
        assert manager.user_profiles_file == temp_config_dir / "user_profiles.json"

    @patch("vllm_cli.config.profiles.ProfileManager._load_default_profiles")
    def test_load_default_profiles(self, mock_load):
        """Test loading default profiles."""
        mock_profiles = {
            "standard": {"description": "Standard profile"},
            "high_throughput": {"max_model_len": 8192},
        }
        mock_load.return_value = mock_profiles

        manager = ProfileManager(Path("/tmp"))
        profiles = manager.default_profiles  # Access as property

        assert profiles == mock_profiles
        assert "standard" in profiles
        assert "high_throughput" in profiles

    def test_load_user_profiles_no_file(self, temp_config_dir):
        """Test loading user profiles when file doesn't exist."""
        manager = ProfileManager(temp_config_dir)
        profiles = manager.user_profiles  # Access as property

        assert profiles == {}

    def test_load_user_profiles_existing(self, temp_config_dir):
        """Test loading existing user profiles."""
        profiles_file = temp_config_dir / "user_profiles.json"
        user_profiles = {
            "custom1": {"gpu_memory_utilization": 0.8},
            "custom2": {"quantization": "awq"},
        }

        with open(profiles_file, "w") as f:
            json.dump(user_profiles, f)

        manager = ProfileManager(temp_config_dir)
        profiles = manager.user_profiles  # Access as property

        assert profiles == user_profiles
        assert "custom1" in profiles
        assert "custom2" in profiles

    def test_save_user_profile(self, temp_config_dir):
        """Test saving a new user profile."""
        manager = ProfileManager(temp_config_dir)

        profile_data = {"max_model_len": 4096, "gpu_memory_utilization": 0.7}

        result = manager.save_user_profile("test_profile", profile_data)
        assert result is True

        # Verify it was saved
        assert "test_profile" in manager.user_profiles
        assert manager.user_profiles["test_profile"] == profile_data

    def test_update_existing_profile(self, temp_config_dir):
        """Test updating an existing user profile."""
        manager = ProfileManager(temp_config_dir)

        # Save initial profile
        initial_data = {"port": 8000}
        manager.save_user_profile("test", initial_data)

        # Update profile
        updated_data = {"port": 9000, "quantization": "awq"}
        manager.save_user_profile("test", updated_data)

        # Verify update
        assert manager.user_profiles["test"] == updated_data

    def test_delete_user_profile(self, temp_config_dir):
        """Test deleting a user profile."""
        manager = ProfileManager(temp_config_dir)

        # Create profiles
        manager.save_user_profile("keep", {"port": 8000})
        manager.save_user_profile("delete", {"port": 9000})

        # Delete one
        result = manager.delete_user_profile("delete")

        assert result is True
        profiles = manager.user_profiles  # Use property instead of method
        assert "keep" in profiles
        assert "delete" not in profiles

    def test_delete_nonexistent_profile(self, temp_config_dir):
        """Test deleting non-existent profile."""
        manager = ProfileManager(temp_config_dir)

        result = manager.delete_user_profile("nonexistent")

        assert result is False

    def test_get_profile_existing(self, temp_config_dir):
        """Test getting an existing profile."""
        manager = ProfileManager(temp_config_dir)

        # Mock default profiles
        with patch.object(
            manager, "default_profiles", {"standard": {"description": "Standard"}}
        ):
            # Add user profile
            manager.save_user_profile("custom", {"port": 8080})

            # Get default profile
            profile = manager.get_profile("standard")
            assert profile == {"description": "Standard"}

            # Get user profile
            profile = manager.get_profile("custom")
            assert profile == {"port": 8080}

    def test_get_profile_nonexistent(self, temp_config_dir):
        """Test getting non-existent profile."""
        manager = ProfileManager(temp_config_dir)

        with patch.object(manager, "default_profiles", {}):
            profile = manager.get_profile("nonexistent")
            assert profile is None

    def test_get_all_profiles(self, temp_config_dir):
        """Test getting all available profiles."""
        manager = ProfileManager(temp_config_dir)

        # Mock default profiles
        with patch.object(
            manager,
            "default_profiles",
            {
                "standard": {"description": "Standard"},
                "high_throughput": {"max_model_len": 8192},
            },
        ):
            # Add user profiles
            manager.save_user_profile("custom1", {"port": 8080})
            manager.save_user_profile("custom2", {"port": 9090})

            all_profiles = manager.get_all_profiles()

            assert len(all_profiles) == 4
            assert "standard" in all_profiles
            assert "high_throughput" in all_profiles
            assert "custom1" in all_profiles
            assert "custom2" in all_profiles
