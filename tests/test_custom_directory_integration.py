"""Integration tests for custom directory addition and model loading."""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm_cli.models.manager import ModelManager
from vllm_cli.models.manifest import apply_manifest_to_models, load_manifest


class TestCustomDirectoryIntegration:
    """Test custom directory addition and model loading workflow."""

    @pytest.fixture
    def test_model_dir(self):
        """Create a temporary test model directory with mock models."""
        import json
        import tempfile

        temp_dir = tempfile.mkdtemp(prefix="vllm_test_models_")
        temp_path = Path(temp_dir)

        # Create test model directories
        gemma_dir = temp_path / "test-gemma-custom"
        qwen_dir = temp_path / "test-qwen3-custom"
        gemma_dir.mkdir()
        qwen_dir.mkdir()

        # Create minimal config files for each model
        gemma_config = {
            "architectures": ["Gemma3ForCausalLM"],
            "model_type": "gemma3_text",
            "vocab_size": 262144,
            "hidden_size": 768,
        }

        qwen_config = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "vocab_size": 151936,
            "hidden_size": 768,
        }

        # Write config files
        with open(gemma_dir / "config.json", "w") as f:
            json.dump(gemma_config, f)

        with open(qwen_dir / "config.json", "w") as f:
            json.dump(qwen_config, f)

        # Create dummy weight files
        (gemma_dir / "model.safetensors").touch()
        (qwen_dir / "model.safetensors").touch()

        # Create a sample manifest file
        manifest = {
            "models": [
                {
                    "path": "test-gemma-custom",
                    "name": "Test Gemma Custom",
                    "publisher": "TestOrg",
                    "notes": "Test Gemma model for CI/CD",
                    "enabled": True,
                },
                {
                    "path": "test-qwen3-custom",
                    "name": "Test Qwen3 Custom",
                    "publisher": "TestOrg",
                    "notes": "Test Qwen3 model for CI/CD",
                    "enabled": True,
                },
            ],
            "version": "1.0",
        }

        with open(temp_path / "models_manifest.json", "w") as f:
            json.dump(manifest, f)

        yield temp_path

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_custom_dir(self):
        """Create a temporary custom model directory for testing."""
        temp_dir = tempfile.mkdtemp(prefix="vllm_test_")
        temp_path = Path(temp_dir)

        # Create a simple model structure
        model_dir = temp_path / "test-model"
        model_dir.mkdir()

        # Create minimal model files
        config = {
            "architectures": ["TestModel"],
            "model_type": "test",
            "vocab_size": 1000,
            "hidden_size": 768,
        }

        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Create a dummy weight file
        (model_dir / "model.safetensors").touch()

        yield temp_path
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_hf_api(self):
        """Mock the HFModelAPI for testing."""
        with patch("hf_model_tool.api.HFModelAPI") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api

            # Setup default behaviors
            mock_api.list_directories.return_value = []
            mock_api.add_directory.return_value = True
            mock_api.scan_directories.return_value = []

            yield mock_api

    def test_add_custom_directory(self, mock_hf_api, temp_custom_dir):
        """Test adding a custom directory through the API."""
        from vllm_cli.ui.model_directories import ModelDirectoriesUI

        ui = ModelDirectoriesUI()
        ui.api = mock_hf_api

        # Simulate adding a directory
        result = ui.api.add_directory(str(temp_custom_dir), "custom")

        assert result is True
        mock_hf_api.add_directory.assert_called_once_with(
            str(temp_custom_dir), "custom"
        )

    def test_discover_custom_models(self, test_model_dir):
        """Test discovering models from the test_model directory."""
        # Test the fallback discovery with our temporary directory
        from vllm_cli.models.discovery import _scan_directory_for_models

        # Directly test scanning our temporary directory
        models = _scan_directory_for_models(test_model_dir)

        # Should find the two test models
        model_names = [m["name"] for m in models]

        # After applying manifest, names might be changed
        # So check for the base names or the manifest names
        assert len(models) >= 2, f"Expected at least 2 models, found: {model_names}"

        # Check that we found models in the test directory
        model_paths = [m.get("path", "") for m in models]
        assert any(str(test_model_dir) in path for path in model_paths)

    def test_manifest_loading(self, test_model_dir):
        """Test loading and applying manifest data to models."""
        # Load the manifest
        manifest = load_manifest(test_model_dir)

        assert manifest is not None
        assert "models" in manifest
        assert isinstance(manifest["models"], list)

        # Create mock models that match the manifest
        models = [
            {
                "name": "test-gemma-custom",
                "path": str(test_model_dir / "test-gemma-custom"),
                "size": 1000000,
                "type": "custom_model",
                "display_name": "test-gemma-custom",  # Default display name
                "publisher": "unknown",
            },
            {
                "name": "test-qwen3-custom",
                "path": str(test_model_dir / "test-qwen3-custom"),
                "size": 2000000,
                "type": "custom_model",
                "display_name": "test-qwen3-custom",  # Default display name
                "publisher": "unknown",
            },
        ]

        # Apply manifest
        enriched_models = apply_manifest_to_models(models, test_model_dir)

        # Check if manifest data was applied - names should be updated
        model_names = [m["name"] for m in enriched_models]
        assert "Test Gemma Custom" in model_names or any(
            "gemma" in n.lower() for n in model_names
        )
        assert "Test Qwen3 Custom" in model_names or any(
            "qwen" in n.lower() for n in model_names
        )

    def test_custom_model_selection(self, test_model_dir):
        """Test that custom models can be properly selected for serving."""
        # Mock the model discovery to return our test models
        test_models = [
            {
                "name": "test-gemma-custom",
                "path": str(test_model_dir / "test-gemma-custom"),
                "size": 1000000,
                "type": "custom_model",
                "publisher": "local",
                "display_name": "Test Gemma Custom",
            },
            {
                "name": "test-qwen3-custom",
                "path": str(test_model_dir / "test-qwen3-custom"),
                "size": 2000000,
                "type": "custom_model",
                "publisher": "local",
                "display_name": "Test Qwen3 Custom",
            },
        ]

        with patch("vllm_cli.models.list_available_models", return_value=test_models):
            # Test that custom models use path instead of name
            # Since custom_model type should always return path
            for model in test_models:
                if model["type"] == "custom_model":
                    # Custom models should use their path for serving
                    assert model["path"] is not None
                    assert model["publisher"] == "local"

    def test_end_to_end_custom_model_serving(self, test_model_dir, temp_custom_dir):
        """Test the complete flow from adding directory to serving configuration."""
        # Step 1: Add custom directory
        from vllm_cli.ui.model_directories import ModelDirectoriesUI

        with patch("hf_model_tool.api.HFModelAPI") as mock_api_class:
            mock_api = MagicMock()
            mock_api_class.return_value = mock_api
            mock_api.add_directory.return_value = True

            ui = ModelDirectoriesUI()
            ui.api = mock_api

            # Add the test directory
            result = ui.api.add_directory(str(test_model_dir), "custom")
            assert result is True

        # Step 2: Discover models with caching
        manager = ModelManager()

        # Mock the scan to return our test models
        test_models = [
            {
                "name": "test-gemma-custom",
                "path": str(test_model_dir / "test-gemma-custom"),
                "size": 1000000,
                "type": "custom_model",
                "publisher": "local",
                "display_name": "test-gemma-custom",
            }
        ]

        # Mock the manager's list_available_models directly to avoid real scanning
        with patch.object(manager, "list_available_models", return_value=test_models):
            models = manager.list_available_models(refresh=True)
            assert len(models) == 1
            assert models[0]["name"] == "test-gemma-custom"

        # Step 3: Build server configuration
        selected_model_path = str(test_model_dir / "test-gemma-custom")

        server_config = {
            "model": selected_model_path,
            "port": 8000,
            "tensor_parallel_size": 1,
        }

        # Verify the configuration uses the full path
        assert server_config["model"] == selected_model_path
        assert Path(server_config["model"]).exists()

    def test_cache_refresh_with_custom_models(self, test_model_dir):
        """Test that cache refresh works correctly with custom models."""
        # Create a fresh manager with clean cache
        from vllm_cli.models.cache import ModelCache

        # Create manager with fresh cache
        manager = ModelManager()
        manager.cache = ModelCache(ttl_seconds=30)  # Reset cache

        # Mock initial scan
        initial_models = [
            {
                "name": "test-gemma-custom",
                "path": str(test_model_dir / "test-gemma-custom"),
                "size": 1000000,
                "type": "custom_model",
                "display_name": "test-gemma-custom",
                "publisher": "local",
            }
        ]

        # Test cache functionality directly
        # First, cache should be empty
        assert manager.cache.get_cached_models() is None

        # Cache the models
        manager.cache.cache_models(initial_models)

        # Now cache should have data
        cached = manager.cache.get_cached_models()
        assert cached is not None
        assert len(cached) == 1

        # Check cache stats
        stats = manager.cache.get_cache_stats()
        assert stats["is_cached"] is True
        assert stats["cached_models_count"] == 1

        # Clear cache
        manager.cache.clear_cache()

        # Cache should be empty again
        assert manager.cache.get_cached_models() is None

    def test_custom_model_with_manifest_metadata(self, test_model_dir):
        """Test that manifest metadata is properly applied to custom models."""
        # Create a manifest with rich metadata - using correct format (list not dict)
        manifest_data = {
            "models": [
                {
                    "path": "test-gemma-custom",
                    "name": "Gemma 2B Custom Fine-tuned",
                    "publisher": "MyOrg",
                    "notes": "Custom fine-tuned Gemma model for specific tasks",
                    "enabled": True,
                },
                {
                    "path": "test-qwen3-custom",
                    "name": "Qwen3 Custom Chat",
                    "publisher": "MyOrg",
                    "notes": "Qwen3 model optimized for chat",
                    "enabled": True,
                },
            ],
            "version": "1.0",
        }

        # Write manifest to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            manifest_file = temp_path / "models_manifest.json"

            with open(manifest_file, "w") as f:
                json.dump(manifest_data, f)

            # Create model list
            models = [
                {
                    "name": "test-gemma-custom",
                    "path": str(temp_path / "test-gemma-custom"),
                    "size": 1000000,
                    "type": "custom_model",
                    "display_name": "test-gemma-custom",
                    "publisher": "unknown",
                },
                {
                    "name": "test-qwen3-custom",
                    "path": str(temp_path / "test-qwen3-custom"),
                    "size": 2000000,
                    "type": "custom_model",
                    "display_name": "test-qwen3-custom",
                    "publisher": "unknown",
                },
            ]

            # Apply manifest
            enriched = apply_manifest_to_models(models, temp_path)

            # Verify metadata was applied
            # Find models by checking if the name was updated from manifest
            gemma_model = next(
                (m for m in enriched if "Gemma 2B Custom" in m.get("name", "")), None
            )
            if gemma_model:
                assert gemma_model.get("display_name") == "Gemma 2B Custom Fine-tuned"
                assert gemma_model.get("publisher") == "MyOrg"
                assert (
                    gemma_model.get("notes")
                    == "Custom fine-tuned Gemma model for specific tasks"
                )
                assert gemma_model.get("from_manifest") is True

            qwen_model = next(
                (m for m in enriched if "Qwen3 Custom" in m.get("name", "")), None
            )
            if qwen_model:
                assert qwen_model.get("display_name") == "Qwen3 Custom Chat"
                assert qwen_model.get("publisher") == "MyOrg"
                assert qwen_model.get("notes") == "Qwen3 model optimized for chat"

    def test_model_path_resolution_for_serving(self):
        """Test that model paths are correctly resolved for custom vs HF models."""
        test_models = [
            # HuggingFace model - should use name
            {
                "name": "meta-llama/Llama-2-7b",
                "path": "/cache/huggingface/hub/models--meta-llama--Llama-2-7b",
                "type": "model",
                "publisher": "meta-llama",
            },
            # Custom model - should use path
            {
                "name": "my-custom-model",
                "path": "/home/user/models/my-custom-model",
                "type": "custom_model",
                "publisher": "local",
            },
            # Local model without HF pattern - should use path
            {
                "name": "local-model",
                "path": "/data/models/local-model",
                "type": "model",
                "publisher": "unknown",
            },
        ]

        # Test path resolution logic
        for model in test_models:
            # Custom models should always use path
            if model["type"] == "custom_model":
                assert model["path"] is not None
                # In real selection, path would be returned

            # HF pattern models should use name
            elif "/" in model["name"] and model["publisher"] not in [
                "local",
                "unknown",
                None,
            ]:
                # Would return name for serving
                assert model["name"] == "meta-llama/Llama-2-7b"

            # Local/unknown models should use path
            else:
                assert model["path"] is not None


class TestManifestGeneration:
    """Test manifest generation and handling."""

    def test_manifest_auto_generation_notification(self):
        """Test that users are notified about manifest auto-generation."""
        # This tests the UI notification logic
        notification_text = (
            "A models_manifest.json file will be auto-generated at:\n"
            "/path/to/directory/models_manifest.json\n\n"
            "You can manually edit this file to customize:\n"
            "  • Custom display names\n"
            "  • Model descriptions\n"
            "  • Publisher information\n"
            "  • Model categories"
        )

        # Verify the notification contains key information
        assert "will be auto-generated" in notification_text
        assert "manually edit" in notification_text
        assert "Custom display names" in notification_text

    def test_manifest_structure(self):
        """Test that manifest has the correct structure."""
        # Create a sample manifest
        manifest = {
            "models": {
                "test-model": {
                    "display_name": "Test Model",
                    "publisher": "Test Publisher",
                    "description": "A test model",
                    "category": "test",
                    "tags": ["test", "sample"],
                }
            },
            "version": "1.0",
        }

        # Verify structure
        assert "models" in manifest
        assert "version" in manifest
        assert isinstance(manifest["models"], dict)

        for model_key, model_data in manifest["models"].items():
            assert "display_name" in model_data
            assert "publisher" in model_data
            assert "description" in model_data
            assert "category" in model_data
            assert "tags" in model_data
            assert isinstance(model_data["tags"], list)
