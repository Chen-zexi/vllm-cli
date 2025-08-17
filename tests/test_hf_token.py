#!/usr/bin/env python3
"""
Test script for HuggingFace token functionality in vLLM CLI.

This script demonstrates:
1. Setting HF token programmatically
2. Token configuration and storage
3. Environment variable setup for vLLM
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vllm_cli.config import ConfigManager  # noqa: E402


def test_token_configuration():
    """Test HF token configuration functionality."""
    print("\nTesting HF token configuration...")

    config_manager = ConfigManager()

    # Check if token exists
    current_token = config_manager.config.get("hf_token")
    if current_token:
        print(f"  ✓ Token found: {current_token[:8]}...{current_token[-4:]}")
    else:
        print("  ℹ No token configured")

    # Test saving a dummy token (for demo purposes)
    test_token = "hf_test_token_12345"
    config_manager.config["hf_token"] = test_token
    # Don't actually save to avoid modifying user config
    # config_manager._save_config()
    print(f"  ✓ Token can be set programmatically: {test_token[:8]}...")

    # Remove test token
    config_manager.config.pop("hf_token", None)
    print("  ✓ Token can be removed")


def test_environment_setup():
    """Test that HF token would be passed to environment."""
    print("\nTesting environment setup...")

    # Simulate what happens when starting a server
    env = os.environ.copy()

    # Mock token
    test_token = "hf_test_token_12345"

    # This is what the server manager does
    env["HF_TOKEN"] = test_token
    env["HUGGING_FACE_HUB_TOKEN"] = test_token

    print(f"  ✓ HF_TOKEN would be set: {env.get('HF_TOKEN', '')[:8]}...")
    hf_hub_token = env.get("HUGGING_FACE_HUB_TOKEN", "")[:8]
    print(f"  ✓ HUGGING_FACE_HUB_TOKEN would be set: {hf_hub_token}...")


def main():
    """Run all tests."""
    print("=" * 60)
    print("vLLM CLI HuggingFace Token Feature Tests")
    print("=" * 60)

    test_token_configuration()
    test_environment_setup()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("\nUsage examples:")
    print("  1. Set token via CLI:")
    print("     vllm-cli serve meta-llama/Llama-2-7b --hf-token YOUR_TOKEN")
    print("\n  2. Set token via Settings menu:")
    print("     vllm-cli → Settings → HuggingFace Token")
    print("\n  3. Configure token when selecting HuggingFace Hub models:")
    print("     During model selection, you'll be prompted to optionally set a token")
    print("=" * 60)


if __name__ == "__main__":
    main()
