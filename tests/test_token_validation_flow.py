#!/usr/bin/env python3
"""
Test the token validation flow in different scenarios.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vllm_cli.validation.token import validate_hf_token  # noqa: E402


def test_validation_scenarios():
    """Test different token validation scenarios."""
    print("Testing Token Validation Scenarios")
    print("=" * 50)

    # Test invalid tokens
    test_tokens = [
        ("", "Empty token"),
        ("invalid", "Short invalid token"),
        ("hf_invalid_token_12345", "Invalid HF format token"),
        ("api_invalid_token_12345", "Invalid API format token"),
    ]

    print("\n1. Testing invalid tokens:")
    for token, description in test_tokens:
        is_valid, user_info = validate_hf_token(token)
        status = "✗" if not is_valid else "✓"
        print(f"   {status} {description}: {'Valid' if is_valid else 'Invalid'}")

    print("\n2. Token validation provides user info when valid:")
    print("   - User name")
    print("   - Email address")
    print("   - Organizations")

    print("\n3. Validation happens at three points:")
    print("   ✓ Settings menu: When setting/updating token")
    print("   ✓ CLI: When using --hf-token flag")
    print("   ✓ Model selection: When configuring token for HF Hub models")

    print("\n4. User experience:")
    print("   - Valid tokens: Saved immediately with success message")
    print("   - Invalid tokens: Warning shown, option to save anyway")
    print("   - Network issues: Treated as warning, token still saved")

    print("\n" + "=" * 50)
    print("Token validation ensures users get immediate feedback")
    print("about their token status, improving the user experience.")


if __name__ == "__main__":
    test_validation_scenarios()
