#!/usr/bin/env python3
"""
Test the HuggingFace token validation API functionality.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vllm_cli.validation.token import (  # noqa: E402
    check_token_has_repo_access,
    get_token_info,
    validate_hf_token,
)


def test_token_validation():
    """Test token validation with different scenarios."""
    print("Testing HuggingFace Token Validation API")
    print("=" * 50)

    # Test with invalid token
    print("\n1. Testing with invalid token:")
    is_valid, user_info = validate_hf_token("invalid_token_123")
    print(f"   Result: {'Valid' if is_valid else 'Invalid'}")
    print(f"   User info: {user_info}")

    # Test with empty token
    print("\n2. Testing with empty token:")
    is_valid, user_info = validate_hf_token("")
    print(f"   Result: {'Valid' if is_valid else 'Invalid'}")

    # Test token info formatting
    print("\n3. Testing token info formatting:")
    info = get_token_info("invalid_token")
    print(f"   Info: {info if info else 'No info available'}")

    print("\n" + "=" * 50)
    print("Note: To test with a real token, set HF_TOKEN environment variable")

    # Check if user has a real token to test
    real_token = os.environ.get("HF_TOKEN")
    if real_token:
        print("\n4. Testing with real token from environment:")
        is_valid, user_info = validate_hf_token(real_token)
        if is_valid:
            print("   ✓ Token is valid!")
            print(f"   User: {user_info.get('name', 'Unknown')}")
            print(f"   Email: {user_info.get('email', 'Not provided')}")

            # Test repo access
            print("\n5. Testing repository access:")
            repos_to_test = [
                "gpt2",  # Public model
                "meta-llama/Llama-2-7b-hf",  # Gated model
            ]
            for repo in repos_to_test:
                has_access = check_token_has_repo_access(real_token, repo)
                print(
                    f"   {repo}: {'✓ Access granted' if has_access else '✗ No access'}"
                )
        else:
            print("   ✗ Token is invalid")
    else:
        print("\nTo test with a real token, run:")
        print("  export HF_TOKEN=your_token_here")
        print("  python tests/test_token_api.py")


if __name__ == "__main__":
    test_token_validation()
