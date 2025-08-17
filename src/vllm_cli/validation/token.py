#!/usr/bin/env python3
"""
HuggingFace token validation utilities.

Provides functions to validate HuggingFace tokens using the official API.
"""
import logging
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# HuggingFace API endpoint for token validation
HF_WHOAMI_ENDPOINT = "https://huggingface.co/api/whoami-v2"


def validate_hf_token(
    token: str, timeout: int = 10
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Validate a HuggingFace token using the official API.

    Args:
        token: HuggingFace token to validate
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_valid, user_info_dict or None)
        - is_valid: True if token is valid, False otherwise
        - user_info: Dictionary with user information if valid, None otherwise
    """
    if not token:
        return False, None

    try:
        response = requests.get(
            HF_WHOAMI_ENDPOINT,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )

        if response.status_code == 200:
            user_info = response.json()
            user_name = user_info.get("name", "Unknown")
            logger.info(f"Token validated successfully for user: {user_name}")
            return True, user_info
        elif response.status_code == 401:
            logger.warning("Token validation failed: Invalid or expired token")
            return False, None
        else:
            logger.warning(
                f"Token validation failed with status {response.status_code}"
            )
            return False, None

    except requests.exceptions.Timeout:
        logger.error("Token validation timed out")
        return False, None
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to HuggingFace API")
        return False, None
    except Exception as e:
        logger.error(f"Unexpected error during token validation: {e}")
        return False, None


def get_token_info(token: str) -> Optional[str]:
    """
    Get a formatted string with token information.

    Args:
        token: HuggingFace token

    Returns:
        Formatted string with user info or None if invalid
    """
    is_valid, user_info = validate_hf_token(token)

    if not is_valid or not user_info:
        return None

    info_parts = [
        f"User: {user_info.get('name', 'Unknown')}",
        f"Email: {user_info.get('email', 'Not provided')}",
    ]

    if user_info.get("orgs"):
        org_names = [org.get("name", "Unknown") for org in user_info.get("orgs", [])]
        info_parts.append(f"Organizations: {', '.join(org_names)}")

    return " | ".join(info_parts)


def check_token_has_repo_access(token: str, repo_id: str) -> bool:
    """
    Check if a token has access to a specific repository.

    This is a simple check that attempts to get repo info with the token.

    Args:
        token: HuggingFace token
        repo_id: Repository ID (e.g., "meta-llama/Llama-2-7b")

    Returns:
        True if token has access, False otherwise
    """
    try:
        # Try to access the model info endpoint
        response = requests.get(
            f"https://huggingface.co/api/models/{repo_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )

        # 200 means we have access
        # 403 means the model exists but we don't have access
        # 404 means the model doesn't exist
        return response.status_code == 200

    except Exception as e:
        logger.error(f"Failed to check repo access: {e}")
        return False
