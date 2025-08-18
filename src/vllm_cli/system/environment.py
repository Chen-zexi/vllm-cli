#!/usr/bin/env python3
"""
System environment utilities.

Provides functions for working with system environments,
package managers, and environment detection.
"""
import logging
import subprocess
from typing import List

logger = logging.getLogger(__name__)


def get_conda_envs() -> List[str]:
    """
    Get list of available conda environments.

    Returns:
        List of environment names
    """
    envs = []

    try:
        result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, check=True
        )

        for line in result.stdout.split("\n"):
            if line and not line.startswith("#"):
                parts = line.split()
                if parts:
                    envs.append(parts[0])

    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not get conda environments")

    return envs
