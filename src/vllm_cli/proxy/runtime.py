#!/usr/bin/env python3
"""
Runtime state management for the proxy server.

This module handles tracking and persisting the runtime state of the proxy server,
including connection details, process information, and configuration.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ProxyRuntimeState:
    """
    Manages the runtime state of the proxy server.

    Tracks running proxy information to enable dynamic connection resolution
    for CLI commands.
    """

    def __init__(self):
        """Initialize the runtime state manager."""
        self.state_dir = Path.home() / ".config" / "vllm-cli"
        self.state_file = self.state_dir / "proxy_runtime.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save_state(
        self,
        host: str,
        port: int,
        pid: Optional[int] = None,
        config_name: Optional[str] = None,
    ) -> bool:
        """
        Save the current proxy runtime state.

        Args:
            host: Proxy server host
            port: Proxy server port
            pid: Process ID of the proxy server
            config_name: Name of the configuration used

        Returns:
            True if state was saved successfully
        """
        try:
            state = {
                "host": host,
                "port": port,
                "pid": pid,
                "start_time": datetime.now().isoformat(),
                "config_name": config_name,
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

            logger.info(f"Saved proxy runtime state: {host}:{port}")
            return True

        except Exception as e:
            logger.error(f"Failed to save proxy runtime state: {e}")
            return False

    def load_state(self) -> Optional[Dict]:
        """
        Load the saved proxy runtime state.

        Returns:
            Dictionary containing runtime state, or None if not found
        """
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)

            # Verify the proxy is still running if PID is available
            if state.get("pid"):
                if not self._is_process_running(state["pid"]):
                    logger.info("Proxy process is no longer running, clearing state")
                    self.clear_state()
                    return None

            return state

        except Exception as e:
            logger.error(f"Failed to load proxy runtime state: {e}")
            return None

    def clear_state(self) -> bool:
        """
        Clear the saved proxy runtime state.

        Returns:
            True if state was cleared successfully
        """
        try:
            if self.state_file.exists():
                self.state_file.unlink()
                logger.info("Cleared proxy runtime state")
            return True

        except Exception as e:
            logger.error(f"Failed to clear proxy runtime state: {e}")
            return False

    def get_connection(self) -> Optional[Tuple[str, int]]:
        """
        Get the connection details for the running proxy.

        Returns:
            Tuple of (host, port) if proxy is running, None otherwise
        """
        state = self.load_state()
        if state:
            return (state["host"], state["port"])
        return None

    def _is_process_running(self, pid: int) -> bool:
        """
        Check if a process with the given PID is running.

        Args:
            pid: Process ID to check

        Returns:
            True if process is running
        """
        try:
            # Send signal 0 to check if process exists
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def get_proxy_connection(
    cli_host: Optional[str] = None,
    cli_port: Optional[int] = None,
    use_env: bool = True,
    use_runtime: bool = True,
) -> Tuple[str, int]:
    """
    Resolve the proxy server connection details.

    Resolution priority:
    1. CLI arguments (if provided)
    2. Environment variables (if use_env=True)
    3. Runtime state file (if use_runtime=True)
    4. Default values (localhost:8000)

    Args:
        cli_host: Host from CLI arguments
        cli_port: Port from CLI arguments
        use_env: Whether to check environment variables
        use_runtime: Whether to check runtime state file

    Returns:
        Tuple of (host, port) for the proxy connection
    """
    # Priority 1: CLI arguments
    if cli_host and cli_port:
        return (cli_host, cli_port)

    # Priority 2: Environment variables
    if use_env:
        env_host = os.getenv("VLLM_PROXY_HOST")
        env_port = os.getenv("VLLM_PROXY_PORT")

        if env_host and env_port:
            try:
                return (env_host, int(env_port))
            except ValueError:
                logger.warning(f"Invalid VLLM_PROXY_PORT value: {env_port}")

    # Priority 3: Runtime state file
    if use_runtime:
        runtime = ProxyRuntimeState()
        connection = runtime.get_connection()
        if connection:
            # Override with CLI args if partially provided
            if cli_host:
                return (cli_host, connection[1])
            if cli_port:
                return (connection[0], cli_port)
            return connection

    # Priority 4: Default values with CLI overrides
    default_host = cli_host or "localhost"
    default_port = cli_port or 8000

    return (default_host, default_port)


def discover_proxy(
    start_port: int = 8000,
    end_port: int = 8010,
    host: str = "localhost",
) -> Optional[Tuple[str, int]]:
    """
    Attempt to discover a running proxy server.

    Args:
        start_port: Starting port to scan
        end_port: Ending port to scan
        host: Host to check

    Returns:
        Tuple of (host, port) if proxy found, None otherwise
    """
    import httpx

    for port in range(start_port, end_port + 1):
        try:
            response = httpx.get(
                f"http://{host}:{port}/proxy/status",
                timeout=1,
            )
            if response.status_code == 200:
                logger.info(f"Discovered proxy server at {host}:{port}")
                return (host, port)
        except Exception:
            continue

    return None
