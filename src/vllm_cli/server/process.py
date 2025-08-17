#!/usr/bin/env python3
"""
Process management utilities for vLLM servers.

Handles server registry, lifecycle management, and cleanup operations.
"""
import atexit
import logging
from datetime import datetime
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .manager import VLLMServer

logger = logging.getLogger(__name__)

# Global registry of active servers
_active_servers: List["VLLMServer"] = []


def cleanup_servers_on_exit() -> None:
    """Clean up all active servers on program exit."""
    if _active_servers:
        logger.info(f"Cleaning up {len(_active_servers)} active server(s) on exit")
        for server in _active_servers[
            :
        ]:  # Use slice to avoid modification during iteration
            try:
                if server.is_running():
                    logger.info(f"Stopping server {server.model} on port {server.port}")
                    server.stop()
            except Exception as e:
                logger.error(f"Error stopping server on exit: {e}")


# Register cleanup function to run on exit
atexit.register(cleanup_servers_on_exit)


def get_active_servers() -> List["VLLMServer"]:
    """
    Get list of all active vLLM servers.

    Combines CLI-managed servers with externally started vLLM processes
    to provide a comprehensive view of all running servers. Automatically
    cleans up terminated servers from the active list.

    Returns:
        List of active VLLMServer instances including both managed
        and detected external servers
    """

    global _active_servers

    # Clean up terminated servers
    _active_servers = _cleanup_terminated_servers(_active_servers)

    # Detect and add external servers
    _add_external_servers_to_registry()

    return _active_servers


def add_server(server: "VLLMServer") -> None:
    """
    Add a server to the active registry.

    Args:
        server: VLLMServer instance to add
    """
    global _active_servers  # noqa: F824
    if server not in _active_servers:
        _active_servers.append(server)


def remove_server(server: "VLLMServer") -> None:
    """
    Remove a server from the active registry.

    Args:
        server: VLLMServer instance to remove
    """
    global _active_servers  # noqa: F824
    if server in _active_servers:
        _active_servers.remove(server)


def _cleanup_terminated_servers(servers: List["VLLMServer"]) -> List["VLLMServer"]:
    """
    Remove terminated servers from the active list.

    Args:
        servers: List of servers to check

    Returns:
        List of still-active servers
    """
    cleaned_servers = []
    for server in servers:
        if server.process is None:
            # Server was never started properly
            continue
        if hasattr(server.process, "poll") and server.process.poll() is not None:
            # Process has terminated (poll() returns exit code)
            logger.info(
                f"Removing terminated server {server.model} from active list "
                f"(exit code: {server.process.poll()})"
            )
            continue
        # Otherwise keep the server
        cleaned_servers.append(server)

    return cleaned_servers


def _add_external_servers_to_registry() -> None:
    """
    Detect and add externally started vLLM servers to the registry.

    Scans for running vLLM processes and adds them to the active servers
    list if they're not already tracked.
    """
    from .discovery import detect_running_vllm_servers
    from .manager import VLLMServer

    detected_servers = detect_running_vllm_servers()
    tracked_pids = {s.process.pid for s in _active_servers if s.process}

    for server_info in detected_servers:
        if server_info["pid"] not in tracked_pids:
            # Create a VLLMServer instance for external server
            external_server = VLLMServer(
                {
                    "model": server_info.get("model", "unknown"),
                    "port": server_info.get("port", 8000),
                }
            )
            # Mark as external server with a simple object containing pid
            external_server.process = type(
                "ExternalProcess", (), {"pid": server_info["pid"]}
            )()
            external_server.start_time = datetime.now()  # Approximate start time
            _active_servers.append(external_server)


def stop_all_servers() -> int:
    """
    Stop all active vLLM servers.

    Returns:
        Number of servers stopped
    """
    servers = get_active_servers()
    stopped = 0

    for server in servers:
        if server.stop():
            stopped += 1

    return stopped


def find_server_by_port(port: int) -> "VLLMServer":
    """
    Find an active server by port number.

    Args:
        port: Port number to search for

    Returns:
        VLLMServer instance or None if not found
    """
    for server in get_active_servers():
        if server.port == port:
            return server

    return None


def find_server_by_model(model: str) -> "VLLMServer":
    """
    Find an active server by model name.

    Args:
        model: Model name to search for

    Returns:
        VLLMServer instance or None if not found
    """
    for server in get_active_servers():
        if server.model == model:
            return server

    return None
