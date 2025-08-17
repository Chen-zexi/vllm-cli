#!/usr/bin/env python3
"""
Server monitoring utilities.

Provides health checking and monitoring capabilities for vLLM servers.
"""
import logging
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from .manager import VLLMServer

logger = logging.getLogger(__name__)


def perform_health_check(server: "VLLMServer") -> Dict[str, Any]:
    """
    Perform a comprehensive health check on a server.

    Args:
        server: VLLMServer instance to check

    Returns:
        Dictionary with health check results
    """
    health_status = {
        "server_id": f"{server.model}:{server.port}",
        "is_running": server.is_running(),
        "process_healthy": False,
        "http_healthy": False,
        "log_status": "unknown",
        "overall_status": "unhealthy",
    }

    if not health_status["is_running"]:
        health_status["log_status"] = "process_not_running"
        return health_status

    # Check process health
    health_status["process_healthy"] = _check_process_health(server)

    # Check HTTP endpoint health
    health_status["http_healthy"] = _check_http_health(server)

    # Check log status
    health_status["log_status"] = _check_log_status(server)

    # Determine overall status
    if health_status["process_healthy"] and health_status["http_healthy"]:
        health_status["overall_status"] = "healthy"
    elif health_status["process_healthy"]:
        health_status["overall_status"] = "starting"
    else:
        health_status["overall_status"] = "unhealthy"

    return health_status


def _check_process_health(server: "VLLMServer") -> bool:
    """Check if the server process is healthy."""
    try:
        if not server.process:
            return False

        # Check if process is still running
        if hasattr(server.process, "poll"):
            return server.process.poll() is None
        elif hasattr(server.process, "pid"):
            try:
                import psutil

                return psutil.pid_exists(server.process.pid)
            except ImportError:
                import os

                try:
                    os.kill(server.process.pid, 0)
                    return True
                except (OSError, ProcessLookupError):
                    return False

        return False
    except Exception as e:
        logger.debug(f"Error checking process health: {e}")
        return False


def _check_http_health(server: "VLLMServer") -> bool:
    """Check if the server HTTP endpoint is responding."""
    try:
        import requests

        # Try to connect to the health endpoint
        response = requests.get(f"http://localhost:{server.port}/health", timeout=5)

        return response.status_code == 200

    except ImportError:
        logger.debug("requests module not available for HTTP health check")
        return False
    except Exception as e:
        logger.debug(f"HTTP health check failed: {e}")
        return False


def _check_log_status(server: "VLLMServer") -> str:
    """Check the status based on recent log messages."""
    try:
        recent_logs = server.get_recent_logs(10)

        if not recent_logs:
            return "no_logs"

        # Look for error indicators in recent logs
        error_indicators = ["error", "failed", "exception", "traceback"]
        for log_line in recent_logs[-5:]:  # Check last 5 lines
            if any(indicator in log_line.lower() for indicator in error_indicators):
                return "errors_detected"

        # Look for startup completion indicators
        startup_indicators = [
            "application startup complete",
            "uvicorn running",
            "started server process",
        ]
        for log_line in recent_logs:
            if any(indicator in log_line.lower() for indicator in startup_indicators):
                return "startup_complete"

        return "starting"

    except Exception as e:
        logger.debug(f"Error checking log status: {e}")
        return "unknown"


def get_server_metrics(server: "VLLMServer") -> Dict[str, Any]:
    """
    Get performance metrics for a server.

    Args:
        server: VLLMServer instance

    Returns:
        Dictionary with server metrics
    """
    metrics = {
        "server_id": f"{server.model}:{server.port}",
        "uptime_seconds": 0,
        "memory_usage_mb": 0,
        "cpu_percent": 0,
        "gpu_memory_used": 0,
        "gpu_memory_total": 0,
    }

    try:
        if server.start_time:
            from datetime import datetime

            uptime = datetime.now() - server.start_time
            metrics["uptime_seconds"] = uptime.total_seconds()

        # Get process metrics if psutil is available
        if server.process and hasattr(server.process, "pid"):
            metrics.update(_get_process_metrics(server.process.pid))

    except Exception as e:
        logger.debug(f"Error getting server metrics: {e}")

    return metrics


def _get_process_metrics(pid: int) -> Dict[str, Any]:
    """Get process-specific metrics."""
    metrics = {}

    try:
        import psutil

        process = psutil.Process(pid)

        # Memory usage
        memory_info = process.memory_info()
        metrics["memory_usage_mb"] = memory_info.rss / 1024 / 1024

        # CPU usage
        metrics["cpu_percent"] = process.cpu_percent()

    except ImportError:
        logger.debug("psutil not available for process metrics")
    except Exception as e:
        logger.debug(f"Error getting process metrics: {e}")

    return metrics


def monitor_all_servers() -> List[Dict[str, Any]]:
    """
    Monitor all active servers and return their status.

    Returns:
        List of server status dictionaries
    """
    from .process import get_active_servers

    server_statuses = []

    for server in get_active_servers():
        try:
            health_status = perform_health_check(server)
            metrics = get_server_metrics(server)

            status = {**health_status, **metrics, "server": server}
            server_statuses.append(status)

        except Exception as e:
            logger.error(f"Error monitoring server {server.model}:{server.port}: {e}")
            server_statuses.append(
                {
                    "server_id": f"{server.model}:{server.port}",
                    "overall_status": "error",
                    "error_message": str(e),
                    "server": server,
                }
            )

    return server_statuses
