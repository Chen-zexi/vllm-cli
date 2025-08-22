#!/usr/bin/env python3
"""
Subprocess-based proxy server management for consistent log handling.
"""
import logging
import subprocess
import time
from collections import deque
from datetime import datetime
from queue import Queue
from threading import Thread
from typing import List, Optional

from .models import ProxyConfig
from .runtime import ProxyRuntimeState

logger = logging.getLogger(__name__)


class ProxyServerProcess:
    """
    Manages the proxy server as a subprocess for consistent log handling.

    Similar to VLLMServer, runs the proxy in a separate process with
    stdout/stderr captured through pipes for clean log monitoring.
    """

    def __init__(self, config: ProxyConfig):
        """
        Initialize the proxy server process manager.

        Args:
            config: ProxyConfig object with server configuration
        """
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.log_queue: Queue[str] = Queue()
        self.log_thread: Optional[Thread] = None
        self._recent_logs: deque = deque(maxlen=1000)
        self.start_time: Optional[datetime] = None
        self.runtime_state = ProxyRuntimeState()

    def start(self) -> bool:
        """
        Start the proxy server subprocess.

        Returns:
            True if server started successfully
        """
        if self.is_running():
            logger.warning("Proxy server is already running")
            return False

        try:
            # Build command to run the proxy server
            cmd = self._build_command()

            logger.info(f"Starting proxy server: {' '.join(cmd)}")

            # Start the process with output capture
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,  # Unbuffered
                universal_newlines=True,
                start_new_session=True,  # Create new process group
            )

            # Start log monitoring thread
            self._start_log_monitor()

            # Set start time
            self.start_time = datetime.now()

            # Give it a moment to start
            time.sleep(1)

            # Check if process is still running
            if not self.is_running():
                logger.error(
                    "Proxy server process terminated immediately after starting"
                )
                return False

            # Save runtime state
            self.runtime_state.save_state(
                host=self.config.host,
                port=self.config.port,
                pid=self.process.pid if self.process else None,
            )

            logger.info(
                f"Proxy server started on {self.config.host}:{self.config.port}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start proxy server: {e}")
            return False

    def stop(self) -> None:
        """Stop the proxy server process."""
        if self.process:
            logger.info("Stopping proxy server process...")
            try:
                # Try graceful termination first
                self.process.terminate()

                # Wait up to 5 seconds for graceful shutdown
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    logger.warning("Proxy server didn't stop gracefully, forcing kill")
                    self.process.kill()
                    self.process.wait(timeout=2)

                logger.info("Proxy server stopped")
            except Exception as e:
                logger.error(f"Error stopping proxy server: {e}")
            finally:
                self.process = None
                # Clear runtime state
                self.runtime_state.clear_state()

    def is_running(self) -> bool:
        """
        Check if the proxy server process is running.

        Returns:
            True if the process is active
        """
        return self.process is not None and self.process.poll() is None

    def get_recent_logs(self, n: int = 50) -> List[str]:
        """
        Get recent log lines from the proxy server.

        Args:
            n: Number of recent lines to return

        Returns:
            List of recent log lines
        """
        # Update recent logs from queue
        self._update_recent_logs_from_queue()

        # Return from memory
        return list(self._recent_logs)[-n:] if self._recent_logs else []

    def _build_command(self) -> List[str]:
        """
        Build the command to start the proxy server.

        Returns:
            List of command arguments
        """
        # Use Python to run the server launcher module
        cmd = [
            "python",
            "-m",
            "vllm_cli.proxy.server_launcher",
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
        ]

        # Add boolean flags
        if self.config.enable_cors:
            cmd.append("--enable-cors")
        if self.config.enable_metrics:
            cmd.append("--enable-metrics")
        if self.config.log_requests:
            cmd.append("--log-requests")

        # Pass the configuration as JSON (for models and other complex data)
        import json

        config_json = json.dumps(self.config.model_dump())
        cmd.extend(["--config-json", config_json])

        return cmd

    def _start_log_monitor(self) -> None:
        """Start a thread to monitor log output from the proxy server."""
        self.log_thread = Thread(target=self._log_monitor_worker, daemon=True)
        self.log_thread.start()

    def _log_monitor_worker(self) -> None:
        """Worker function for log monitoring thread."""
        try:
            while self.process and self.process.poll() is None:
                if self.process.stdout:
                    line = self.process.stdout.readline()
                    if line:
                        line = line.rstrip()
                        # Remove misleading "(Press CTRL+C to quit)" from uvicorn logs
                        if "(Press CTRL+C to quit)" in line:
                            line = line.replace(" (Press CTRL+C to quit)", "")
                        # Add to queue for UI
                        self.log_queue.put(line)
                        # Add to recent logs buffer
                        self._recent_logs.append(line)
                    else:
                        # No output available, small sleep
                        time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in log monitor thread: {e}")

    def _update_recent_logs_from_queue(self) -> None:
        """Update recent logs buffer from the queue."""
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
                # Already in _recent_logs from monitor thread
            except Exception:
                break

    @property
    def port(self) -> int:
        """Get the port number."""
        return self.config.port

    @property
    def host(self) -> str:
        """Get the host address."""
        return self.config.host
