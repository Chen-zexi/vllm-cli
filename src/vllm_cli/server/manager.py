#!/usr/bin/env python3
"""
vLLM server manager module.

Contains the core VLLMServer class responsible for managing individual
vLLM server instances including configuration, lifecycle, and monitoring.
"""
import logging
import os
import shlex
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Dict, List, Optional, TextIO, Union

from ..config import ConfigManager

logger = logging.getLogger(__name__)


class VLLMServer:
    """
    Manages a vLLM server instance with comprehensive lifecycle control.

    This class handles the complete lifecycle of a vLLM server including
    configuration, process management, logging, monitoring, and cleanup.
    It supports both programmatically started servers and external server
    detection and management.

    Attributes:
        config: Server configuration dictionary
        model: Model name being served
        port: Port number the server is listening on
        process: Subprocess or external process reference
        log_file: File handle for server logs
        log_queue: Thread-safe queue for log lines
        log_thread: Background thread for log monitoring
        start_time: Timestamp when server was started
        log_path: Path to the server's log file
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize a vLLM server instance.

        Sets up the server configuration, logging infrastructure, and
        prepares the instance for server lifecycle management.

        Args:
            config: Server configuration dictionary containing model name,
                   port, and other vLLM-specific parameters
        """
        self.config: Dict[str, Any] = config

        # Extract model name - handle both string and dict (LoRA config) formats
        model_config = config.get("model", "unknown")
        if isinstance(model_config, dict):
            # LoRA configuration - extract base model name
            self.model: str = model_config.get("model", "unknown")
        else:
            # Regular string model name
            self.model: str = model_config

        self.port: int = config.get("port", 8000)
        self.process: Optional[Union[subprocess.Popen, Any]] = None
        self.log_file: Optional[TextIO] = None
        self.log_queue: Queue[str] = Queue()
        self.log_thread: Optional[Thread] = None
        self.start_time: Optional[datetime] = None
        self.log_path: Path
        self._recent_logs: List[str] = []

        # Setup logging
        log_dir = Path.home() / ".vllm-cli" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"vllm_{self.model.replace('/', '_')}_{timestamp}.log"

    def start(self) -> bool:
        """
        Start the vLLM server.

        Returns:
            True if server started successfully, False otherwise
        """
        if self.is_running():
            logger.warning(f"Server already running for {self.model}")
            return False

        try:
            # Build command - run directly without conda wrapper
            full_cmd = self._build_command()

            logger.info(f"Starting vLLM server with command: {' '.join(full_cmd)}")

            # Open log file for writing
            self.log_file = open(self.log_path, "w", buffering=1)

            # Set environment to ensure unbuffered output
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            # Add HuggingFace token if configured
            config_manager = ConfigManager()
            hf_token = config_manager.config.get("hf_token")
            if hf_token:
                env["HF_TOKEN"] = hf_token
                env["HUGGING_FACE_HUB_TOKEN"] = hf_token  # Some tools use this variant
                logger.info("HuggingFace token configured for server")

            # Start the process with proper pipe configuration
            # Use start_new_session=True to create a new process group
            # This prevents the child process from receiving Ctrl+C signals
            self.process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,  # Unbuffered
                universal_newlines=True,
                env=env,
                start_new_session=True,  # Create new process group to isolate from parent signals
            )

            # Start log monitoring thread
            self._start_log_monitor()

            # Set start time and add to registry
            self.start_time = datetime.now()
            from .process import add_server

            add_server(self)

            # Return immediately - let the UI monitor the startup
            logger.info(
                f"vLLM server process launched for {self.model} on port {self.port}"
            )
            return True

        except Exception as e:
            logger.error(f"Error starting vLLM server: {e}")
            return False

    def _build_command(self) -> List[str]:
        """
        Build the vLLM command from configuration using the schema.

        Returns:
            List of command arguments
        """
        # Use the new ConfigManager to build CLI args
        config_manager = ConfigManager()

        # Start with the vllm command
        cmd = ["vllm"]

        # Add the arguments built from schema
        cli_args = config_manager.build_cli_args(self.config)
        cmd.extend(cli_args)

        # Handle extra_args if present (backward compatibility)
        if self.config.get("extra_args"):
            self._add_extra_args_to_command(cmd)

        logger.debug(f"Built command: {' '.join(cmd)}")
        return cmd

    def _add_extra_args_to_command(self, cmd: List[str]) -> None:
        """
        Add extra arguments to the vLLM command.

        Args:
            cmd: Command list to extend with extra arguments
        """
        try:
            extra_args = shlex.split(self.config["extra_args"])
            if extra_args:
                logger.info(f"Adding extra arguments: {extra_args}")
                cmd.extend(extra_args)
        except ValueError as e:
            logger.warning(f"Failed to parse extra_args: {e}")

    def _start_log_monitor(self) -> None:
        """
        Start a thread to monitor log output from the vLLM server process.

        Creates a daemon thread that continuously reads stdout/stderr from
        the server process, writes to log files, and queues log lines for
        real-time display in the UI.
        """
        self.log_thread = Thread(target=self._log_monitor_worker, daemon=True)
        self.log_thread.start()

    def _log_monitor_worker(self) -> None:
        """
        Worker function for log monitoring thread.

        Continuously reads log output from the server process and handles
        file writing and queue management for UI display.
        """
        try:
            self.log_queue.put("Starting vLLM server process...")

            while self._should_continue_monitoring():
                if self._process_log_line():
                    continue
                else:
                    # No line read, check if process still running
                    if self.process and self.process.poll() is not None:
                        break
                    time.sleep(0.01)  # Small sleep to prevent busy waiting

        except Exception as e:
            logger.error(f"Error in log monitor: {e}")
            self.log_queue.put(f"Log monitoring error: {e}")

    def _should_continue_monitoring(self) -> bool:
        """
        Check if log monitoring should continue.

        Returns:
            True if monitoring should continue, False otherwise
        """
        return (
            self.is_running()
            and self.process
            and hasattr(self.process, "stdout")
            and self.process.stdout
        )

    def _process_log_line(self) -> bool:
        """
        Process a single log line from the server output.

        Reads one line from the process stdout, writes it to the log file,
        and adds it to the queue for UI display after filtering and formatting.

        Returns:
            True if a line was processed, False if no line was available
        """
        if not (self.process and self.process.stdout):
            return False

        line = self.process.stdout.readline()
        if not line:
            return False

        # Write to log file
        self._write_to_log_file(line)

        # Process for UI display
        line_stripped = line.strip()
        if line_stripped and self._should_display_log_line(line_stripped):
            # Truncate overly long lines
            if len(line_stripped) > 200:
                line_stripped = line_stripped[:197] + "..."

            self.log_queue.put(line_stripped)
            logger.debug(f"Log captured: {line_stripped[:50]}...")

        return True

    def _write_to_log_file(self, line: str) -> None:
        """
        Write a log line to the log file.

        Args:
            line: Log line to write
        """
        if self.log_file and not self.log_file.closed:
            self.log_file.write(line)
            self.log_file.flush()

    def _should_display_log_line(self, line: str) -> bool:
        """
        Determine if a log line should be displayed in the UI.

        Filters out certain system messages that are not useful for users.

        Args:
            line: Stripped log line to check

        Returns:
            True if the line should be displayed, False otherwise
        """
        # Filter out conda activation and system messages
        filtered_messages = [
            "Activated CUDA",
            "Using nvcc from:",
            "Cuda compilation tools",
        ]
        return not any(skip in line for skip in filtered_messages)

    def stop(self) -> bool:
        """
        Stop the vLLM server.

        Returns:
            True if server stopped successfully, False otherwise
        """
        if not self.process:
            return False

        try:
            # Check if this is an external server
            if hasattr(self.process, "terminate"):
                # Normal subprocess
                # Since we created a new session, we need to terminate the entire process group
                try:
                    # Get the process group ID (pgid)
                    # Since we used start_new_session=True, the pgid is the same as the pid
                    pgid = self.process.pid

                    # Send SIGTERM to the entire process group for graceful shutdown
                    logger.info(f"Sending SIGTERM to process group {pgid}")
                    os.killpg(pgid, signal.SIGTERM)

                    # Wait for process to terminate
                    try:
                        self.process.wait(timeout=10)
                        logger.info(f"Process group {pgid} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        # Force kill if not terminated
                        logger.warning(
                            f"Process group {pgid} did not terminate, sending SIGKILL"
                        )
                        os.killpg(pgid, signal.SIGKILL)
                        self.process.wait()
                        logger.info(f"Process group {pgid} force killed")

                except ProcessLookupError:
                    # Process already dead
                    logger.info("Process already terminated")
                except PermissionError as e:
                    # Fall back to terminating just the process
                    logger.warning(
                        f"Permission denied for process group kill: {e}, falling back to process termination"
                    )
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
            elif hasattr(self.process, "pid"):
                # External server - use os.kill
                try:
                    os.kill(self.process.pid, signal.SIGTERM)
                    # Wait a bit
                    time.sleep(2)
                    # Check if still running
                    if self.is_running():
                        os.kill(self.process.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass  # Process already dead

            # Close log file if we have one
            if hasattr(self, "log_file") and self.log_file:
                self.log_file.close()

            # Remove from registry
            from .process import remove_server

            remove_server(self)

            logger.info(f"vLLM server stopped for {self.model}")
            return True

        except Exception as e:
            logger.error(f"Error stopping vLLM server: {e}")
            return False

    def is_running(self) -> bool:
        """
        Check if the server is running.

        Returns:
            True if server is running, False otherwise
        """
        if not self.process:
            return False

        # Check if this is an external server (detected, not started by us)
        if hasattr(self.process, "poll"):
            # Normal subprocess
            return self.process.poll() is None
        elif hasattr(self.process, "pid"):
            # External server - check if PID still exists
            try:
                import psutil

                return psutil.pid_exists(self.process.pid)
            except ImportError:
                # Fallback to os method
                try:
                    os.kill(self.process.pid, 0)
                    return True
                except (OSError, ProcessLookupError):
                    return False

        return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get server status information.

        Returns:
            Dictionary with status information
        """
        status = {
            "model": self.model,
            "port": self.port,
            "running": self.is_running(),
            "pid": self.process.pid if self.process else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "log_path": str(self.log_path) if self.log_path else None,
        }

        if self.is_running() and self.start_time:
            uptime = datetime.now() - self.start_time
            status["uptime_seconds"] = uptime.total_seconds()
            status["uptime_str"] = str(uptime).split(".")[0]

        return status

    def get_recent_logs(self, n: int = 20) -> List[str]:
        """
        Get recent log lines from the server output.

        Retrieves the most recent log lines from either the in-memory queue
        or the log file on disk. This method ensures logs are available even
        after the monitoring thread has processed them.

        Args:
            n: Number of recent lines to return (default: 20)

        Returns:
            List of recent log lines, with the most recent last
        """
        # Update recent logs from queue
        self._update_recent_logs_from_queue()

        # Return from memory if available
        if self._recent_logs:
            return self._recent_logs[-n:]

        # Fallback to reading from file
        return self._read_logs_from_file(n)

    def _update_recent_logs_from_queue(self) -> None:
        """
        Update the recent logs cache from the log queue.

        Drains all available log lines from the queue and adds them
        to the recent logs cache, maintaining a maximum of 100 lines.
        """
        temp_logs = []

        # Get all available logs from queue (don't block)
        while not self.log_queue.empty():
            try:
                log_line = self.log_queue.get_nowait()
                temp_logs.append(log_line)
            except Empty:
                break

        if temp_logs:
            self._recent_logs.extend(temp_logs)
            # Keep only the last 100 logs in memory to prevent unbounded growth
            self._recent_logs = self._recent_logs[-100:]

    def _read_logs_from_file(self, n: int) -> List[str]:
        """
        Read recent log lines from the log file.

        Args:
            n: Number of lines to read

        Returns:
            List of log lines from file
        """
        if not (self.log_path and self.log_path.exists()):
            return []

        try:
            with open(self.log_path, "r") as f:
                file_logs = f.readlines()
                if file_logs:
                    # Get last n lines, strip them
                    logs = [line.strip() for line in file_logs[-n:]]
                    # Update recent logs cache
                    self._recent_logs = logs[-100:]
                    return logs
        except Exception as e:
            logger.debug(f"Error reading log file: {e}")

        return []

    def restart(self) -> bool:
        """
        Restart the server with the same configuration.

        Returns:
            True if restart successful, False otherwise
        """
        logger.info(f"Restarting vLLM server for {self.model}")

        # Stop if running
        if self.is_running():
            self.stop()
            time.sleep(2)  # Wait a bit before restarting

        # Start again
        return self.start()

    def health_check(self) -> bool:
        """
        Perform a health check on the server.

        Returns:
            True if server is healthy, False otherwise
        """
        if not self.is_running():
            return False

        try:
            import requests

            # Try to connect to the server
            response = requests.get(f"http://localhost:{self.port}/health", timeout=5)

            return response.status_code == 200

        except Exception:
            # If requests fails, just check process
            return self.is_running()
