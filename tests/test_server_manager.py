"""Tests for server management."""

from datetime import datetime
from unittest.mock import Mock, patch

from vllm_cli.server import process
from vllm_cli.server.manager import VLLMServer


class TestVLLMServer:
    """Test VLLMServer functionality."""

    def test_init(self, mock_server_config):
        """Test VLLMServer initialization."""
        server = VLLMServer(mock_server_config)

        assert server.config == mock_server_config
        assert server.model == "meta-llama/Llama-2-7b-hf"
        assert server.port == 8000
        assert server.process is None
        assert server.is_running() is False  # is_running is a method

    def test_build_command(self, mock_server_config):
        """Test building vLLM command from config."""
        server = VLLMServer(mock_server_config)

        command = server._build_command()

        # Command is a list
        command_str = " ".join(command)
        assert "vllm" in command_str
        assert "serve" in command_str  # vLLM uses 'serve' subcommand
        assert "meta-llama/Llama-2-7b-hf" in command_str
        assert "--port" in command_str
        assert "8000" in command_str

    def test_build_command_with_quantization(self):
        """Test building command with quantization."""
        config = {"model": "test-model", "port": 8000, "quantization": "awq"}
        server = VLLMServer(config)

        command = server._build_command()
        command_str = " ".join(command)

        assert "--quantization" in command_str
        assert "awq" in command_str

    def test_build_command_with_tensor_parallel(self):
        """Test building command with tensor parallelism."""
        config = {"model": "test-model", "port": 8000, "tensor_parallel_size": 4}
        server = VLLMServer(config)

        command = server._build_command()
        command_str = " ".join(command)

        assert "--tensor-parallel-size" in command_str
        assert "4" in command_str

    def test_build_command_with_lora_modules(self):
        """Test building command with LoRA modules."""
        config = {
            "model": "base-model",
            "port": 8000,
            "enable_lora": True,
            "lora_modules": "adapter1=/path/to/adapter1 adapter2=/path/to/adapter2",
        }
        server = VLLMServer(config)

        command = server._build_command()
        command_str = " ".join(command)

        assert "--enable-lora" in command_str
        assert "--lora-modules" in command_str
        assert "adapter1=/path/to/adapter1" in command_str

    def test_stop_server(self, mock_server_config):
        """Test stopping a running server."""
        server = VLLMServer(mock_server_config)

        # Mock a running process
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.terminate = Mock()
        mock_process.wait = Mock()
        mock_process.pid = 12345  # Add pid as integer
        server.process = mock_process

        with patch("os.killpg"):  # Mock os.killpg
            with patch("vllm_cli.server.process.remove_server"):
                result = server.stop()

        assert result is True

    def test_health_check_running(self, mock_server_config):
        """Test health check for running server."""
        server = VLLMServer(mock_server_config)

        # Mock running process
        mock_process = Mock()
        mock_process.poll.return_value = None
        server.process = mock_process

        is_healthy = server.health_check()

        assert is_healthy

    def test_health_check_stopped(self, mock_server_config):
        """Test health check for stopped server."""
        server = VLLMServer(mock_server_config)

        # No process
        server.process = None

        is_healthy = server.health_check()

        assert not is_healthy

    def test_get_recent_logs(self, mock_server_config, temp_config_dir):
        """Test getting recent server logs."""
        server = VLLMServer(mock_server_config)

        # Create a mock log file
        log_file = temp_config_dir / "test.log"
        log_content = "Server started\nListening on port 8000\n"
        log_file.write_text(log_content)

        server.log_path = log_file

        # Mock recent logs
        server.recent_logs = ["Server started", "Listening on port 8000"]

        logs = server.get_recent_logs(n=2)

        assert len(logs) == 2
        assert "Server started" in logs[0]
        assert "Listening on port 8000" in logs[1]

    def test_get_status(self, mock_server_config):
        """Test getting server status."""
        server = VLLMServer(mock_server_config)

        # Mock running server
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        server.process = mock_process
        server.start_time = datetime.now()

        status = server.get_status()

        assert status["running"] is True
        assert status["pid"] == 12345
        assert status["port"] == 8000
        assert status["model"] == "meta-llama/Llama-2-7b-hf"
        assert "uptime_seconds" in status

    def test_restart(self, mock_server_config):
        """Test restarting server."""
        server = VLLMServer(mock_server_config)

        # Mock running server
        mock_process = Mock()
        mock_process.poll.return_value = None
        server.process = mock_process

        with patch.object(server, "stop", return_value=True):
            with patch.object(server, "start", return_value=True):
                result = server.restart()

        assert result is True


class TestServerProcess:
    """Test server process management functionality."""

    @patch("vllm_cli.server.process._active_servers", [])
    def test_cleanup_servers_on_exit(self):
        """Test cleanup function for servers on exit."""
        mock_server1 = Mock()
        mock_server1.is_running.return_value = True
        mock_server1.model = "model1"
        mock_server1.port = 8000

        mock_server2 = Mock()
        mock_server2.is_running.return_value = True
        mock_server2.model = "model2"
        mock_server2.port = 8001

        process._active_servers.extend([mock_server1, mock_server2])

        process.cleanup_servers_on_exit()

        mock_server1.stop.assert_called_once()
        mock_server2.stop.assert_called_once()

    @patch("vllm_cli.server.process._active_servers", [])
    def test_cleanup_servers_handles_errors(self):
        """Test cleanup handles errors gracefully."""
        mock_server = Mock()
        mock_server.is_running.return_value = True
        mock_server.stop.side_effect = Exception("Stop failed")
        mock_server.model = "model"
        mock_server.port = 8000

        process._active_servers.append(mock_server)

        # Should not raise exception
        process.cleanup_servers_on_exit()
        mock_server.stop.assert_called_once()

    def test_register_server(self):
        """Test registering a new server."""
        # Save original state
        original_servers = process._active_servers.copy()

        try:
            mock_server = Mock()

            # Register server
            process.add_server(mock_server)

            assert mock_server in process._active_servers
        finally:
            # Restore original state
            process._active_servers = original_servers
