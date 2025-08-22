#!/usr/bin/env python3
"""
Launcher script for the proxy server subprocess.

This module is run as the entry point for the proxy server subprocess,
parsing command-line arguments and starting the FastAPI/uvicorn server.
"""
import argparse
import json
import logging
import socket
import sys

import uvicorn

from .models import ProxyConfig
from .server import ProxyServer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="vLLM Multi-Model Proxy Server")

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",  # nosec B104
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )
    parser.add_argument(
        "--enable-cors", action="store_true", help="Enable CORS support"
    )
    parser.add_argument(
        "--enable-metrics", action="store_true", help="Enable metrics endpoint"
    )
    parser.add_argument("--log-requests", action="store_true", help="Log all requests")
    parser.add_argument(
        "--config-json",
        type=str,
        required=True,
        help="JSON configuration for the proxy (includes models)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the proxy server launcher."""
    args = parse_args()

    try:
        # Parse the configuration JSON
        config_dict = json.loads(args.config_json)
        proxy_config = ProxyConfig(**config_dict)

        # Create the proxy server
        logger.info(
            f"Initializing proxy server on {proxy_config.host}:{proxy_config.port}"
        )
        proxy_server = ProxyServer(proxy_config)

        # Register already-running models with the router
        # This handles the case where models were started before the proxy
        for model in proxy_config.models:
            if model.enabled:
                # First check if port is in use
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)  # 1 second timeout for connection attempt
                port_result = sock.connect_ex(("localhost", model.port))
                sock.close()

                if port_result == 0:  # Port is in use
                    # Try to verify the server is actually ready with a health check
                    backend_url = f"http://localhost:{model.port}"
                    server_ready = False

                    try:
                        import httpx

                        # Try health endpoint first
                        with httpx.Client() as client:
                            try:
                                response = client.get(
                                    f"{backend_url}/health", timeout=2
                                )
                                server_ready = response.status_code == 200
                            except Exception:
                                # If health endpoint fails, try models endpoint
                                try:
                                    response = client.get(
                                        f"{backend_url}/v1/models", timeout=2
                                    )
                                    server_ready = response.status_code == 200
                                except Exception:
                                    # Server is running but not yet ready
                                    logger.debug(
                                        f"Model '{model.name}' on port "
                                        f"{model.port} not ready yet"
                                    )
                    except ImportError:
                        # httpx not available, assume server is ready if port is open
                        server_ready = True
                        logger.debug(
                            "httpx not available for health check, "
                            "assuming server is ready"
                        )

                    if server_ready:
                        proxy_server.router.add_backend(
                            model.name,
                            backend_url,
                            (
                                model.model_dump()
                                if hasattr(model, "model_dump")
                                else model.dict()
                            ),
                        )
                        logger.info(
                            f"Registered running model '{model.name}' at {backend_url}"
                        )
                    else:
                        logger.info(
                            f"Model '{model.name}' on port {model.port} is "
                            "starting up, will register when ready"
                        )
                else:
                    logger.debug(
                        f"Model '{model.name}' not running on port {model.port}, "
                        "will register when started"
                    )

        # Configure uvicorn logging
        log_config = uvicorn.config.LOGGING_CONFIG.copy()

        # Run the server
        logger.info(f"Starting proxy server on {proxy_config.host}:{proxy_config.port}")
        uvicorn.run(
            proxy_server.app,
            host=proxy_config.host,
            port=proxy_config.port,
            log_level="info",
            log_config=log_config,
            access_log=True,
        )

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse configuration JSON: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start proxy server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
