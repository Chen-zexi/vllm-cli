#!/usr/bin/env python3
"""
Launcher script for the proxy server subprocess.

This module is run as the entry point for the proxy server subprocess,
parsing command-line arguments and starting the FastAPI/uvicorn server.
"""
import argparse
import json
import logging
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

        # Register models with the router if they are configured
        for model in proxy_config.models:
            if model.enabled:
                backend_url = f"http://localhost:{model.port}"
                proxy_server.router.add_backend(
                    model.name,
                    backend_url,
                    (
                        model.model_dump()
                        if hasattr(model, "model_dump")
                        else model.dict()
                    ),
                )
                logger.info(f"Registered model '{model.name}' at {backend_url}")

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
