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


def verify_pending_models_limited(proxy_server):
    """
    Limited verification of pending models - stops when done or after max time.

    This function runs for a limited time and stops when:
    - All pending models are verified (activated or failed)
    - Maximum time limit is reached (5 minutes)
    - No progress is made after multiple attempts
    """
    import time

    import httpx

    max_total_time = 300  # 5 minutes maximum
    check_interval = 5  # Start with 5 seconds
    max_interval = 30  # Max interval between checks
    start_time = time.time()
    no_progress_count = 0
    max_no_progress = 5  # Stop after 5 checks with no progress

    logger.info("Starting limited background verification of pending models")

    while (time.time() - start_time) < max_total_time:
        # First check if we have any pending models
        try:
            # Query the registry status to check pending count
            with httpx.Client() as client:
                # Get registry status first
                registry_response = client.get(
                    f"http://localhost:{proxy_server.config.port}/proxy/registry",
                    timeout=5,
                )

                if registry_response.status_code == 200:
                    registry_data = registry_response.json()
                    models = registry_data.get("models", [])

                    # Count pending models
                    pending_count = sum(
                        1
                        for model in models
                        if model.get("registration_status") == "pending"
                    )

                    if pending_count == 0:
                        logger.info(
                            "No pending models remaining, stopping verification"
                        )
                        return

                    logger.debug(f"{pending_count} models still pending verification")
        except Exception as e:
            logger.debug(f"Failed to check registry status: {e}")

        # Wait before calling refresh
        time.sleep(check_interval)

        try:
            # Call the proxy's refresh endpoint to verify pending models
            with httpx.Client() as client:
                response = client.post(
                    f"http://localhost:{proxy_server.config.port}/proxy/refresh_models",
                    timeout=10,
                )

                if response.status_code == 200:
                    result = response.json()
                    summary = result.get("summary", {})

                    if summary.get("registered", 0) > 0:
                        logger.info(
                            f"Background verification: "
                            f"{summary['registered']} models activated"
                        )
                        # Reset interval and no-progress counter on success
                        check_interval = 5
                        no_progress_count = 0
                    else:
                        # No new registrations, increase interval
                        check_interval = min(check_interval * 1.5, max_interval)
                        no_progress_count += 1

                        if no_progress_count >= max_no_progress:
                            logger.info(
                                "No progress after multiple attempts, "
                                "stopping verification"
                            )
                            return

        except Exception as e:
            logger.debug(f"Background verification error: {e}")
            # Increase interval on error
            check_interval = min(check_interval * 1.5, max_interval)
            no_progress_count += 1

    logger.info(
        f"Background verification stopped after {max_total_time} seconds (timeout)"
    )


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

        # Pre-register all configured models with PENDING status
        # This is the ONLY time we use the config - at startup
        models_to_verify = []

        for model in proxy_config.models:
            if model.enabled:
                # Pre-register the model in the registry with PENDING status
                gpu_ids = model.gpu_ids if hasattr(model, "gpu_ids") else []

                # Pre-register in the runtime registry
                success = proxy_server.registry.pre_register(
                    port=model.port,
                    gpu_ids=gpu_ids,
                    config_name=model.name,
                )

                if success:
                    logger.info(
                        f"Pre-registered model '{model.name}' on port {model.port} "
                        "with PENDING status"
                    )
                    models_to_verify.append(model)
                else:
                    logger.warning(
                        f"Failed to pre-register model '{model.name}' "
                        f"on port {model.port}"
                    )

        # If we have models to verify, start a background thread
        if models_to_verify:
            # Immediately try to verify once
            try:
                import httpx

                with httpx.Client() as client:
                    response = client.post(
                        f"http://localhost:{proxy_config.port}/proxy/refresh_models",
                        timeout=5,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        summary = result.get("summary", {})
                        if summary.get("registered", 0) > 0:
                            logger.info(
                                f"Initial verification: "
                                f"{summary['registered']} models activated"
                            )
            except Exception as e:
                logger.debug(f"Initial verification attempt failed: {e}")

            # Start background thread for continuous verification
            import threading

            verify_thread = threading.Thread(
                target=verify_pending_models_limited,
                args=(proxy_server,),
                daemon=True,
            )
            verify_thread.start()
            logger.info(
                f"Started limited background thread to verify {len(models_to_verify)} "
                "pre-registered models (max 5 minutes)"
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
